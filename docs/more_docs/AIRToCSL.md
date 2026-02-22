# AIR-to-CSL: Lowering AIR Dialect to Cerebras CSL

This document describes the `air-to-csl` pass, which lowers AMD's AIR (Accelerator Interface for Reconfigurable computing) MLIR dialect to Cerebras CSL (Cerebras Software Language) text files. The pass bridges two spatial computing models: AIR's hierarchical launch/segment/herd abstraction originally targeting AMD Versal AI Engines, and Cerebras' layout/PE-program/runtime abstraction targeting the Wafer-Scale Engine.

## Motivation

AIR provides a hardware-agnostic spatial programming model with three hierarchy levels (`air.launch`, `air.segment`, `air.herd`) that express host orchestration, spatial PE allocation, and per-PE kernel code respectively. These three levels map naturally onto the three components of a Cerebras CSL program:

| AIR Construct | CSL Output | Role |
|---|---|---|
| `air.launch` | `run.py` | Host-side orchestration: load, run, data transfers, stop |
| `air.segment` | `layout.csl` | Spatial topology: PE grid dimensions, tile-to-code assignment |
| `air.herd` body | `pe_program.csl` | Per-PE kernel: memory declarations, compute functions, exports |

This makes AIR a viable intermediate representation for targeting Cerebras hardware, reusing the existing front-end pipeline (linalg -> scf.parallel -> air.herd) while swapping only the back-end.

## How it works internally

### Pass registration and infrastructure

The pass is registered as a core conversion pass (always available, not gated behind `AIR_ENABLE_AIE` or `AIR_ENABLE_GPU`). It plugs into MLIR's pass infrastructure through the standard TableGen pipeline:

1. **`Passes.td`** declares the pass with `def AIRToCSL : Pass<"air-to-csl", "ModuleOp">` and a single `output-dir` string option.
2. **TableGen** generates `Passes.h.inc` containing the `AIRToCSLBase<DerivedT>` CRTP template class that provides `clOutputDir`, `runOnOperation()` dispatching, and pass metadata.
3. **`PassDetail.h`** defines `GEN_PASS_DEF_AIRTOCSL` to instantiate the template.
4. **`Passes.cpp`** calls `registerAIRToCSL()` during `registerConversionPasses()`.
5. **`CMakeLists.txt`** adds `AIRToCSLPass.cpp` to the unconditional `CONVERSION_SOURCES`.

The pass class inherits from the generated base:

```cpp
class AIRToCSLPass : public air::impl::AIRToCSLBase<AIRToCSLPass> {
  void runOnOperation() override {
    auto module = getOperation();
    llvm::sys::fs::create_directories(clOutputDir);
    CSLEmitter emitter(clOutputDir);
    emitter.emit(module);
  }
};
```

### The CSLEmitter: IR walk and file generation

Unlike `air-to-aie` which lowers AIR ops into another MLIR dialect (the AIE dialect), `air-to-csl` is a **text emitter**. It walks the MLIR module, collects structural metadata, and writes three text files using `llvm::raw_fd_ostream`. No new MLIR dialect is introduced.

#### Phase 1: Structure collection

The emitter walks the module top-down through the AIR hierarchy:

```
module.walk(LaunchOp)
  -> launch.walk(SegmentOp)
      -> segment.walk(HerdOp)
```

For each `air.herd`, it records:
- **Grid dimensions** via `herd.getNumCols()` and `herd.getNumRows()`. These are extracted from the herd's size operands (e.g., `%c2 = arith.constant 2` bound to `in (%sx=%c2, %sy=%c2)`).
- **Kernel arguments** via `herd.getKernelArguments()`. Each `memref<NxTy>` argument becomes an exported array in both `pe_program.csl` and `layout.csl`.

If no `air.launch`/`air.segment` wrapper exists (bare herd), the emitter synthesizes an implicit single segment.

#### Phase 2: Exported array metadata

For every memref-typed kernel argument on the herd, the emitter creates an `ExportedArray` record:

```cpp
struct ExportedArray {
  std::string name;       // "arg_0", "arg_1", ...
  std::string cslType;    // "[*]f32"
  bool mutable_;          // host read/write access
  int64_t numElements;    // flat product of static shape dims
  Type elemType;          // MLIR element type for numpy dtype selection
};
```

These drive export declarations in all three output files.

#### Phase 3: layout.csl emission

The `emitLayoutCSL()` method produces the spatial configuration:

```csl
const memcpy = @import_module("<memcpy/get_params>", .{ .width = W, .height = H });

layout {
  @set_rectangle(W, H);
  @set_tile_code(col, row, "pe_program.csl", .{ .memcpy_params = memcpy.get_params(col) });
  // ... for each (col, row) in the grid
  @export_name("arg_0", [*]f32, true);
  @export_name("init_and_compute", fn()void);
}
```

The segment's width/height (from `getNumCols()`/`getNumRows()` on segment or its first herd) map directly to `@set_rectangle`. Each tile in the 2D grid gets a `@set_tile_code` call pointing to the same `pe_program.csl`, parameterized with memcpy parameters keyed by column index.

#### Phase 4: pe_program.csl emission

The `emitPEProgramCSL()` method generates per-PE kernel code in three sub-phases:

**a) Global declarations.** Each memref kernel argument becomes a global CSL array:

```
memref<1024xf32> herd arg  ->  var arg_0: [1024]f32;
```

Multi-dimensional shapes are flattened: `memref<4x6xf32>` becomes `[24]f32`. Each exported array gets a pointer constant and a `comptime` export:

```csl
const arg_0_ptr: [*]f32 = &arg_0;
comptime { @export_symbol(arg_0_ptr, "arg_0"); }
```

Local `memref.alloc` ops within the herd body become additional global arrays (CSL has no stack allocation; all PE memory is global).

**b) Compute function.** The herd body is emitted into a `fn compute() void` by walking each operation:

| MLIR Operation | CSL Emission |
|---|---|
| `arith.constant 0 : index` | `const c_0: i32 = 0;` |
| `memref.load %buf[%idx]` | `const ld_1: f32 = buf[c_0];` |
| `arith.addf %a, %b` | `var v_2: f32 = ld_1 + ld_2;` |
| `arith.mulf %a, %b` | `var v_3: f32 = a * b;` |
| `memref.store %v, %buf[%idx]` | `buf[c_0] = v_2;` |
| `scf.for %iv = %lb to %ub` | `var i_4: i32 = lb; while (i_4 < ub) : (i_4 += 1) { ... }` |
| `air.execute { ... }` | Inlines the body directly |
| `air.dma_memcpy_nd` | `// TODO: data movement` |

The `emitOp()` dispatcher uses LLVM's `dyn_cast` chain. Each MLIR SSA value is mapped to a fresh CSL variable name via a `DenseMap<Value, std::string>` dictionary. The `freshName()` counter ensures unique names (`c_0`, `ld_1`, `v_2`, ...).

Multi-dimensional `memref.load/store` indices are linearized to row-major flat indices at emit time using the static shape:
```
%v = memref.load %A[%i, %j] : memref<4x6xf32>
  -> const ld_5: f32 = A[i_3*6 + j_4];
```

Tile IDs (`%tx`, `%ty` from `air.herd tile(%tx, %ty)`) are bound to constant `"0"` since the emitter currently generates a single program for all PEs.

**c) Wrapper and exports.** The compute function is wrapped in `init_and_compute()` which calls `sys_mod.unblock_cmd_stream()` after computation -- a Cerebras requirement to allow subsequent memcpy commands from the host.

#### Phase 5: run.py emission

The `emitRunPy()` method generates a Python host script using the Cerebras `SdkRuntime` API:

1. **Boilerplate**: argparse for `--name` (compiled output dir) and `--cmaddr` (system address).
2. **Symbol resolution**: `runner.get_id('arg_0')` for each exported array.
3. **Load/run**: `runner.load()` then `runner.run()`.
4. **H2D transfers**: Commented-out `memcpy_h2d` templates for each mutable array, with correct element counts (`W * H * numElements`) and `MemcpyDataType` selection based on element bitwidth.
5. **Compute launch**: `runner.launch('init_and_compute', nonblock=False)`.
6. **D2H transfers**: Active `memcpy_d2h` calls for each exported array.
7. **Stop**: `runner.stop()`.

### Type mapping

| MLIR Type | CSL Type | NumPy dtype |
|---|---|---|
| `f32` | `f32` | `np.float32` |
| `f16` | `f16` | `np.float16` |
| `i32` | `i32` | `np.int32` |
| `i16` | `i16` | `np.int32` |
| `index` | `i32` | `np.int32` |

## Usage

```bash
air-opt input.mlir -air-to-csl="output-dir=./output"
```

This writes `layout.csl`, `pe_program.csl`, and `run.py` into `./output/`. To then compile and run on Cerebras hardware:

```bash
cslc --arch=wse3 ./output/layout.csl --fabric-dims=9,4 \
  --fabric-offsets=4,1 -o out --memcpy --channels 1
cs_python ./output/run.py --name out
```

## Example

**Input MLIR** (a 2x2 herd performing element-wise multiply):

```mlir
func.func @gemv(%A: memref<24xf32>, %x: memref<6xf32>, %y: memref<4xf32>) {
  %c1 = arith.constant 1 : index
  air.launch (%tx) in (%sx=%c1) args(%a0=%A, %a1=%x, %a2=%y)
      : memref<24xf32>, memref<6xf32>, memref<4xf32> {
    air.segment @seg0 args(%s0=%a0, %s1=%a1, %s2=%a2)
        : memref<24xf32>, memref<6xf32>, memref<4xf32> {
      %c2 = arith.constant 2 : index
      air.herd @herd0 tile(%htx, %hty) in (%hsx=%c2, %hsy=%c2)
          args(%h0=%s0, %h1=%s1, %h2=%s2)
          : memref<24xf32>, memref<6xf32>, memref<4xf32> {
        %zero = arith.constant 0 : index
        %v = memref.load %h1[%zero] : memref<6xf32>
        %w = memref.load %h0[%zero] : memref<24xf32>
        %prod = arith.mulf %v, %w : f32
        memref.store %prod, %h2[%zero] : memref<4xf32>
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
```

**Generated layout.csl:**

```csl
const memcpy = @import_module("<memcpy/get_params>", .{ .width = 2, .height = 2 });

layout {
  @set_rectangle(2, 2);

  @set_tile_code(0, 0, "pe_program.csl", .{ .memcpy_params = memcpy.get_params(0) });
  @set_tile_code(1, 0, "pe_program.csl", .{ .memcpy_params = memcpy.get_params(1) });
  @set_tile_code(0, 1, "pe_program.csl", .{ .memcpy_params = memcpy.get_params(0) });
  @set_tile_code(1, 1, "pe_program.csl", .{ .memcpy_params = memcpy.get_params(1) });

  @export_name("arg_0", [*]f32, false);
  @export_name("arg_1", [*]f32, false);
  @export_name("arg_2", [*]f32, false);
  @export_name("init_and_compute", fn()void);
}
```

**Generated pe_program.csl:**

```csl
param memcpy_params: comptime_struct;
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

var arg_0: [24]f32;
var arg_1: [6]f32;
var arg_2: [4]f32;
const arg_0_ptr: [*]f32 = &arg_0;
const arg_1_ptr: [*]f32 = &arg_1;
const arg_2_ptr: [*]f32 = &arg_2;

fn compute() void {
  const c_0: i32 = 0;
  const ld_1: f32 = arg_1[c_0];
  const ld_2: f32 = arg_0[c_0];
  var v_3: f32 = ld_1 * ld_2;
  arg_2[c_0] = v_3;
}

fn init_and_compute() void {
  compute();
  sys_mod.unblock_cmd_stream();
}

comptime {
  @export_symbol(arg_0_ptr, "arg_0");
  @export_symbol(arg_1_ptr, "arg_1");
  @export_symbol(arg_2_ptr, "arg_2");
  @export_symbol(init_and_compute);
}
```

## Files changed

| File | Change |
|---|---|
| `mlir/include/air/Conversion/Passes.td` | Added `AIRToCSL` pass definition with `output-dir` option |
| `mlir/include/air/Conversion/PassDetail.h` | Added `#define GEN_PASS_DEF_AIRTOCSL` in the core (non-AIE-gated) section |
| `mlir/include/air/Conversion/Passes.h` | Added `#include "air/Conversion/AIRToCSLPass.h"` |
| `mlir/include/air/Conversion/AIRToCSLPass.h` | **New** -- header declaring `createAIRToCSLPass()` factory |
| `mlir/lib/Conversion/AIRToCSLPass.cpp` | **New** -- 749-line pass implementation with `CSLEmitter` |
| `mlir/lib/Conversion/Passes.cpp` | Added registration macro and `registerAIRToCSL()` call |
| `mlir/lib/Conversion/CMakeLists.txt` | Added `AIRToCSLPass.cpp` to core `CONVERSION_SOURCES` |
| `mlir/test/Conversion/AIRToCSL/basic.mlir` | **New** -- 1x1 herd vector-add test |
| `mlir/test/Conversion/AIRToCSL/gemv.mlir` | **New** -- 2x2 herd GEMV test |

## Current limitations and future work

- **Single PE program**: all tiles in the grid run the same `pe_program.csl`. Per-tile specialization based on `%tx`/`%ty` tile IDs is not yet implemented.
- **No inter-PE communication**: `air.channel` and `air.dma_memcpy_nd` are emitted as TODO comments rather than CSL routing/color/task constructs.
- **No DSD (Data Structure Descriptor) usage**: memory accesses use scalar array indexing rather than CSL's efficient DSD-based bulk operations.
- **Static shapes only**: dynamic memref dimensions are unsupported.
- **Host data flow**: `run.py` H2D transfers are generated as commented-out templates; the user must fill in actual data initialization.
