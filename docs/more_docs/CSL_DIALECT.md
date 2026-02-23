# CSL Dialect Reference

The `csl` dialect provides a structured MLIR representation of Cerebras Systems Language (CSL) programs targeting the Wafer-Scale Engine (WSE). It mirrors the key CSL constructs — layout blocks, PE tile assignment, color-based routing, DSD-driven data movement, task-based kernels, and host runtime interfaces — as first-class MLIR operations with verification and type safety.

This dialect is part of **Phase 2** of the [design plan](design_air_to_csl.md) (Section 9.2): building `csl.*` MLIR ops that mirror CSL language constructs, enabling MLIR-level verification and optimization before text emission via `air-translate --csl-emit`.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Dialect Architecture](#2-dialect-architecture)
3. [Types](#3-types)
4. [Enumerations](#4-enumerations)
5. [Operations by Category](#5-operations-by-category)
   - [5.1 Layout](#51-layout)
   - [5.2 Placement](#52-placement)
   - [5.3 Routing](#53-routing)
   - [5.4 Kernel](#54-kernel)
   - [5.5 Data Movement](#55-data-movement)
   - [5.6 Runtime](#56-runtime)
6. [Building a Complete Program](#6-building-a-complete-program)
7. [Testing](#7-testing)
8. [Build System Integration](#8-build-system-integration)
9. [Design Rationale](#9-design-rationale)

---

## 1. Motivation

The Phase 1 text emitter (`AIRToCSLPass.cpp`) walks AIR IR and directly emits CSL source files. While functional, it has fundamental limitations:

- No MLIR-level verification of the generated code structure.
- No composability with optimization passes between AIR lowering and text emission.
- The emitter is a long `dyn_cast` chain that grows unmanageably as more CSL constructs are needed.
- Cannot represent or optimize inter-PE communication at the IR level.

The CSL dialect addresses these by representing CSL constructs as proper MLIR operations that participate in MLIR's verification, transformation, and analysis infrastructure.

---

## 2. Dialect Architecture

All operations share the single `csl` MLIR dialect namespace (`xilinx::csl` in C++) but are organized into six **conceptual sub-categories** across separate TableGen files for clarity:

| Category | TableGen File | Purpose |
|---|---|---|
| Layout | `CSLLayoutOps.td` | Spatial grid configuration |
| Placement | `CSLPlacementOps.td` | Tile-to-code mapping |
| Routing | `CSLRoutingOps.td` | Color declaration and routing |
| Kernel | `CSLKernelOps.td` | PE-level compute (functions, tasks, variables) |
| Data Movement | `CSLDataMovementOps.td` | DSD creation and bulk moves |
| Runtime | `CSLRuntimeOps.td` | Module system, exports, comptime blocks |

Shared definitions (the dialect itself, base op class, types, enums) live in `CSLBase.td`. All sub-category files include `CSLBase.td` and are aggregated by `CSLOps.td`.

### Directory Structure

```
mlir/include/air/Dialect/CSL/
├── CSLBase.td              # Dialect, types, enums
├── CSLOps.td               # Master include (aggregates all .td files)
├── CSLLayoutOps.td         # csl.layout, csl.set_rectangle
├── CSLPlacementOps.td      # csl.set_tile_code
├── CSLRoutingOps.td        # csl.color, csl.route
├── CSLKernelOps.td         # csl.func, csl.task, csl.return, csl.var
├── CSLDataMovementOps.td   # csl.get_mem_dsd, csl.get_fab_dsd, csl.mov
├── CSLRuntimeOps.td        # csl.import_module, csl.export_name, csl.export_symbol,
│                           # csl.comptime, csl.module, csl.param
├── CSLDialect.h            # C++ type declarations
├── CSLOps.h                # C++ op header
└── CMakeLists.txt          # TableGen generation rules

mlir/lib/Dialect/CSL/IR/
├── CSLDialect.cpp          # Dialect initialization, type parsing/printing
├── CSLOps.cpp              # Custom parsers/printers for csl.func and csl.task
└── CMakeLists.txt          # Library build rules

mlir/test/Dialect/CSL/
├── layout_ops.mlir         # Layout op tests
├── placement_ops.mlir      # Placement op tests
├── routing_ops.mlir        # Routing op tests
├── kernel_ops.mlir         # Kernel op tests
├── datamovement_ops.mlir   # Data movement op tests
├── runtime_ops.mlir        # Runtime op tests
├── complete_program.mlir   # Full program integration test
├── roundtrip.mlir          # Parse-print-parse consistency test
├── invalid.mlir            # Verifier diagnostic tests
└── invalid_parse.mlir      # Parser rejection tests
```

---

## 3. Types

The dialect defines three custom types, all declared in C++ and referenced via `DialectType` predicates in TableGen.

| Type | MLIR Syntax | Description |
|---|---|---|
| `ColorType` | `!csl.color` | A WSE communication color handle. Colors identify logical channels on the fabric. |
| `DsdType` | `!csl.dsd` | A Data Structure Descriptor. DSDs describe patterns of memory or fabric access. |
| `ImportedModuleType` | `!csl.imported_module` | A handle to an imported CSL library module. |

### Parsing

Types are parsed/printed via `CSLDialect::parseType` and `CSLDialect::printType` using keyword matching:

```
!csl.color           → ColorType
!csl.dsd             → DsdType
!csl.imported_module → ImportedModuleType
```

---

## 4. Enumerations

Two `I32EnumAttr` enumerations are defined in `CSLBase.td`:

### `Direction` (routing directions)

```
NORTH = 0
SOUTH = 1
EAST  = 2
WEST  = 3
RAMP  = 4
```

`RAMP` represents the connection to the host/off-chip memory, a unique WSE concept where the "ramp" lanes connect PE tiles to external DRAM.

### `DsdKind` (DSD categories)

```
mem1d  = 0   # 1D contiguous memory DSD
mem2d  = 1   # 2D strided memory DSD (future)
fabin  = 2   # Fabric input DSD (receive from network)
fabout = 3   # Fabric output DSD (send to network)
```

---

## 5. Operations by Category

### 5.1 Layout

#### `csl.layout`

Top-level container for spatial configuration. Corresponds to the `layout { }` block in CSL, which is executed on the host to configure the PE grid before program launch.

**Traits:** `SingleBlock`, `NoTerminator`

```mlir
csl.layout {
  csl.set_rectangle 2, 2
  csl.set_tile_code 0, 0 file("pe_program.csl")
  csl.export_name "data" : f32
}
```

**Rationale:** CSL separates layout (host-side spatial configuration) from PE code. The `csl.layout` region captures this separation at the IR level, constraining spatial-configuration ops to a well-defined scope.

#### `csl.set_rectangle`

Sets the rectangular PE grid dimensions. Corresponds to `@set_rectangle(width, height)` in CSL.

**Arguments:** `I64Attr:$width`, `I64Attr:$height`

```mlir
csl.set_rectangle 4, 8    // 4 columns, 8 rows
csl.set_rectangle 750, 994 // WSE-3 maximum
```

### 5.2 Placement

#### `csl.set_tile_code`

Assigns a CSL source module to a PE at grid position (x, y). Corresponds to `@set_tile_code(x, y, "file.csl", .{ params })` in CSL.

**Arguments:** `I64Attr:$x`, `I64Attr:$y`, `StrAttr:$file`, `OptionalAttr<DictionaryAttr>:$params`

```mlir
// Without params — all tiles run the same code
csl.set_tile_code 0, 0 file("pe_program.csl")

// With compile-time params — per-tile specialization
csl.set_tile_code 1, 1 file("pe_program.csl") params({col = 1 : i32, row = 1 : i32})
```

**Rationale:** The optional `params` dictionary mirrors CSL's compile-time parameter passing, enabling per-tile specialization without duplicating modules. This is a key difference from AIE, where code is cloned per tile.

### 5.3 Routing

#### `csl.color`

Declares a communication color with a given integer ID. Colors are the fundamental routing primitive on the WSE fabric — each color identifies a logical channel through which wavelets (data packets) travel.

**Arguments:** `I32Attr:$id`
**Results:** `!csl.color`

```mlir
%c0 = csl.color 0 : !csl.color
%c1 = csl.color 1 : !csl.color
```

#### `csl.route`

Binds a color to a fabric direction. Determines how wavelets tagged with this color propagate through the on-chip network.

**Arguments:** `CSL_ColorType:$color`, `CSL_DirectionEnum:$direction`

```mlir
%c = csl.color 0 : !csl.color
csl.route %c dir(EAST)    // Wavelets on color 0 travel east
csl.route %c dir(RAMP)    // Color 0 also connects to host DRAM
```

**Rationale:** CSL's routing model is color-centric: you declare colors, then bind them to directions. This is fundamentally different from AIE's circuit-switched flows or TT's NoC multicast. Representing colors as SSA values enables dataflow analysis of the communication graph.

### 5.4 Kernel

#### `csl.func`

Defines a CSL function on a PE. The body contains standard MLIR operations (`arith`, `scf`, `memref`) representing the compute kernel. Corresponds to `fn name() void { ... }` in CSL.

**Traits:** `IsolatedFromAbove`, `Symbol`

```mlir
csl.func @compute() {
  // Standard MLIR compute ops go here
  csl.return
}
```

**Custom assembly format:** The parser and printer are hand-written in `CSLOps.cpp` to produce clean `csl.func @name() { ... }` syntax rather than the generic op format.

#### `csl.task`

Defines an event-driven task on a PE. Tasks are triggered when a wavelet arrives on the bound color. This is the fundamental execution mechanism on the WSE.

**Traits:** `IsolatedFromAbove`, `Symbol`
**Arguments:** `SymbolNameAttr:$sym_name`, `I32Attr:$color_id`

```mlir
csl.task @recv_data() color(3) {
  csl.return
}
```

**Rationale:** Tasks are unique to WSE's event-driven execution model. Unlike AIE (sequential within a core) or TT (kernel-launch-based), WSE tasks are activated by wavelet arrival on a specific color. The `color(N)` syntax makes this binding explicit.

#### `csl.return`

Terminator for `csl.func` and `csl.task` regions.

**Traits:** `Terminator`, `ParentOneOf<["FuncOp", "TaskOp"]>`

```mlir
csl.func @f() {
  csl.return    // required terminator
}
```

#### `csl.var`

Declares a PE-local variable. Corresponds to `var name: [N]T` in CSL.

**Traits:** `Symbol`
**Arguments:** `SymbolNameAttr:$sym_name`, `TypeAttr:$type`

```mlir
csl.var @buf : memref<1024xf32>
csl.var @matrix : memref<32x32xf16>
csl.var @scalar : f32
```

### 5.5 Data Movement

#### `csl.get_mem_dsd`

Creates a memory-backed Data Structure Descriptor referencing a contiguous region of PE-local memory. Corresponds to `@get_dsd(mem1d_dsd, ...)` in CSL.

**Arguments:** `AnyMemRef:$buffer`, `Index:$length`
**Results:** `!csl.dsd`

```mlir
%dsd = csl.get_mem_dsd %buf, %len : memref<1024xf32>, index -> !csl.dsd
```

#### `csl.get_fab_dsd`

Creates a fabric-backed DSD. Used for streaming data to/from neighboring PEs via colors.

**Arguments:** `CSL_DsdKindEnum:$kind`, `CSL_ColorType:$color`, `Index:$length`
**Results:** `!csl.dsd`

```mlir
%c = csl.color 0 : !csl.color
%dsd_in  = csl.get_fab_dsd fabin  %c, %len : !csl.color, index -> !csl.dsd
%dsd_out = csl.get_fab_dsd fabout %c, %len : !csl.color, index -> !csl.dsd
```

**Rationale:** DSDs are WSE's primary mechanism for efficient bulk data access. By representing them as SSA values in MLIR, optimization passes can analyze and transform DSD patterns — for example, converting scalar loops into bulk DSD moves, or merging compatible DSDs.

#### `csl.mov`

Performs a hardware-accelerated bulk data move between two DSDs. Corresponds to `@mov32` or `@mov16` builtins in CSL.

**Arguments:** `CSL_DsdType:$dst`, `CSL_DsdType:$src`

```mlir
csl.mov %dst_dsd, %src_dsd : !csl.dsd, !csl.dsd
```

### 5.6 Runtime

#### `csl.import_module`

Imports a CSL library module by name, optionally passing compile-time parameters. Corresponds to `@import_module("name", params)` in CSL.

**Arguments:** `StrAttr:$module_name`, `OptionalAttr<DictionaryAttr>:$params`
**Results:** `!csl.imported_module`

```mlir
%mod = csl.import_module "<memcpy/memcpy>" : !csl.imported_module
%mod2 = csl.import_module "<memcpy/get_params>" params({width = 2 : i32, height = 2 : i32}) : !csl.imported_module
```

#### `csl.export_name`

Declares a host-visible symbol in the layout. Used inside `csl.layout` blocks to make PE data and entry points accessible via the `SdkRuntime` host API. Corresponds to `@export_name("name", type, is_ptr)` in CSL.

**Arguments:** `StrAttr:$sym_name`, `TypeAttr:$type`

```mlir
csl.export_name "data" : f32
csl.export_name "compute" : () -> ()
csl.export_name "buffer" : memref<1024xf32>
```

#### `csl.export_symbol`

Exports a symbol from a PE module. Used inside `csl.comptime` blocks to make symbols visible to the layout or host. Corresponds to `@export_symbol(sym, "alias")` in CSL.

**Arguments:** `FlatSymbolRefAttr:$sym`, `OptionalAttr<StrAttr>:$alias`

```mlir
csl.export_symbol @compute
csl.export_symbol @buf_ptr alias("data")
```

#### `csl.comptime`

A region evaluated at compile time by the CSL compiler. Used for binding tasks to colors, exporting symbols, and other static configuration. Corresponds to `comptime { ... }` in CSL.

**Traits:** `SingleBlock`, `NoTerminator`

```mlir
csl.comptime {
  csl.export_symbol @compute
  csl.export_symbol @buf alias("buffer")
}
```

#### `csl.module`

Defines a CSL module (PE program) that can be assigned to PE tiles. Contains parameter declarations, variable declarations, functions, tasks, and comptime blocks.

**Traits:** `IsolatedFromAbove`, `Symbol`, `SymbolTable`, `SingleBlock`, `NoTerminator`

```mlir
csl.module @pe_program {
  csl.param @tile_id : i32
  csl.var @buf : memref<1024xf32>
  csl.func @compute() { csl.return }
  csl.comptime { csl.export_symbol @compute }
}
```

**Rationale:** `csl.module` has both `Symbol` (so it can be referenced by name from `csl.set_tile_code`) and `SymbolTable` (so it can contain named symbols like `csl.var`, `csl.func`, `csl.param`). `IsolatedFromAbove` enforces that PE code cannot reference values from the layout scope — matching CSL's strong separation between host-side layout code and PE-side kernel code.

#### `csl.param`

Declares a compile-time module parameter. Parameters are specialized per-tile when assigned via `csl.set_tile_code`. Corresponds to `param name: type;` in CSL.

**Traits:** `Symbol`
**Arguments:** `SymbolNameAttr:$sym_name`, `TypeAttr:$type`

```mlir
csl.param @memcpy_params : i64
csl.param @tile_id : i32
```

---

## 6. Building a Complete Program

A CSL program in MLIR follows a two-part structure that mirrors real CSL code: a **layout** section (host-side spatial configuration) and one or more **module** definitions (PE-side code).

### Step 1: Define the PE grid

The `csl.layout` block sets up the grid and assigns code to tiles:

```mlir
csl.layout {
  csl.set_rectangle 2, 2
  csl.set_tile_code 0, 0 file("pe_program.csl")
  csl.set_tile_code 1, 0 file("pe_program.csl")
  csl.set_tile_code 0, 1 file("pe_program.csl")
  csl.set_tile_code 1, 1 file("pe_program.csl")
```

### Step 2: Export host-visible symbols

Still inside the layout, declare symbols the host can access:

```mlir
  csl.export_name "arg_0" : memref<24xf32>
  csl.export_name "init_and_compute" : () -> ()
}
```

### Step 3: Define the PE module

The `csl.module` contains all PE-level declarations:

```mlir
csl.module @pe_program {
  csl.param @memcpy_params : i64          // compile-time params
  csl.var @arg_0 : memref<24xf32>         // PE-local memory
```

### Step 4: Write compute functions and tasks

```mlir
  csl.func @compute() {
    // Standard MLIR ops (arith, scf, memref) go here
    csl.return
  }
```

### Step 5: Export symbols at compile time

The `csl.comptime` block binds PE-internal symbols to host-visible names:

```mlir
  csl.comptime {
    csl.export_symbol @arg_0 alias("arg_0")
    csl.export_symbol @init_and_compute
  }
}
```

### Complete Example

Putting it all together — a GEMV kernel on a 2x2 grid:

```mlir
csl.layout {
  csl.set_rectangle 2, 2
  csl.set_tile_code 0, 0 file("pe_program.csl")
  csl.set_tile_code 1, 0 file("pe_program.csl")
  csl.set_tile_code 0, 1 file("pe_program.csl")
  csl.set_tile_code 1, 1 file("pe_program.csl")
  csl.export_name "arg_0" : memref<24xf32>
  csl.export_name "arg_1" : memref<6xf32>
  csl.export_name "arg_2" : memref<4xf32>
  csl.export_name "init_and_compute" : () -> ()
}

csl.module @pe_program {
  csl.param @memcpy_params : i64
  csl.var @arg_0 : memref<24xf32>
  csl.var @arg_1 : memref<6xf32>
  csl.var @arg_2 : memref<4xf32>

  csl.func @compute() {
    csl.return
  }

  csl.func @init_and_compute() {
    csl.return
  }

  csl.comptime {
    csl.export_symbol @arg_0 alias("arg_0")
    csl.export_symbol @arg_1 alias("arg_1")
    csl.export_symbol @arg_2 alias("arg_2")
    csl.export_symbol @init_and_compute
  }
}
```

### With Routing and Data Movement

Adding inter-PE communication:

```mlir
// Routing configuration
%send_color = csl.color 0 : !csl.color
%ack_color  = csl.color 1 : !csl.color
csl.route %send_color dir(EAST)
csl.route %ack_color dir(WEST)

// Data movement inside a function
func.func @transfer(%buf : memref<256xf32>) {
  %len = arith.constant 256 : index
  %c = csl.color 0 : !csl.color

  %mem_dsd = csl.get_mem_dsd %buf, %len : memref<256xf32>, index -> !csl.dsd
  %fab_out = csl.get_fab_dsd fabout %c, %len : !csl.color, index -> !csl.dsd

  csl.mov %fab_out, %mem_dsd : !csl.dsd, !csl.dsd
  return
}
```

---

## 7. Testing

Tests live in `mlir/test/Dialect/CSL/` and use the MLIR FileCheck framework. The lit test target `check-airmlir-dialect-csl` runs all CSL dialect tests.

### Running Tests

```bash
# Run all CSL dialect tests via lit (from the build directory)
cd build && lit --config-prefix=lit.site mlir/test/Dialect/CSL/ -v

# Run as part of the full test suite
cd build && lit --config-prefix=lit.site mlir/test/ -v
```

### Test Categories

| File | Category | Description |
|---|---|---|
| `layout_ops.mlir` | Positive | Grid dimensions, single PE, WSE-3 max, composite layout |
| `placement_ops.mlir` | Positive | Tile code assignment, params, different modules per tile |
| `routing_ops.mlir` | Positive | All five directions, multiple colors, high color IDs |
| `kernel_ops.mlir` | Positive | Variables, functions, tasks with various color IDs |
| `datamovement_ops.mlir` | Positive | Memory/fabric DSDs, bulk moves, multi-color pipelines |
| `runtime_ops.mlir` | Positive | Module imports, exports, params, comptime blocks |
| `complete_program.mlir` | Integration | Full GEMV-like program with all six categories |
| `roundtrip.mlir` | Roundtrip | `--verify-roundtrip` parse-print-parse consistency |
| `invalid.mlir` | Negative | Verifier rejects misplaced ops (wrong parent, missing traits) |
| `invalid_parse.mlir` | Negative | Parser rejects malformed syntax (wrong keywords, bad enums) |

---

## 8. Build System Integration

### CMake Targets

- **`CSLDialect`** — Static library containing the dialect implementation.
- **`csl-headers`** — Generated TableGen headers.
- **`check-airmlir-dialect-csl`** — Lit test target for CSL dialect tests.

### Dependencies

The CSL dialect depends on:
- MLIR core (`MLIRIR`, `MLIRSupport`, `MLIRInferTypeOpInterface`)
- MLIR dialect interfaces (`MLIRCallInterfaces`, `MLIRSideEffectInterfaces`)

It is registered in `mlir/lib/InitAll.cpp` alongside the AIR and AIRRt dialects, so `air-opt` automatically recognizes `csl.*` operations.

---

## 9. Design Rationale

### Single Dialect, Multiple TableGen Files

Rather than creating six separate MLIR dialects (`csl_layout`, `csl_routing`, etc.), all operations share one `csl` dialect namespace. This avoids the overhead of six dialect registrations, six C++ namespaces, and cross-dialect type/attribute sharing boilerplate. The six conceptual categories are separated via TableGen files for maintainability.

### Custom Types vs. Attributes

Colors, DSDs, and imported modules are represented as MLIR types (not attributes) because they flow through SSA dataflow — a `csl.color` produces a `!csl.color` value consumed by `csl.route` and `csl.get_fab_dsd`. This enables standard MLIR dataflow analysis.

### `IsolatedFromAbove` on Module/Func/Task

The `IsolatedFromAbove` trait on `csl.module`, `csl.func`, and `csl.task` mirrors CSL's strong scoping rules: PE code cannot reference layout-level values, and function bodies cannot capture variables from their enclosing scope. This enables safe parallel compilation and separate code generation.

### `SymbolTable` on `csl.module`

`csl.module` carries both `Symbol` (to be referenced by name) and `SymbolTable` (to contain named symbols). This is necessary because `csl.var`, `csl.param`, `csl.func`, and `csl.task` all carry the `Symbol` trait, which requires their parent to have `SymbolTable`.

### `SingleBlock` + `NoTerminator` on Region Ops

`csl.layout`, `csl.comptime`, and `csl.module` use `SingleBlock` + `NoTerminator` because they represent declarative regions (configuration blocks) rather than control-flow regions. There is no branching or early return within a layout or comptime block.

### Custom Parsers for `csl.func` and `csl.task`

These ops use hand-written parsers/printers (in `CSLOps.cpp`) instead of declarative `assemblyFormat` because:
1. `csl.func @name() { ... }` syntax is cleaner than what the generic format would produce.
2. `csl.task @name() color(N) { ... }` requires custom parsing of the `color(N)` suffix after the argument list.

All other ops use declarative `assemblyFormat` defined in TableGen, which is preferred for maintainability.

### Relationship to AIR Dialect

The CSL dialect is a **lowering target** for AIR. The `air-opt --air-to-csl` pass (under development) transforms AIR operations into CSL operations:

```
air.launch       → csl.layout + csl.set_rectangle
air.segment      → csl.set_tile_code (for each PE)
air.herd body    → csl.module with csl.func/csl.task
air.channel.*    → csl.color + csl.route + csl.get_fab_dsd + csl.mov
```

The CSL dialect is then consumed by `air-translate --csl-emit`, which walks the `csl.*` ops and emits syntactically valid CSL source files (`layout.csl`, `pe_program.csl`, `run.py`).
