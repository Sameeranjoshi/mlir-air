# MLIR-AIR-GPU Agent Memory

## Key Accessor Patterns (Verified Against Generated Code)

### CSL ExportSymbolOp
- `getSym()` returns `llvm::StringRef` directly (NOT `FlatSymbolRefAttr`)
- Correct: `exportSym.getSym().str()`
- Wrong: `exportSym.getSym().getValue().str()` (compile error: no member `getValue` in `StringRef`)

### CSL ColorOp
- `getId()` returns `std::optional<int32_t>` — dereference with `*id` not `.getInt()`

### CSL DsdKind enum
- Inside `namespace xilinx::csl`, use `DsdKind::fabin` (no `csl::` prefix needed)

### CSL Direction enum
- NORTH=0, SOUTH=1, EAST=2, WEST=3, RAMP=4
- Inside `namespace xilinx::csl`, use `Direction::NORTH` etc.

### CSL RouteOp (optional dirs)
- `getInputDir()` and `getOutputDir()` return `std::optional<Direction>` (OptionalAttr)
- Correct: `if (auto inDir = routeOp.getInputDir()) { ... *inDir ... }`
- Wrong: using in switch directly (compile error)

### CSL CodeRegionOp (optional region_name)
- `getRegionName()` returns `std::optional<StringRef>`
- Use: `if (auto name = regionOp.getRegionName()) pyName = name->str();`

## CSLToTextTranslation.cpp Architecture (single-pass, refactored)

File: `mlir/lib/Targets/CSLToTextTranslation.cpp`

**Key classes:**
1. `CSLWriter` — indented CSL text output for `.csl` PE program files
2. `CSLTextEmitter` — orchestrates all translation
3. `emitLayoutBody()` — shared single-pass helper, called by both `emitLayoutPy` and `emitRunPy`

**Output files:** `name.csl` (module/kernel), `layout.py` (layout only), `run.py` (layout + runtime)

**emitLayoutBody dispatch (source order, single pass):**
- `csl.color` → `color_N = layout.alloc_color()`
- `csl.sym_color "name"` → `name = Color('name')`
- `csl.scoped_color %r "name"` → `rPyName_name = r.color('name')`
- `csl.route in(X) out(Y)` → `rp_N = RoutingPosition(); rp_N.set_input([Route.X]); rp_N.set_output([Route.Y])`
- `csl.code_region "pyname" ...` → `pyname = layout.create_code_region('./file.csl', 'pyname', w, h)` + inline paint ops
- `csl.place` → `regionPyName.place(x, y)`
- `csl.set_param_all` → `regionPyName.set_param_all('name', value)`
- `csl.set_param at(x,y)` → `regionPyName.set_param(IntVector(x,y), 'name', value)`
- `csl.set_param_color` → `regionPyName.set_param_all(colorPyName)`
- `csl.input_port` / `csl.output_port` → `portN = regionPyName.create_{input,output}_port(...)`
- `csl.dataflow %s -> %d` → `layout.connect(s, d)`
- `csl.input_stream` / `csl.output_stream` → `in_streamN` / `out_streamN = layout.create_{input,output}_stream(...)`
- `csl.export_name` → `layout.export_name(...)`

**Python naming:**
- `csl.code_region "name"` attr → use `name` as Python var; no attr → `codeN`
- `csl.sym_color "rx1"` → Python var IS `rx1`
- `csl.scoped_color %sender1 "tx"` → Python var is `sender1_tx`
- routes: `rp_0`, `rp_1`, ...; ports: `port0`, `port1`, ...; streams: `in_streamN`, `out_streamN`

**run.py imports (new format):**
```python
from cerebras.geometry.geometry import IntVector
from cerebras.sdk.runtime.sdkruntimepybind import (
    Color, Edge, Route, RoutingPosition,
    SdkLayout, SdkTarget, SdkRuntime, SimfabConfig, get_platform,
)
layout = SdkLayout(platform)   # run.py uses platform
layout = SdkLayout()            # layout.py (no platform)
```

## New CSL Ops (tutorials 01-05)

`CSLLayoutOps.td` active ops:
- `csl.sym_color "name" : !csl.color`
- `csl.scoped_color %region "name" : !csl.color`
- `csl.set_param %region at(x,y) param("name") value(V : T)`
- `csl.set_param_color %region color(%c)`
- `csl.input_port %region color(%c) edge(E) routes(%r1,...) size(N) : !csl.port`  (variadic routes)
- `csl.output_port %region color(%c) edge(E) routes(%r1,...) size(N) : !csl.port` (variadic routes)
- `csl.input_stream %port : !csl.stream`
- `csl.output_stream %port : !csl.stream`
- `csl.routing_position in("DIR") out("DIR,DIR") : i32`  (multi-dir, CSV strings)
- `csl.get_edge_routing edge(E) routes(%r1,...) : i32`
- `csl.paint_all %r color(%c) routes(%rp,...) edge_routes(%er,...)`  (AttrSizedOperandSegments)
- `csl.paint_range %r rect(x1,y1,x2,y2) color(%c) routes(%rp,...)`
- `csl.set_param_range_color %r rect(x1,y1,x2,y2) color(%c)`
- `csl.set_param_range_named %r rect(x1,y1,x2,y2) param("name") color(%c)`

NOTE: `csl.input_port`/`csl.output_port` use `routes` (plural) not `route` — update older tests.

`CSLRoutingOps.td`:
- `csl.route` in/out dirs are now `OptionalAttr` — `in(D)` and/or `out(D)` are optional
- `csl.route in(RAMP) : i32` (input only), `csl.route out(RAMP) : i32` (output only)

`csl.code_region` gets optional `region_name` string attr as first positional token.

`csl.param @c : !csl.color` in kernel body → emits `param c: color;` in .csl

**Python imports (updated for Tutorial 05):**
```python
from cerebras.geometry.geometry import IntVector, IntRectangle
from cerebras.sdk.runtime.sdkruntimepybind import (
    Color, Edge, Route, RoutingPosition, get_edge_routing,
    SdkLayout, SdkTarget, SdkRuntime, SimfabConfig, get_platform,
)
```

## Pre-existing Test Failures
- `mlir/test/Dialect/CSL/layout_stress.mlir` — fails with `invalid kind of attribute specified`
  on `csl.kernel` custom op parsing (unrelated to emitter work).

## CSL Data Tasks (trigger_kind = "data_task")

- `csl.task @name attributes {trigger_kind = "data_task", color = @stream_or_color} { ^bb0(%val: f32): ... }`
- `color` attr may reference the stream name (`@ch01`) OR the color name (`@ch01_color`)
- `updateDataTaskColors()` in `CSLLowerDataflowData.cpp` resolves stream→color before erasure
- `csl.dataflow.send_wavelet @stream value(%v) : f32` — 1-element fabout DSD + sync @fmovs
- Emitter declares `<color>_in_q: input_queue = @get_input_queue(N)` for each data task
- Emitter emits: `const sym_id: data_task_id = @get_data_task_id(<color>_in_q)` + `@bind_data_task`
- Task body args use names `a0, a1, ...`; must be populated in nameMap BEFORE emitFuncBody

## Build Pattern
After source changes to `mlir/lib/Targets/` or `mlir/include/air/Dialect/CSL/`:
```bash
cd build && ninja AIRTargets   # fast build check
cd build && ninja install      # full install
```

If ninja thinks CSLTransforms is up-to-date despite source changes:
```bash
touch mlir/lib/Dialect/CSL/Transforms/TheFile.cpp && cd build && ninja install
```

## Test Pattern (lit has config issue — use FileCheck directly)
```bash
air-translate --emit-csl --csl-output-dir=/tmp/out input.mlir
FileCheck input.mlir --input-file=/tmp/out/file.csl --check-prefix=PREFIX
FileCheck input.mlir --input-file=/tmp/out/run.py   --check-prefix=RUN_PY
```

## New Kernel Ops: csl.get_x_coord / csl.get_y_coord + arith.select

Added in commit `6fcae473`.

- `csl.get_x_coord : i16` / `csl.get_y_coord : i16` in CSLOps.td, carry `[Pure]` trait
- Emitter emits `const tN: u16 = layout_mod.get_x_coord()` — note **u16** not i16!
  (SDK's layout_mod.get_x_coord() returns u16; i16 causes type mismatch on WSE-3)
- To use in arithmetic: cast via `arith.index_cast %x_i16 : i16 to index` then use
  `arith.remui %x, %two : index` (index maps to u16 in CSL)
- `arith.select` now supported in emitFuncBody: emits `var tN: T = if (cond) t else f;`
- Adding `[Pure]` trait to CSL ops requires `mlir/Interfaces/SideEffectInterfaces.h`
  in `CSLOps.h` — without it, `mlir::ConditionallySpeculatable` is undefined at compile time
