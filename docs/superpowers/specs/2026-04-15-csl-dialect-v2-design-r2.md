# CSL Dialect Family — v2 Design (Revision 2)

**Date:** 2026-04-15 (r2 — updated same day after additional brainstorming)
**Status:** Design approved. Implementation underway (V1 core complete; passes + emitters in progress).
**Supersedes:** [`2026-04-15-csl-dialect-v2-design.md`](2026-04-15-csl-dialect-v2-design.md) — same design,
extended with (a) direction-annotation clarification, (b) unified subgrid model for data and placement,
(c) implementation status table, (d) exact op signatures for all V1 ops.
**Prior art:** SPADA (arXiv:2511.09447), MLIR Stencils-CSL (arXiv:2601.17754).
**Scope:** CSL dialect op surface, type system, lowering pipeline, emit targets.

---

## 0. TL;DR

| Decision | Choice | Rationale |
|---|---|---|
| Programming model | **Actor model** — `csl.program` is a template placed onto PEs | CSL itself is template-first; lowest impedance to emit |
| Top container | **`csl.wafer` inside standard `module {}`** | Like `aie.device` — op is the program container |
| Region structure | **Three siblings**: `csl.program`, `csl.layout`, `csl.host` | Mirror CSL's 3-file structure; each emitter reads one dialect |
| Comptime params | **Block args with `!csl.comptime<T>` type** | SSA-idiomatic; maps to CSL `param` |
| Export direction | **Auto-derived** by `-csl-derive-exports` from host transfer ops | User never writes `direction =`; it is computed and set as IR attr by pass |
| `csl_layout.export` | **Explicit in V1** — redundant but intentional; could be auto-generated later | Keeps emitters fully decoupled for now |
| Layout emit | **sdkLayout Python API** → `csl_layout.py` | sdkLayout auto-assigns color IDs; no `-csl-allocate-colors` pass needed |
| `csl.import` | **Implicit** — injected by emitter, not in IR | Always needed, always emitted; no user boilerplate |
| SPMD placement | **`csl_layout.place_grid`** with subgrid range | Single template → N placements via `-csl-elaborate-grid` |
| Data sharding | **Unified subgrid model**: `csl.data` carries same subgrid range as `place_grid` | SPADA's `place` block applies range to BOTH placement AND data; we follow this |
| Inter-PE comms | **Deferred** — reserve op-space for `csl.stream` | SPADA's `relative_stream(dx,dy)` is the right model |
| V1 pass | **One `-air-to-csl` pass** | Simpler than 3 passes; separate translate targets for emit |

---

## 1. Design Philosophy

### Actor Model vs. SPADA/Stencil Source

| Source-level concept | SPADA / Stencils-CSL (arXiv) | Our CSL dialect (IR level) |
|---|---|---|
| `place kernel in [0:W, 0:H] { data }` | SPADA `place` block — placement + data context | `csl_layout.place_grid @prog from (0,0) to (W-1,H-1)` + `csl.data @A shard [0:W,0:H]` |
| `dataflow in [0:W, 0:H] { streams }` | SPADA `dataflow` block | Future: `csl.stream @s { offset = (dx,dy) }` |
| `compute in [0:W, 0:H] { fns }` | SPADA `compute` block | `csl.func @f()` inside `csl.program` |
| Stencil global array `A[I][J]` | Stencils-CSL: global decl, compiler shards | Future: `csl.data @A : memref<1024x1024xf32> shard [0:W, 0:H]` |
| Halo exchange | Stencils-CSL: halo regions + async communication | Future: `csl.stream` with `halo` attribute |
| Color allocation | SPADA Section VI-B: checkerboard decomposition | sdkLayout auto-assigns; `csl_layout.color @sym` for explicit names |
| `@set_tile_code(x,y,f,params)` | CSL runtime binding | `csl_layout.place @prog at (x,y) {params}` → emitted as sdkLayout `create_code_region` |
| `@export_symbol(sym)` | CSL PE-side export | `csl.export @sym` (direction added by pass, emits `@export_symbol` in CSL) |

**Why Actor Model at this level:** CSL's hardware model has up to 850K PEs. SPADA and Stencils-CSL are
source-level abstractions that LOWER TO the actor model at the CSL IR level. Our dialect sits below
those frontends and above CSL text. Actor model (write once, place N times) is the correct IR form.

### The Subgrid Principle (new in r2)

SPADA's key insight: the subgrid range `[0:W, 0:H]` is **both** a placement declaration AND a data
partition context. Our dialect adopts this:

- `csl_layout.place_grid @pe from (0,0) to (W-1,H-1)` → placement subgrid
- `csl.data @A shard [0:W, 0:H]` → data sharding subgrid (same dimensions, different meaning)

The `-csl-shard-data` pass reads both. For V1 (AIR input), data is already PE-local, so `csl.data`
is not needed yet. But the design is unified: subgrid expressions appear in both layers.

### Direction Annotation (clarified in r2)

```mlir
// User writes (direction-free):
csl.export @a { alias = "a" }

// After -csl-derive-exports pass (IR attribute added, not user-written):
csl.export @a { alias = "a", direction = "in" }
```

**Why materialize direction as an IR attribute (not derive at emit time):**
- The program emitter (`--emit-csl-program`) must know whether to emit `@export_symbol` for each export.
  If direction were derived on-the-fly, the program emitter would need to walk `csl.host` ops —
  breaking the three-emitter independence guarantee.
- Materializing in IR makes the pipeline observable: you can dump IR after `-csl-derive-exports`
  and see all directions at a glance, which aids debugging.
- The direction attribute is the *communication channel* between the pass and the emitter. This is
  standard MLIR practice (e.g., `affine.parallel` carries reduction kinds set by analysis passes).

**Can `csl_layout.export` be removed?** In principle yes — it is derivable from `csl.export direction=in/out`.
For V1 we keep both explicitly: `csl.export` drives the PE-side `@export_symbol`, `csl_layout.export`
drives `layout.add_field`. Future: a `-csl-hoist-layout-exports` pass could generate layout exports
automatically from program exports, eliminating the redundancy.

---

## 2. Top-Level Structure: `csl.wafer`

```mlir
module {
  csl.wafer @wseprog { arch = "wse3" } {

    // (1) PE kernel template — models pe_program.csl
    csl.program @vecadd_pe {
      %a = csl.var @a : memref<256xf32>
      %b = csl.var @b : memref<256xf32>
      %c = csl.var @c : memref<256xf32>
      csl.func @compute { csl.return }
      csl.export @a { alias = "a" }
      csl.export @b { alias = "b" }
      csl.export @c { alias = "c" }
      csl.export @compute { kind = "func" }
    }

    // (2) Layout — placement + routing (models csl_layout.py via sdkLayout)
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
      csl_layout.place @vecadd_pe at (0, 0)
      csl_layout.export "a" from @vecadd_pe::@a
      csl_layout.export "b" from @vecadd_pe::@b
      csl_layout.export "c" from @vecadd_pe::@c
      csl_layout.export "compute" from @vecadd_pe::@compute {kind = "func"}
    }

    // (3) Host — runtime orchestration (models run.py via SdkRuntime)
    csl.host @main(%a_in: memref<256xf32>, %b_in: memref<256xf32>,
                   %c_out: memref<256xf32>) {layout = @main_layout} {
      csl_host.memcpy_h2d %a_in to @main_layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.memcpy_h2d %b_in to @main_layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.launch @main_layout::@compute
      csl_host.memcpy_d2h @main_layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
    }
  }
}
```

**`csl.wafer` properties:**
- `arch` attribute: `"wse2"` or `"wse3"` — hardware constraints for passes
- Single symbol table inside `csl.wafer`; all three siblings cross-reference by symbol
- Multiple wafers per module are allowed (for testing)

---

## 3. `csl.program` — PE Kernel Template

### 3.1 Complete Op Reference (V1)

| Op | MLIR signature | Emits to CSL | Notes |
|---|---|---|---|
| `csl.program` | `csl.program @sym (%p: !csl.comptime<T>, ...) { ... }` | `pe_program.csl` file | SymbolTable region; `HasParent<WaferOp>` |
| `csl.var` | `%v = csl.var @sym : memref<NxT>` | `var @sym: [N]T;` | Mutable PE-local buffer |
| `csl.const` | `csl.const @sym : memref<NxT> = dense<...>` | `const @sym: [N]T = ...;` | Read-only PE-local |
| `csl.func` | `csl.func @sym { ... csl.return }` | `fn @sym() void { ... }` | Body: plain MLIR ops |
| `csl.export` | `csl.export @sym { alias = "str" }` | `comptime { @export_symbol(@sym, "str") }` | direction added by pass |
| `csl.export` (func) | `csl.export @sym { kind = "func" }` | `comptime { @export_symbol(@sym, "sym") }` | No alias needed for funcs |
| `csl.return` | `csl.return` | (implicit in CSL) | Terminates `csl.func` |

**NOT in IR (injected by emitter):**
- `@import_module("<memcpy/memcpy>", ...)` — always emitted when any export exists
- `comptime { ... }` block wrapper — emitted from `csl.export` ops

### 3.2 Comptime Parameters

Block arguments with `!csl.comptime<T>` type. Map to CSL `param` declarations.

```mlir
csl.program @gemv_pe(
    %M: !csl.comptime<i16>,   // → param M: i16;
    %N: !csl.comptime<i16>,   // → param N: i16;
    %col: !csl.comptime<i16>  // → param col: i16;
) {
  // %M, %N usable in memref shapes (constant-folded at emit)
  %buf = csl.var @y : memref<?xf32>   // ? resolved from %N at emit
}
```

At placement, concrete values bind to these params as attributes on the `csl_layout.place` op.
Emitter passes them to `region.set_param(IntVector(col, row), "col", col)`.

### 3.3 Direction Annotation (auto-derived)

Direction is set by `-csl-derive-exports`. Values: `"in"` / `"out"` / `"internal"`.

Emitter behavior:
- `direction = "in"` or `"out"`: emit `var @sym: [N]T;` + `@export_symbol(@sym, "alias")`
- `direction = "internal"`: emit `var @sym: [N]T;` only (no `@export_symbol`)
- `kind = "func"`: emit `@export_symbol(@fn, "fn")` regardless of direction

### 3.4 Data Declarations: V1 and Future

**V1 (PE-local, AIR already partitions):**
```mlir
csl.var @a : memref<256xf32>      // mutable PE-local buffer
csl.const @lut : memref<16xi32>   // read-only PE-local constant
```

**Future V2 (global data with subgrid sharding):**
```mlir
// User writes global logical array; compiler shards it using the placement subgrid
csl.data @A : memref<1024xf32> {
  // Subgrid expression: shard dim 0 over the W-PE layout dimension
  // Matches csl_layout.place_grid from (0,0) to (W-1, 0)
  shard_dim = 0 : i64,   // shard along first memref dimension
  halo = 0 : i64          // no halo for vecadd; use 1+ for stencils
}
// After -csl-shard-data pass (reads layout W from csl_layout.place_grid):
// → csl.var @A : memref<(1024/W)xf32>
```

The subgrid connection: `csl.data` sharding uses the SAME `(W, H)` dimensions as
`csl_layout.place_grid from (0,0) to (W-1, H-1)`. The `-csl-shard-data` pass:
1. Reads the layout's `place_grid` range → gets grid dimensions (W, H)
2. Reads `csl.data shard_dim, halo` → computes per-PE buffer size
3. Replaces `csl.data` with `csl.var` of sliced type
4. Inserts `csl.stream` declarations for halo exchange

This mirrors SPADA's `place ... in [0:W, 0:H] { data decls }` where the range
applies to both placement and data. The Stencils-CSL paper (arXiv:2601.17754)
provides prior art for the halo exchange instantiation.

### 3.5 Reserved Op-Space (V1 stubs)

```tablegen
// Defined in TableGen, no lowering until needed:
def CSL_TaskOp   // csl.task @sym { color_sym = @c }
def CSL_StreamOp // csl.stream @sym { offset = (dx, dy) }
def CSL_DataOp   // csl.data @sym : memref<...> { shard_dim, halo }
```

---

## 4. `csl.layout` — Placement + Routing

### 4.1 Complete Op Reference (V1)

| Op | MLIR signature | Emits to | Notes |
|---|---|---|---|
| `csl.layout` | `csl.layout {width = W, height = H} @sym { ... }` | `def get_layout(target):` | `HasParent<WaferOp>` |
| `csl_layout.place` | `csl_layout.place @prog at (x, y)` | `region.set_param(...)` + `region.place(x,y)` | V1: 1×1 only |
| `csl_layout.place_grid` | `csl_layout.place_grid @prog from (x0,y0) to (x1,y1) { params }` | Loop of `place` calls | Expanded by `-csl-elaborate-grid` |
| `csl_layout.export` | `csl_layout.export "sym" from @prog::@exp` | `layout.add_field("sym", ...)` | V1: required per export |
| `csl_layout.color` | `csl_layout.color @sym` | `region.color("sym")` | Deferred: V1.1 |
| `csl_layout.route` | `csl_layout.route @sym { dir = EAST }` | `region.paint_all(color, routes)` | Deferred: V1.1 |

**Emitted `csl_layout.py` structure:**
```python
from cerebras.sdk.runtime.sdkruntimepybind import SdkLayout, SdkTarget

def get_layout(target: SdkTarget) -> SdkLayout:
    layout = SdkLayout(target)
    region = layout.create_code_region("vecadd_pe.csl", "vecadd", 1, 1)
    region.place(0, 0)
    return layout
```

**Key SdkLayout API facts:**
- `SdkLayout(target)` — takes `SdkTarget`, NOT `(width, height)`
- `layout.create_code_region(source, name, w, h)` → `CodeRegion`
- `region.set_param_all(name, val)` — same value for all PEs
- `region.set_param(IntVector(col, row), name, val)` — per-PE specialization
- `region.color(name)` → `Color` — auto-allocated physical ID
- `region.place(x, y)` — position on wafer

### 4.2 SPMD via `place_grid`

```mlir
csl.layout {width = 4 : i64, height = 1 : i64} @gemv_layout {
  csl_layout.place_grid @gemv_pe
    from (0, 0) to (3, 0)
    {M = 4 : i16, N = 6 : i16}
  // $col placeholder: expanded by -csl-elaborate-grid to 4 place ops
  // with col = 0, 1, 2, 3
  csl_layout.export "y" from @gemv_pe::@y
  csl_layout.export "compute" from @gemv_pe::@compute {kind = "func"}
}
```

`-csl-elaborate-grid` turns `place_grid from (0,0) to (3,0)` into:
```mlir
csl_layout.place @gemv_pe at (0, 0) {col = 0 : i16, M = 4 : i16, N = 6 : i16}
csl_layout.place @gemv_pe at (1, 0) {col = 1 : i16, M = 4 : i16, N = 6 : i16}
csl_layout.place @gemv_pe at (2, 0) {col = 2 : i16, M = 4 : i16, N = 6 : i16}
csl_layout.place @gemv_pe at (3, 0) {col = 3 : i16, M = 4 : i16, N = 6 : i16}
```

---

## 5. `csl.host` — Runtime Orchestration

### 5.1 Complete Op Reference (V1)

| Op | MLIR signature | Emits to Python | Notes |
|---|---|---|---|
| `csl.host` | `csl.host @sym(%arg: T, ...) {layout = @L} { ... }` | `def main(...): ... with SdkRuntime(arts) as r:` | `HasParent<WaferOp>` |
| `csl_host.memcpy_h2d` | `csl_host.memcpy_h2d %src to @layout::@sym {px,py,w,h} : memref<NxT>` | `runner.memcpy_h2d(id, src, px, py, w, h, N)` | direction source for derive-exports |
| `csl_host.memcpy_d2h` | `csl_host.memcpy_d2h @layout::@sym to %dst {px,py,w,h} : memref<NxT>` | `runner.memcpy_d2h(dst, id, px, py, w, h, N)` | direction source for derive-exports |
| `csl_host.launch` | `csl_host.launch @layout::@sym` | `runner.launch("sym", nonblock=False)` | Synchronous RPC |

`N` (elem_per_pe) is inferred at emit time from the memref element count ÷ (`width * height`).

**Emitted `run.py` structure:**
```python
from csl_layout import get_layout
from cerebras.sdk.client import SdkRuntime
import numpy as np

def main(target, a_in: np.ndarray, b_in: np.ndarray, c_out: np.ndarray):
    artifacts = get_layout(target).compile("out/")
    N = a_in.size
    with SdkRuntime(artifacts) as runner:
        a_id = runner.get_id("a")
        b_id = runner.get_id("b")
        c_id = runner.get_id("c")
        runner.memcpy_h2d(a_id, a_in, 0, 0, 1, 1, N)
        runner.memcpy_h2d(b_id, b_in, 0, 0, 1, 1, N)
        runner.launch("compute", nonblock=False)
        runner.memcpy_d2h(c_out, c_id, 0, 0, 1, 1, N)
```

### 5.2 Symbol Reference Verification

`@main_layout::@a` is a nested `SymbolRefAttr`. The verifier checks:
1. `@main_layout` resolves to a `csl.layout` inside the same `csl.wafer`
2. `@a` resolves to a `csl_layout.export` inside that layout

Compile-time catch for: wrong export name, wrong type, missing export.

---

## 6. Type System

| Type | TableGen definition | Purpose | Status |
|---|---|---|---|
| `!csl.comptime<T>` | `CSL_ComptimeType` with `TypeParameter<"mlir::Type">` | Compile-time-only value (CSL `param`) | **Implemented** |
| `!csl.color` | `CSL_ColorType` (registered) | Color resource handle | Reserved, no lowering |
| `!csl.stream<T>` | `CSL_StreamType` (registered) | Inter-PE stream handle | Reserved, no lowering |
| `!csl.dsd` | `CSL_DSDType` (registered) | Data structure descriptor | Reserved, no lowering |

**`!csl.comptime<T>` semantics:**
- Only appears as block argument type on `csl.program`
- Cannot be stored to memory (verifier error)
- Can be used in `affine_map` and as `memref` dimension sizes (constant-folded at emit)
- Verifier: inner type must be integer or float (no nested comptime)

---

## 7. Lowering Pipeline

```
vecadd.mlir (AIR dialect)
  │
  │  -air-to-csl
  │    ├─ air.herd body → csl.program (one per unique herd)
  │    ├─ air.herd shape + x_loc/y_loc → csl_layout.place_grid
  │    ├─ air.dma_memcpy_nd (L3↔L2) → csl_host.memcpy_h2d/d2h
  │    └─ air.launch → csl_host.launch
  ▼
csl.wafer { program + layout(place_grid) + host(no directions yet) }
  │
  │  -csl-elaborate-grid     [V1.1 — needed for NxN grids]
  │    └─ place_grid → N individual place ops
  ▼
csl.wafer { individual place ops per PE }
  │
  │  -csl-derive-exports     [DONE]
  │    └─ scans memcpy_h2d/d2h → sets direction on csl.export ops
  ▼
csl.wafer { fully annotated, ready for emit }
  │
  ├─ air-translate --emit-csl-program  →  vecadd_pe.csl    (CSL text)
  ├─ air-translate --emit-csl-layout   →  csl_layout.py    (sdkLayout Python)
  └─ air-translate --emit-csl-host     →  run.py           (SdkRuntime Python)
```

### 7.1 `-air-to-csl` Pass

**Pre-conditions:**
- `air.herd` has `x_loc`/`y_loc` attrs (set by placement pass, or default 0,0)
- `memref.alloc` inside herd has memory-space 2 (L1/PE-local)

**Transforms:**
- herd shape → `csl_layout.place_grid` with grid range `(0,0) to (cols-1, rows-1)`
- herd body → `csl.program` body (clone ops, replace `%tx/%ty` with `!csl.comptime<i16>` args)
- `memref.alloc` (space 2) → `csl.var`
- `air.dma_memcpy_nd` → `csl_host.memcpy_h2d` or `csl_host.memcpy_d2h`
- `air.launch` args → `csl.host` args
- Emit `csl.export` for each `csl.var` accessible from host
- Emit `csl_layout.export` mirroring `csl.export`

### 7.2 `-csl-derive-exports` Pass

**Algorithm:**
1. Collect `csl.export` ops by alias → `DenseMap<StringAttr, ExportOp>`
2. Walk `csl.host` → find `csl_host.memcpy_h2d @layout::@alias` → set `direction = "in"`
3. Walk `csl.host` → find `csl_host.memcpy_d2h @layout::@alias to` → set `direction = "out"`
4. All remaining `csl.export` without direction → set `direction = "internal"`

Key implementation note: `ExportOp::getSym()` returns `StringRef` (MLIR 22 unwraps
`FlatSymbolRefAttr`); `MemcpyH2DOp::getSym()` returns `SymbolRefAttr` (nested ref).
Use `leafRef()` helper to extract the alias name from the nested sym ref.

### 7.3 Emit Targets

| Flag | Output | Reads | Key emission rules |
|---|---|---|---|
| `--emit-csl-program` | `<prog>.csl` per `csl.program` | `csl.program` + direction attrs | `direction=in/out` → `@export_symbol`; `direction=internal` → no export |
| `--emit-csl-layout` | `csl_layout.py` | `csl.layout` | `create_code_region`, `set_param`, `place`, `add_field` |
| `--emit-csl-host` | `run.py` | `csl.host` | `get_id`, `memcpy_h2d`, `memcpy_d2h`, `launch` inside `with SdkRuntime` |

Each emitter is an independent walker. `--emit-csl-program` never reads `csl.host`; it only needs
the direction attr that `-csl-derive-exports` already set on `csl.export`.

---

## 8. V1 Scope and Implementation Status

### 8.1 What V1 Requires

| Component | Status | Notes |
|---|---|---|
| `!csl.comptime<T>` type | **Done** | TypeStorage + hash_value, roundtrip tests pass |
| `csl.wafer` op | **Done** | Custom parse/print (attr-dict before region needs custom format) |
| `csl.program` op | **Done** | Custom parse/print; block args with `!csl.comptime<T>` |
| `csl.layout` op | **Done** | Custom parse/print; `{width,height}` attrs |
| `csl.host` op | **Done** | Custom parse/print; `layout =` dict attr |
| `csl.var`, `csl.func`, `csl.export`, `csl.return` | **Done** | Declarative formats |
| `csl_layout` sub-dialect | **Done** | `place`, `export` ops; roundtrip tests pass |
| `csl_host` sub-dialect | **Done** | `memcpy_h2d`, `memcpy_d2h`, `launch` ops; roundtrip tests pass |
| Dialect registration in `InitAll` | **Done** | `CSLLayoutDialect`, `CSLHostDialect` registered |
| `-csl-derive-exports` pass | **Done** | Annotates all 3 directions; 3 FileCheck tests pass |
| `--emit-csl-program` emitter | **Implemented, needs tests** | In `CSLV2ToPy.cpp` |
| `--emit-csl-layout` emitter | **Implemented, needs tests** | In `CSLV2ToPy.cpp` |
| `--emit-csl-host` emitter | **Implemented, needs tests** | In `CSLV2ToPy.cpp` |
| `-air-to-csl` conversion pass | **Pending** | Task 7 in plan |
| End-to-end `vecadd` test | **Pending** | Depends on `-air-to-csl` |

**Total tests:** 234 passing (as of implementation session).

### 8.2 V1 Explicitly Out of Scope

- `csl.task`, `csl.stream`, `csl.phase` (inter-PE async)
- `!csl.color`, `!csl.dsd`, `!csl.stream<T>` type lowering
- `csl_layout.color`, `csl_layout.route` (routing)
- `csl_layout.place_grid` + `-csl-elaborate-grid` (NxN grids — V1.1)
- `csl.data` sharding (stencil frontends — V2)
- `-csl-shard-data` pass
- `-csl-hoist-layout-exports` (auto-generate layout exports from program exports)

### 8.3 V1.1 (second milestone — multi-PE)

- `csl_layout.place_grid` op (already in sub-dialect)
- `-csl-elaborate-grid` pass
- `!csl.comptime<T>` wired end-to-end with per-PE param binding
- `csl_layout.color` + sdkLayout color assignment
- Basic inter-PE: `csl.stream` + `csl.task`
- Checkerboard decomposition (SPADA Section VI-B)

---

## 9. Data Sharding Design (Future V2 — Stencil Frontends)

### 9.1 Unified Subgrid Model

The key design principle: **subgrid expressions appear in both placement and data layers,
and they refer to the same grid dimensions.**

```
Placement layer (csl.layout):
  csl_layout.place_grid @laplacian_pe from (0,0) to (W-1,H-1)
                                               ↑
                                         defines grid (W×H)

Data layer (csl.program):
  csl.data @in_field : memref<1024x1024xf32> {
    shard_dim = 0, halo = 1
  }
  // -csl-shard-data pass reads W from layout → PE gets memref<(1024/W + 2) x 1024 x f32>
```

The connection: `-csl-shard-data` reads the `place_grid` range from `csl.layout` (within the same
`csl.wafer`) to determine (W, H). `csl.data` provides the per-dimension sharding spec (which dim
to shard, how much halo). This mirrors SPADA's `place kernel in [0:W, 0:H] { data decls }` where
the range applies to both placement and data declarations.

### 9.2 `csl.data` Op Definition (Future)

```mlir
// User writes global logical array; compiler shards it
csl.program @laplacian_pe(%W: !csl.comptime<i16>, %H: !csl.comptime<i16>) {

  csl.data @in_field : memref<1024x1024xf32> {
    shard_dim = 0 : i64,   // shard rows across PEs (H dimension)
    halo = 1 : i64         // 1-row halo for stencil
  }
  // → After -csl-shard-data: csl.var @in_field : memref<(1024/H + 2) x 1024 x f32>

  csl.data @result : memref<1024x1024xf32> {
    shard_dim = 0 : i64,
    halo = 0 : i64
  }
  // → After -csl-shard-data: csl.var @result : memref<(1024/H) x 1024 x f32>

  csl.func @laplacian() { ... }
}
```

### 9.3 `-csl-shard-data` Pass (Future)

Algorithm:
1. For each `csl.wafer`: find `csl.layout` and extract `place_grid` dimensions (W, H)
2. For each `csl.data` in `csl.program`:
   a. Determine per-PE size: `global_size[shard_dim] / W_or_H + 2*halo`
   b. Replace `csl.data` with `csl.var` of sliced memref type
   c. If `halo > 0`: insert `csl.stream` declarations for neighbor communication
3. Add `%W` and `%H` as `!csl.comptime<i16>` block args to `csl.program` (if not present)

---

## 10. Inter-PE Communication (Future V1.1+)

SPADA model: streams are declared with *relative offsets*, no absolute PE coordinates:

```mlir
csl.program @stencil_pe(%col: !csl.comptime<i16>) {
  csl.stream @east { offset = (1, 0) }   // send to PE at (col+1, row)
  csl.stream @west { offset = (-1, 0) }  // receive from PE at (col-1, row)

  csl.task @recv_west { color_sym = @west_color } {
    // triggered when data arrives from west neighbor
  }
  csl.func @send_east() { /* send on @east */ }
}
```

Layout layer binds stream symbols to colors:
```mlir
csl.layout {width = 4 : i64, height = 1 : i64} @stencil_layout {
  csl_layout.color @east_color    // sdkLayout auto-assigns physical ID
  csl_layout.color @west_color
  csl_layout.route @east_color { direction = "EAST" }
  csl_layout.route @west_color { direction = "WEST" }
}
```

Color checkerboard (SPADA Section VI-B): even/odd PE parity → distinct color IDs →
conflict-free routing. sdkLayout handles this automatically when using `region.color()`.

---

## 11. Open Questions Resolved

| # | Question | Decision |
|---|---|---|
| 1 | Should direction be user-written or auto-derived? | **Auto-derived** by `-csl-derive-exports`; never user-written |
| 2 | Can direction annotation be skipped (derive at emit time)? | **No** — emitters must be independent; direction attr is the pass-to-emitter channel |
| 3 | Is `csl_layout.export` redundant with `csl.export direction=in/out`? | **Redundant in principle; kept explicit in V1** for emitter decoupling; future: auto-hoist pass |
| 4 | Subgrid expressions: layout only or data too? | **Both** — `place_grid` range and `csl.data shard_dim` use same (W,H) from layout; pass connects them |
| 5 | Sketch A (tile-centric) vs. Sketch C (actor model)? | **Sketch C** — Actor model; SPADA/Stencils-CSL lower to actor model at IR level |
| 6 | `csl.task` from day one? | **Reserved stub in TableGen, no lowering in V1** |
| 7 | DSDs as types or ops? | **Neither in V1** — `csl-elaborate-dsds` pass future work |
| 8 | Comptime: dict vs block-args? | **Block args with `!csl.comptime<T>`** — SSA-idiomatic |
| 9 | Color IDs: explicit or pass-assigned? | **sdkLayout-assigned** — `region.color("name")` auto-allocates |
| 10 | AIRToCSL: 3 passes or 1? | **One `-air-to-csl` pass** + three independent translate targets |
| 11 | `--emit-csl-rt` backward compat? | **Removed** — three individual targets only |
| 12 | Migration strategy? | **Parallel for two milestones** (feature flag), then delete v1 |

---

## 12. Migration from v1 Dialect

1. **PR 1:** Add new dialect family alongside old. Existing tests unchanged. Feature flag `--use-csl-v2`.
2. **PR 2:** Port `vecadd.mlir` test to v2 pipeline end-to-end. Both paths pass.
3. **PR 3:** Delete old `csl.*`, `csl_rt.*` dialects. One `-air-to-csl` pass, three emitters.

The v2 IR is **not backward compatible** with v1 (different op names, different structure).
Migration is a cut, not a gradual rename.

---

## 13. Implementation Notes (from build session)

Key MLIR 22 / C++ lessons from implementing V1:

| Problem | Fix |
|---|---|
| `attr-dict` before `$region` triggers format validator error | Use `hasCustomAssemblyFormat = 1` + manual parse/print |
| `SizedRegion<1>` with `{}` gives 0-block region | After `parseRegion`, call `body->emplaceBlock()` if empty |
| Block arg prints as `<block argument> of type...` | Use `printer.printOperand(arg)` not `printer.getStream() << arg` |
| `ExportOp::getSym()` returns `StringRef` not `FlatSymbolRefAttr` | MLIR 22: `FlatSymbolRefAttr` unwrapped at `getSym()` boundary |
| `getSymbolAttrName()` resolves to wrong class | Use `mlir::SymbolTable::getSymbolAttrName()` with explicit namespace |
| `mlir::hash_value` vs `llvm::hash_value` | Use `mlir::hash_value` for `!csl.comptime<T>` TypeStorage |
| FileCheck: `// CHECK-LABEL: module` fails with multiple test modules | Anchor on `csl.wafer @name` instead |
| `%{{.*}}` matches variable name AND trailing attrs | Use `%{{[^ ]*}}` to stop at whitespace |
