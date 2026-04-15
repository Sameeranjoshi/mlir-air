# CSL Dialect Family — v2 Design

**Date:** 2026-04-15
**Status:** Design approved in brainstorming session. Ready for implementation planning.
**Predecessor:** [`2026-04-14-csl-redesign-design.md`](2026-04-14-csl-redesign-design.md) — evaluated Sketches A/B/C,
recommended Sketch C. This document is the concrete spec derived from that recommendation plus today's
additional brainstorming informed by the SPADA paper (arXiv:2511.09447) and the MLIR Stencils-CSL paper
(arXiv:2601.17754).
**Scope:** CSL dialect op surface, type system, lowering pipeline, emit targets. V1 = functions only.
No implementation code in this doc.

---

## 0. TL;DR

| Decision | Choice | Rationale |
|---|---|---|
| Programming model | **Actor model (Approach B)** — `csl.program` is a template placed onto PEs | CSL itself is template-first; lowest impedance to emit |
| Top container | **`csl.wafer(@arch)` inside standard `module {}`** | Like `aie.device` — the op is the program container, `module` is just the MLIR wrapper |
| Region structure | **Three siblings inside wafer**: `csl.program`, `csl.layout`, `csl.host` | Mirror CSL's 3-file structure in IR; each emitter reads one dialect |
| Comptime params | **Block args with `!csl.comptime<T>` type** | SSA-idiomatic MLIR; type marks compile-time-only constraint |
| Layout emit | **sdkLayout Python API** → `csl_layout.py`, imported by `run.py` | sdkLayout handles color allocation; no need for `csl-allocate-colors` pass |
| `csl.import` | **Implicit** — injected by emitter, not in IR | Eliminates boilerplate; `<memcpy/memcpy>` is always needed, always emitted |
| Export direction | **Auto-derived** by `-csl-derive-exports` pass from host transfers | User never writes `dir=`; direction is computed from `memcpy_h2d`/`d2h` in `csl.host` |
| SPMD placement | **`csl_layout.place_grid`** with subgrid range | Scales to 1M PEs; single template → N placements |
| Data placement | **`csl.var` for V1** (PE-local, already partitioned by AIR); **`csl.data` with sharding for future** | Reserve op-space for stencil-style global-data sharding |
| Inter-PE comms | **Deferred to v2** — reserve op-space for `csl.stream` with relative offsets | SPADA's `relative_stream(dx,dy)` is the right model |
| DSDs | **Deferred to v2** — inferred by `csl-elaborate-dsds` pass | No manual DSD ops for now; memref accesses vectorized by pass |
| V1 pass | **One `-air-to-csl` pass** emitting all three regions | Simpler than 3 passes; separate translate targets for emit |
| Boilerplate | **Template + pass approach** | `@export_symbol`, `comptime {}` blocks, `@set_tile_code` — all generated |

---

## 1. Design Philosophy

### Why Actor Model (B) over SPADA/AIE tile-centric (A)

The Cerebras WSE has up to 850,000 PEs. Approach A (tile-centric: enumerate each PE explicitly, bind
resources to tiles by SSA) does not scale — AIE can do `aie.tile(col, row)` for ~32 cores, but we cannot
write 850,000 `csl.pe(col, row)` ops.

**CSL itself is template-first (Actor model):** you write one `pe_program.csl` file and bind it to a
rectangle via `@set_tile_code`. Our IR mirrors this directly, with the lowest impedance path to emit.

SPADA (arXiv:2511.09447) and the Stencils-CSL paper (arXiv:2601.17754) use subgrid range expressions
for *source language* programming of WSE. But both lower to CSL's Actor model at the IR level. Our
CSL dialect sits *below* SPADA/stencil frontends and *above* CSL text — so Actor model is correct here.

**SPADA's three-block structure informs our layering:**

| SPADA construct | Maps to our dialect | Where |
|---|---|---|
| `place ... in [0:W, 0:H] { data decls }` | `csl.var` (V1) or future `csl.data @A shard_over` | `csl.program` |
| `dataflow ... in [0:W, 0:H] { streams }` | Future `csl.stream @s relative(dx, dy)` | `csl.program` |
| `compute ... in [0:W, 0:H] { fns }` | `csl.func`, `csl.task` | `csl.program` |
| Subgrid range `[0:W, 0:H]` as placement | `csl_layout.place_grid` | `csl.layout` |
| Checkerboard color assignment | `sdkLayout` Python API handles this | `csl.layout` emit |

**Why subgrid expressions also belong in the data layer (not just layout):**
The stencils paper shards global arrays across PEs with halo regions. SPADA's `place` block allocates
data per-PE subgrid. For stencil frontends (future), users write global `A[I][J]` and the compiler
shards it. We reserve `csl.data` op-space for this. For V1 (AIR input), data is already PE-local.

---

## 2. Top-Level Structure: `csl.wafer`

The program container op, sitting inside the standard MLIR `module {}` — exactly like `aie.device`
sits inside `module`. The `module` is just the MLIR wrapper; `csl.wafer` is the logical symbol table
for the whole WSE program.

```mlir
module {
csl.wafer @wseprog { arch = "wse3" } {

  // (1) PE kernel template(s)
  csl.program @vecadd_pe(%col: !csl.comptime<i16>, %width: !csl.comptime<i16>) {
    csl.var @a : memref<256xf32>
    csl.var @b : memref<256xf32>
    csl.var @c : memref<256xf32>
    csl.func @compute() {
      // plain memref / arith / scf
    }
    csl.export @a { alias = "a" }
    csl.export @b { alias = "b" }
    csl.export @c { alias = "c" }
    csl.export @compute { kind = func }
  }

  // (2) Layout — placement + routing intent (emits csl_layout.py via sdkLayout)
  csl.layout (1, 1) @main_layout {
    csl_layout.place @vecadd_pe at (0, 0) {
      col = 0 : i16, width = 1 : i16
    }
  }

  // (3) Host — runtime orchestration (emits run.py)
  csl.host @main(%a_in: memref<256xf32>, %b_in: memref<256xf32>,
                  %c_out: memref<256xf32>) {
    layout = @main_layout
  } {
    // No explicit load/run/stop — context manager handles lifecycle in emitter
    csl_host.memcpy_h2d %a_in to @main_layout::@a { px=0, py=0, width=1, height=1 }
    csl_host.memcpy_h2d %b_in to @main_layout::@b { px=0, py=0, width=1, height=1 }
    csl_host.launch @main_layout::@compute
    csl_host.memcpy_d2h @main_layout::@c to %c_out { px=0, py=0, width=1, height=1 }
  }
}
} // module
```

**Properties:**
- `arch` attribute: `"wse2"` or `"wse3"` — sets hardware constraints for passes
- `module {}` is the standard MLIR outer wrapper; `csl.wafer` is the actual program container
- Single symbol table inside `csl.wafer`: all three regions are siblings, cross-reference by symbol
- One `csl.wafer` per emitted WSE program. Multiple wafers in one module are allowed for testing.

---

## 3. `csl.program` — PE Kernel Template

Models one `pe_program.csl` file. **Has no coordinates, no rectangle, no host.** Pure compute template.

### 3.1 Op reference (V1)

| Op | Purpose | Emits to CSL |
|---|---|---|
| `csl.program @name(%params...)` | PE program module (SymbolTable region) | `pe_program.csl` file |
| `csl.var @sym : memref<...>` | PE-local mutable buffer | `var @sym: [N]T;` |
| `csl.const @sym : memref<...>` | PE-local read-only data | `const @sym: [N]T = ...;` |
| `csl.func @sym() { ... }` | PE function (plain MLIR body) | `fn @sym() void { ... }` |
| `csl.export @sym { alias?, kind? }` | Mark symbol host-visible | `comptime { @export_symbol(...) }` block |
| `csl.task @sym { color_sym = @c }` | PE task (reserved, V1 stub) | future: `task @sym` binding |

**NOT in the IR** (injected by emitter):
- `@import_module("<memcpy/memcpy>", ...)` — always emitted when any export or host transfer exists
- `comptime { @export_symbol(...) }` block — emitted from `csl.export` ops

### 3.2 Comptime Parameters

Block arguments with `!csl.comptime<T>` type. These map to CSL `param` declarations.

```mlir
csl.program @gemv_pe(
    %M: !csl.comptime<i16>,      // → param M: i16;
    %N: !csl.comptime<i16>,      // → param N: i16;
    %col: !csl.comptime<i16>     // → param col: i16;
) {
  // %M, %N, %col usable in comptime contexts within the body
  // e.g., csl.var @A : memref<?xf32>  where ? is derived from %M * %N at emit
}
```

**Design rationale for `!csl.comptime<T>` type over attribute dict:**
- Stays idiomatic MLIR (SSA block arguments)
- Usable in `affine.map` / `memref<[%N]xf32>` dynamic shapes within the program
- The type tag `!csl.comptime<T>` prevents use in runtime SSA contexts
- Maps cleanly to CSL's `param` declaration (compile-time constant)

**At placement time** (`csl_layout.place` or `place_grid`), concrete values are bound to these params
as attributes on the placement op. The emitter passes them to `@set_tile_code(x, y, file, .{M=4, N=6, ...})`.

### 3.3 Export Direction (auto-derived, not user-written)

```mlir
// User writes:
csl.export @a { alias = "a" }

// After -csl-derive-exports pass runs (scans csl.host):
csl.export @a { alias = "a", direction = in }
```

The `-csl-derive-exports` pass:
1. Scans `csl.host` for `csl_host.memcpy_h2d ... to @layout::@sym` → sets `direction = in` on
   the corresponding `csl.export` in the program
2. Scans `csl_host.memcpy_d2h @layout::@sym to ...` → sets `direction = out`
3. Symbols with no transfer: `direction = internal` (not exported to host, only between PEs)

The CSL emitter uses `direction`:
- `in` / `out`: emit as `var @sym: [N]T;` (mutable, host-accessible)
- `internal`: emit as `var` without `@export_symbol`

**Why auto-derive instead of user-written:** eliminates duplication that existed in the v1 design
(direction split between `csl.export_symbol` in kernel and `csl.export_name` in host). One source of
truth: the host transfer op. The program exports are direction-free until the pass runs.

### 3.4 Data Declarations

**V1 (PE-local, from AIR which already partitions data):**

```mlir
csl.var @a : memref<256xf32>      // PE-local mutable buffer
csl.const @lut : memref<16xi32>   // PE-local read-only constant
```

**Future: Global data with sharding (for stencil frontends):**

```mlir
// User writes global array, compiler shards it
csl.data @A : memref<1024x1024xf32> {
  shard_dim = 0,              // shard along rows
  halo = 1                    // 1-element halo for stencil neighbors
}
// After -csl-shard-data pass:
// → csl.var @A : memref<(1024/W + 2*halo) x 1024 x f32>  on each PE
```

The `csl.data` op is a placeholder with reserved op-name. Implementation deferred to when a stencil
frontend is added. The SPADA paper's `place` block and the Stencils-CSL paper's halo exchange are the
prior art for this transform.

### 3.5 Functions (V1)

```mlir
csl.func @compute() {
  // body is plain MLIR: arith, scf, memref, vector, affine
  // can reference csl.var results via SSA use-def
  // NO csl.* routing ops in V1
  %0 = memref.load %a[%i] : memref<256xf32>
  %1 = memref.load %b[%i] : memref<256xf32>
  %2 = arith.addf %0, %1 : f32
  memref.store %2, %c[%i] : memref<256xf32>
}
```

### 3.6 Reserved op-space (V1 stubs, not implemented)

| Op | Purpose | When needed |
|---|---|---|
| `csl.task @sym { color_sym = @c }` | Event-driven task bound to a color | When adding inter-PE comms |
| `csl.stream @sym { offset = (dx, dy) }` | Relative inter-PE stream declaration | When adding inter-PE comms |
| `csl.phase { ... }` | Temporal scope for routing declarations | When adding phased comms |

These ops are defined in TableGen (with a `// V1 stub: not yet lowered` comment) so that future work
can add their bodies without changing the dialect version or breaking existing tests.

---

## 4. `csl.layout` — Placement + Routing Intent

Models what goes into `csl_layout.py` (via sdkLayout Python API). **Has no compute.** Just rectangle,
placement, routing declarations, and host-visible exports.

### 4.1 Op reference

| Op | Purpose | Emits to |
|---|---|---|
| `csl.layout (W, H) @name { ... }` | Rectangle declaration, top container | `sdklayout.get_layout(W, H)` |
| `csl_layout.place @prog at (x, y) { params }` | Bind one PE at (x,y) to a program with params | `layout.set_tile_code(x, y, "prog.csl", ...)` |
| `csl_layout.place_grid @prog from (x0,y0) to (x1,y1) { params }` | Bind a subgrid (SPMD) | Expanded to N `place` ops by `-csl-elaborate-grid`, then emitted as loop |
| `csl_layout.color @sym` | Symbolic color declaration | `layout.get_color(id)` — ID assigned by sdkLayout |
| `csl_layout.route @sym { from, to, color }` | Routing declaration | `layout.set_fabric_*_route(...)` |
| `csl_layout.export @sym from @prog::@export { type, mutable }` | Host-visible symbol | `layout.add_field(@sym, type, mutable)` |

**V1 uses only `place` (or `place_grid`) and `export`.** Colors and routes are deferred to the first
multi-PE example that needs inter-PE communication.

### 4.2 SPMD via `place_grid`

For NxN (e.g., 4-PE row of gemv):

```mlir
csl.layout (4, 1) @gemv_layout {
  csl_layout.place_grid @gemv_pe
    from (0, 0) to (3, 0)
    params = {
      M = 4 : i16,
      N = 6 : i16,
      col = $col : i16        // $col is a per-PE placeholder, resolved by elaborate-grid
    }
  csl_layout.export @y from @gemv_pe::@y_ptr { type = memref<4xf32>, mutable }
  csl_layout.export @compute from @gemv_pe::@compute { kind = func }
}
```

The `-csl-elaborate-grid` pass turns `place_grid from (0,0) to (3,0)` into four `csl_layout.place`
ops, substituting `$col` with concrete values (0, 1, 2, 3). The emitter then loops over these ops.

### 4.3 sdkLayout emit strategy

`csl.layout` emits to `csl_layout.py`, not `layout.csl`. Rationale:
- sdkLayout Python API handles color allocation automatically — no `csl-allocate-colors` pass needed
- Python is more composable for routing rules than declarative `.csl` layout syntax
- `run.py` imports: `from csl_layout import get_layout`
- Separation of concerns: `csl_layout.py` grows independently if routing gets complex

Emitted `csl_layout.py` structure:

```python
# csl_layout.py  (renamed from layout.py to avoid Python module name collision)
from cerebras.sdk.runtime.sdkruntimepybind import SdkLayout, SdkTarget

def get_layout(target: SdkTarget) -> SdkLayout:
    layout = SdkLayout(target)
    # create_code_region(source_file, region_name, width, height)
    region = layout.create_code_region("gemv_pe.csl", "gemv", 4, 1)
    # Set constant params on all PEs in region
    region.set_param_all("M", 4)
    region.set_param_all("N", 6)
    # Set per-PE params (col varies per PE)
    for col in range(4):
        region.set_param(IntVector(col, 0), "col", col)
    # Place region at origin (0, 0) on the wafer
    region.place(0, 0)
    return layout
```

**Key SdkLayout API facts (verified from SDK docs):**
- `SdkLayout(target)` — takes an `SdkTarget` or `SdkExecutionPlatform`, NOT `(width, height)`
- `layout.create_code_region(source, name, w, h)` → `CodeRegion` — the central object
- `region.set_param_all(name, val)` — same value for all PEs in the region
- `region.set_param(IntVector(col, row), name, val)` — per-PE specialization
- `region.set_param_range(IntRectangle(...), name, val)` — rectangular sub-region
- `region.color(name)` → `Color` — creates a named color; physical ID is auto-allocated (0–23)
- `region.paint_all(color, routes)` — sets routing for all PEs
- `region.place(x, y)` — positions the region at wafer coordinates
- `layout.compile(out_prefix)` → `SdkCompileArtifacts` — compiles to ELF

---

## 5. `csl.host` — Runtime Orchestration

Models `run.py`. References the layout and its exports by symbol. **Has no compute, no PE-local ops.**

### 5.1 Op reference

The `csl.host` op's body emits inside a `with SdkRuntime(artifacts) as runner:` block. There are no
explicit load/run/stop ops — the Python context manager (`__enter__`/`__exit__`) handles the WSE
program lifecycle.

| Op | Purpose | Emits to Python |
|---|---|---|
| `csl.host @name(%args...) { layout = @L } { ... }` | Host runner entry point | `def main(...): artifacts = get_layout(target).compile(...); with SdkRuntime(artifacts) as runner:` |
| `csl_host.memcpy_h2d %src to @layout::@sym { px, py, width, height }` | Transfer host → device | `id = runner.get_id("sym"); runner.memcpy_h2d(id, src, px, py, w, h, N)` |
| `csl_host.memcpy_d2h @layout::@sym to %dst { px, py, width, height }` | Transfer device → host | `id = runner.get_id("sym"); runner.memcpy_d2h(dst, id, px, py, w, h, N)` |
| `csl_host.launch @layout::@sym` | RPC call into PE function | `runner.launch("sym", nonblock=False)` |

**`px`, `py`, `width`, `height`** are integer attributes on the memcpy ops specifying the PE rectangle.
`N` (elem_per_pe) is inferred at emit time from the memref element count in the source/dest operand
divided by `width * height`.

### 5.2 Symbol references for verification

`@main_layout::@a` is a **nested symbol reference** — the verifier checks that:
1. `@main_layout` is a valid `csl.layout` in the same `csl.wafer`
2. `@a` is a valid `csl_layout.export` inside that layout

This enables compile-time catching of: wrong export name, wrong type, missing export.
Today these errors surface only at `cslc` compile time. With verification, they surface in `air-opt`.

### 5.3 `run.py` emit

```python
# Emitted run.py
from csl_layout import get_layout
from cerebras.sdk.client import SdkRuntime
import numpy as np

def main(target, a_in: np.ndarray, b_in: np.ndarray, c_out: np.ndarray):
    artifacts = get_layout(target).compile("out/")
    N = a_in.size  # elem_per_pe for a 1x1 layout
    with SdkRuntime(artifacts) as runner:
        a_id = runner.get_id("a")
        b_id = runner.get_id("b")
        c_id = runner.get_id("c")
        runner.memcpy_h2d(a_id, a_in, 0, 0, 1, 1, N)
        runner.memcpy_h2d(b_id, b_in, 0, 0, 1, 1, N)
        runner.launch("compute", nonblock=False)
        runner.memcpy_d2h(c_out, c_id, 0, 0, 1, 1, N)
```

**No app-specific test data, validators, or hardcoded shapes.** `run.py` is a trivial orchestration
shell. The Python test harness (`pytest`) provides inputs and checks outputs.

**SdkRuntime lifecycle:** `with SdkRuntime(artifacts) as runner:` loads the compiled ELF, starts
the WSE program, and tears down on exit. No explicit `load()/run()/stop()` calls — the context
manager handles everything. `runner.get_id(sym)` resolves a symbol name to its compiled integer ID
before the memcpy calls.

---

## 6. Type System

| Type | Purpose | CSL analog |
|---|---|---|
| `!csl.comptime<T>` | Compile-time-only value (param) | `param x: T;` |
| `!csl.color` | Color resource handle (future) | `color` identifier (0–23) |
| `!csl.stream<T>` | Inter-PE stream handle (future) | fabric route + color binding |
| `!csl.dsd` | Data structure descriptor (future) | `dsd` (mem1d/mem4d/fabin/fabout) |

For V1, only `!csl.comptime<T>` is needed. The others are declared in the type registry but have no
lowering.

**`!csl.comptime<T>` semantics:**
- Only appears as block argument type on `csl.program`
- Cannot be stored to memory (`memref.store` of `!csl.comptime<i16>` is a verifier error)
- Can be used in `affine_map` and as `memref` dimension sizes (constant-folded at emit time)
- At layout placement, bound to a concrete integer attribute

---

## 7. Lowering Pipeline

```
vecadd.mlir (AIR)
  │
  │  -air-to-csl
  │    ├─ extracts air.herd body → csl.program (one per unique herd)
  │    ├─ maps air.herd shape + x_loc/y_loc → csl_layout.place_grid
  │    ├─ maps air.dma_memcpy_nd → csl_host.memcpy_h2d/d2h
  │    └─ maps air.launch context → csl_host.load/launch/stop
  ▼
csl.wafer { csl.program + csl.layout (place_grid) + csl.host (no directions yet) }
  │
  │  -csl-elaborate-grid
  │    └─ expands place_grid → N individual place ops
  ▼
csl.wafer { place ops for each PE }
  │
  │  -csl-derive-exports
  │    └─ scans csl_host.memcpy_h2d/d2h → sets direction on csl.export ops
  ▼
csl.wafer { fully annotated, ready for emit }
  │
  ├─ air-translate --emit-csl-program  →  vecadd_pe.csl   (one per csl.program)
  ├─ air-translate --emit-csl-layout   →  csl_layout.py   (sdkLayout calls)
  └─ air-translate --emit-csl-host     →  run.py          (SdkRuntime calls)

(All three can also be triggered by: air-translate --emit-csl-rt → runs all three)
```

### 7.1 `-air-to-csl` pass (replacing today's `-air-to-csl-dialect` + `-csl-to-csl-rt`)

**Pre-conditions (same as AIRToAIE):**
- `air.herd` has `x_loc`/`y_loc` attrs (set by a prior placement-hint pass, or defaulting to 0,0)
- `memref.alloc` inside herd has memory-space 2 (L1/PE-local)

**What it does:**
- Reads herd shape → emits `csl_layout.place_grid` with comptime param bindings
- Clones herd body into `csl.program`, replaces `%tx`/`%ty` with comptime param references
- Turns `memref.alloc` (mem space 2) → `csl.var`
- Turns `air.dma_memcpy_nd` (L3↔L2) → `csl_host.memcpy_h2d/d2h`

**What it does NOT do:**
- No hardcoded shapes, arg counts, or test data
- No routing or color assignment
- No direction inference (that's `-csl-derive-exports`)
- No DSD elaboration

### 7.2 Emit targets

| Target flag | Output file | Reads |
|---|---|---|
| `--emit-csl-program` | `<prog_sym>.csl` per `csl.program` | `csl.program` ops only |
| `--emit-csl-layout` | `csl_layout.py` | `csl.layout` ops only |
| `--emit-csl-host` | `run.py` | `csl.host` ops only |

`--emit-csl-rt` is **removed** — no backward compatibility needed at this stage of the project.
All three targets are independent; a test runner script invokes all three in sequence.

Each emitter is a focused walker reading only its own dialect namespace. Splitting the current
933-line `CSLRuntimeToPy.cpp` into `ProgramEmitter.cpp`, `LayoutEmitter.cpp`, `HostEmitter.cpp`
(~200 lines each) is a mechanical decomposition.

---

## 8. V1 Scope

Minimum op surface to get `vecadd.mlir` (1×1 herd) working end-to-end on the new IR.

**Ops needed for V1:**
- `csl.wafer`, `csl.program`, `csl.layout`, `csl.host` (structural containers)
- `csl.var` (one per buffer)
- `csl.func` (one per PE function)
- `csl.export` (without direction; derived by pass)
- `csl_layout.place` (single PE, 1×1)
- `csl_layout.export` (one per host-visible symbol)
- `csl_host.load`, `csl_host.run`, `csl_host.memcpy_h2d`, `csl_host.memcpy_d2h`,
  `csl_host.launch`, `csl_host.stop`
- `-air-to-csl` pass (replaces both current conversion passes)
- `-csl-derive-exports` pass
- Three emitters in `CSLToPy.cpp`

**Explicitly out of scope for V1:**
- `csl.task`, `csl.stream`, `csl.phase` (inter-PE, async)
- `!csl.color`, `!csl.dsd`, `!csl.stream<T>` types
- `csl_layout.color`, `csl_layout.route` (routing)
- `csl_layout.place_grid` + `-csl-elaborate-grid` pass (needed for NxN; add in v1.1)
- `csl.data` sharding (stencil frontends)
- DSD elaboration pass

**V1.1 (second milestone — 2×2 or 1×4 multi-PE):**
- `csl_layout.place_grid` + `-csl-elaborate-grid`
- `!csl.comptime<T>` block arguments wired end-to-end
- `csl_layout.color` + sdkLayout color assignment
- Basic inter-PE: `csl.stream` + `csl.task`

---

## 9. Data Sharding (Future — Stencil Frontends)

For frontends that write global arrays and rely on the compiler to shard:

```mlir
csl.program @laplacian_pe(%row: !csl.comptime<i16>, %W: !csl.comptime<i16>) {

  // Global array with sharding annotation (future csl.data op)
  csl.data @in_field : memref<1024x1024xf32> {
    shard_dim = 0,          // shard rows across PEs
    halo = 1                // 1-row halo for stencil neighbors
  }

  // After -csl-shard-data pass:
  // csl.var @in_field : memref<(1024/H + 2)x1024xf32>

  csl.func @laplacian() {
    // reads from @in_field (local slice) and neighbor slices via streams
  }
}
```

The `-csl-shard-data` pass:
1. Reads `shard_dim` and `halo` from `csl.data`
2. Reads the placement grid dimensions from `csl.layout`
3. Replaces `csl.data` with `csl.var` of appropriate sliced memref type
4. Inserts `csl.stream` declarations for halo exchange communication

This is the SPADA `place` block generalized. The stencil-CSL paper (arXiv:2601.17754) provides
prior art for the halo exchange pattern.

---

## 10. Inter-PE Communication (Future — v1.1+)

Modeled after SPADA's `relative_stream` — no absolute PE coordinates in the program:

```mlir
csl.program @stencil_pe(%col: !csl.comptime<i16>) {

  // Stream declarations: relative offset, no absolute coords
  csl.stream @east  { offset = (1, 0) }    // send to PE at (col+1, row)
  csl.stream @west  { offset = (-1, 0) }   // receive from PE at (col-1, row)

  csl.task @recv_west { color_sym = @west_color } {
    // triggered when data arrives from west neighbor
  }

  csl.func @send_east() {
    // send data to east neighbor on @east stream
  }
}
```

The layout layer binds abstract stream symbols to concrete colors:

```mlir
csl.layout (W, 1) @stencil_layout {
  csl_layout.color @east_color
  csl_layout.color @west_color
  csl_layout.route @east_color { direction = EAST }
  csl_layout.route @west_color { direction = WEST }
  // sdkLayout assigns physical color IDs; checkerboard decomposition handles conflicts
}
```

SPADA's checkerboard decomposition algorithm (Section VI-B of arXiv:2511.09447) is directly
applicable here: even/odd PE parity → distinct color IDs → conflict-free routing by construction.

---

## 11. Migration from v1 Dialect

The v1 dialect (`csl.*` + `csl_rt.*`) stays in the tree during migration. Migration plan:

1. **PR 1:** Add new dialect family (`csl_program.*`, `csl_layout.*`, `csl_host.*`) alongside old.
   All existing tests unchanged. Feature flag `--use-csl-v2` on `-air-to-csl`.
2. **PR 2:** Port `vecadd.mlir` test to v2 pipeline end-to-end. Both paths pass.
3. **PR 3:** Delete old `csl.*`, `csl_rt.*` dialects and their passes/emitters.
   One `-air-to-csl` pass, three emitters.

The v2 IR is **not backward compatible** with v1 (different op names, different structure). The
migration is a cut, not a gradual rename.

---

## 12. Open Questions Resolved (from previous design doc §12)

| Q# | Question | Decision |
|---|---|---|
| 1 | Sketch A or C? | **Sketch C** (three-layer split), implemented as `csl.wafer` with siblings |
| 2 | Three-submodule shape acceptable? | **Yes** — siblings inside `csl.wafer`, not nested modules |
| 3 | `csl.task` from day one? | **Reserved stub in TableGen, no lowering in V1** |
| 4 | DSDs as types or ops? | **Neither in V1.** Future: `csl-elaborate-dsds` pass; keep memrefs, pass vectorizes |
| 5 | Comptime: dict vs block-args? | **Block args with `!csl.comptime<T>` — in V1** |
| 6 | Color IDs: explicit or pass-assigned? | **sdkLayout-assigned** — no explicit IDs in IR; `region.color("name")` auto-allocates |
| 7 | One `csl_program.module` per kernel or per PE source file? | **One per kernel symbol** — SPMD is trivial |
| 8 | AIRToCSL split: 3 passes or 1? | **One `-air-to-csl` pass** + separate translate targets |
| 9 | Migration: parallel or replace? | **Parallel for two milestones** (feature flag), then delete |
| 10 | `--emit-csl-rt` stays as front door? | **Removed** — no backward compat needed; three individual targets only |
| 11 | Export direction: user-written or derived? | **Auto-derived** by `-csl-derive-exports` pass |
| 12 | `csl.wafer` vs `module`? | **`csl.wafer` inside standard `module {}`** — same pattern as `aie.device` |
| 13 | `layout.py` naming? | **`csl_layout.py`** — avoids Python module name collision |

## 13. Remaining Open Questions

All previously listed open questions have been resolved. Remaining:

1. **`SdkRuntime` API version.** The appliance API (`cerebras.sdk.client.SdkRuntime`) uses context
   manager with `with SdkRuntime(artifacts) as runner:`. The older simulator API uses `load()/run()/stop()`.
   The emitter should target the appliance API (newer, context manager). If simulator compat is needed,
   add an emitter flag `--target-sdk-api={appliance,simulator}` at emit time.

2. **`SdkTarget` threading.** `get_layout(target)` needs a `target` argument. In the emitter,
   this comes from the `arch` attribute on `csl.wafer`. Define a mapping:
   `arch = "wse3"` → `SdkTarget.WSE3`, etc.

3. **`hstack`/`vstack` for multi-region programs.** When multiple `csl.program` types are placed
   on non-overlapping sub-grids, should the emitter use `layout.hstack([r1, r2])` to compose them,
   or let each region `place()` independently? Decide when the first 2-program example is built.

---

## 14. Reference: Concept Mapping Table

Complete cross-reference of the design space:

| Concept | CSL language | SPADA (arXiv:2511.09447) | Stencils-CSL (arXiv:2601.17754) | AIE dialect | Our dialect |
|---|---|---|---|---|---|
| Hardware container | `layout { @set_rectangle }` | Implicit (whole fabric) | Implicit | `aie.device(@arch)` | `csl.wafer(@arch)` |
| PE handle | `@set_tile_code(x, y, ...)` | `i, j in [0:W, 0:H]` | Grid coordinate | `aie.tile(col, row)` | `csl.pe(col, row)` in layout only |
| PE program | `pe_program.csl` file | `compute` block body | Stencil kernel | `aie.core(%t) { }` | `csl.program @sym(comptime args)` |
| Compile-time param | `param x: T;` | Template arg `<K>` | Domain parameter | N/A (each core has own ELF) | `%x: !csl.comptime<T>` block arg |
| PE-local data | `var a: [N]f32;` | `f32[K] a` in place block | PE-local buffer | `aie.buffer(%t)` | `csl.var @a : memref<NxT>` |
| Global sharded data | N/A (manual per-PE) | `place` block + subgrid | Tiled field array | N/A | Future `csl.data @A shard_over` |
| PE function | `fn compute() void` | `compute` block | Stencil body | `aie.core` body | `csl.func @compute()` |
| PE task | `task @t fn(...) void` | N/A explicit | N/A | DMA-completion handler | `csl.task @t { color_sym }` (stub) |
| Export to host | `@export_symbol(ptr)` | Kernel argument | Output field | `aie.shim_dma_allocation` | `csl.export @sym { alias }` |
| Placement | `@set_tile_code(x, y, file, params)` | Implicit (subgrid range) | Auto-placed | `aie.core(%tile)` body | `csl_layout.place @prog at (x,y) {params}` |
| SPMD placement | N/A (manual loop) | Subgrid range `[0:W, 0:H]` | Tiled domain | `air.herd [Cx, Ry]` | `csl_layout.place_grid from (x0,y0) to (x1,y1)` |
| Color / channel | `color = @get_color(id)` | Auto-assigned (checkerboard) | Auto-assigned | `aie.flow` | `csl_layout.color @sym` (ID from sdkLayout) |
| Routing | `@set_color_config(...)` | Auto (checkerboard decomp) | Auto | `aie.switchbox` | `csl_layout.route @sym { dir }` |
| Inter-PE stream | Fabric DSD + color | `stream<T> s = relative_stream(dx, dy)` | Halo exchange | `aie.objectfifo` | Future `csl.stream @s { offset=(dx,dy) }` |
| Host transfer | `memcpy_h2d` / `memcpy_d2h` | Kernel in/out args | Field I/O | `aiex.npu.dma_memcpy_nd` | `csl_host.memcpy_h2d/d2h` |
| Host launch | `runner.launch(sym)` | Implicit kernel exec | Implicit | `aiex.npu.*` sequence | `csl_host.launch @sym` |
| Layout file | `layout.csl` | N/A | N/A | `aie_inc.cpp` (C++ config) | `csl_layout.py` (sdkLayout API) |
| Runtime file | `run.py` | JSON metadata + runtime | Python runner | xclbin + host C++ | `run.py` (SdkRuntime) |

---

*End of v2 design spec. Next step: invoke `superpowers:writing-plans` to create the implementation plan.*
