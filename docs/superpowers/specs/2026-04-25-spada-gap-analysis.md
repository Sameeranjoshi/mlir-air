# SPADA → mlir-air Gap Analysis

**Date:** 2026-04-25
**Status:** Analysis (read-only research artifact; no code changes proposed)
**Branch:** `air-to-fire`

**Question this report answers:** What is missing from mlir-air's CSL/AIR
dialect stack such that a developer could rewrite SPADA's sample kernels in
MLIR-AIR and run them through `air-opt | air-translate --emit-csl | cslc`
to a SUCCESS! on the simulator?

**References:**
- SPADA repo at HEAD: `git@github.com:spcl/spada.git`, commit `88e79fb` ("Prepare for Publication", PR #60). Cloned into `/tmp/spada` (116 Python source files, ~14 113 LOC across `spada/syntax/`, `spada/lowering/`, `spada/placement/`, `spada/runtime/`).
- SPADA paper: Gianinazzi, Ben-Nun, Hoefler, *SPADA: A Spatial Dataflow Architecture Programming Language*, arXiv:2511.09447, 2025 (referenced inline; not re-quoted here).
- mlir-air state: `/home/bricklib_dataflow/air-csl/mlir-air` on branch `air-to-fire`, last commit `da824329`.
- Companion specs: `2026-04-13-air-to-csl-vecadd-design.md`, `2026-04-15-csl-dialect-v2-design-r2.md`, `2026-04-17-csl-v4-subgrid-design.md`, `2026-04-21-csl-auto-vectorize-design.md`.

**Design rule applied to this report:** enumerate, do not propose
implementations. Each gap is classified Trivial / Medium / Large in scope
only; no plans, no patches.

---

## 1. SPADA repository at a glance

```
/tmp/spada
├── irspec/                     # mkdocs-rendered language reference
├── samples/
│   ├── stencils.py             # GT4Py stencil definitions
│   ├── advanced_stencils.py
│   ├── benchmarks/             # 5 kernels × 5 domain sizes (.spst/.sptl)
│   └── spatial/
│       ├── simple/             # add, copy, forward_sum, backward_sum,
│       │                       # mult_scalar, streaming_copy
│       ├── blas/               # axpy, matvec, gemv, gemv_twophase
│       ├── collectives/        # 15 files: broadcast, chain/tree/scalar/twophase
│       │                       # reduce, allreduce — both 1D and 2D
│       └── stencils/           # laplacian, laplacian_routed
└── spada/
    ├── cli/                    # sptlc + GT4Py driver
    ├── syntax/
    │   ├── common/             # base AST node, visitor
    │   ├── stencil_ir/         # GT4Py-level stencil IR
    │   ├── spatial_ir/         # SPADA-IR (the user-facing language)
    │   └── csl/                # CSL emission utilities + optimization passes
    ├── lowering/               # stencil → spatial → CSL pipeline
    ├── placement/              # checkerboard + heuristic PE placement
    └── runtime/                # cs_python launcher (calls Cerebras SDK)
```

LOC tally for the parts of SPADA most relevant to this analysis:

| File | LOC |
|---|---|
| `spada/syntax/spatial_ir/irnodes.py` | 1466 |
| `spada/syntax/spatial_ir/canonicalization.py` | 770 |
| `spada/lowering/spatial_ir_to_csl.py` | 1647 |
| `spada/lowering/stencil_to_spatial.py` (+ \_compute, \_dataflow, \_place, \_routing) | 1781 (sum) |
| `spada/syntax/csl/statements.py` | 714 |
| `spada/syntax/csl/dsd_ops.py` | 579 |
| `spada/syntax/csl/tasks.py` | 564 |
| `spada/syntax/csl/task_recycling.py` | 561 |
| `spada/placement/*.py` | ~1144 |
| `spada/runtime/runtime.py` | 459 |

**Compiler driver:** `sptlc` (entry point: `spada.cli.compiler:main`) takes a
`.sptl` file plus `--param NAME=VAL` flags and writes a directory of CSL
sources + `metadata.json`. The runtime entry point is `cs_python
spada/runtime/runtime.py output/ in_field.npy`.

---

## 2. SPADA compiler workflow

### 2.1 Pipeline diagram

```
  GT4Py stencils (Python)
            │
            ▼   spada/lowering/gt4py_to_stencil_ir.py  (551 LOC)
  Stencil IR  (.spst)         spada/syntax/stencil_ir/
            │                   • domain_inference, def_use, extent_inference
            │                   • SSA, type_inference, canonicalization
            ▼   spada/lowering/stencil_to_spatial.py        (255 LOC orchestrator)
            │   spada/lowering/stencil_to_spatial_place.py  (258 LOC)  – field placement
            │   spada/lowering/stencil_to_spatial_compute.py (827 LOC) – stencil → compute blocks
            │   spada/lowering/stencil_to_spatial_dataflow.py (225 LOC) – dataflow blocks
            │   spada/lowering/stencil_to_spatial_routing.py (402 LOC)  – checkerboard routing
            ▼
  Spatial IR  (.sptl, the user-facing SPADA language)
            │   spatial_ir grammar: spada/syntax/spatial_ir/language.lark
            │   parser:             spada/syntax/spatial_ir/parser.py + lark_to_ir.py
            │   IR nodes (44 dataclasses): irnodes.py
            │
            ▼   spada/lowering/spatial_ir_to_csl.py  (1647 LOC)
            │   canonicalize_kernel():
            │     • inline_metaprogramming  (canonicalization.py)
            │     • canonicalize_phases
            │     • reduce_streams
            │     • canonicalize_subgrids   (canonical_subgrids.py)
            │     • resolve_auto_hops
            │     • inline_phases
            │   ↓
            │   consolidate_rectangles_to_equivalence_classes()
            │   detect_stream_argument_extents()
            │   lower_bulk_communication()      → foreach/map iterators
            │   lower_array_assignment()
            │   lower_arguments_to_extern()
            │   copy_elimination.{remove_extern_field_copies,
            │                     eliminate_redundant_copies,
            │                     prune_unused_fields}
            │   _add_benchmarking_fields()
            │   _collect_colors_globally()      → channel→color allocator
            │   for each rect: generate_rectangle()
            │     • dsd_ops.detect_dsd_ops          (auto-vec)
            │     • tasks.build_task_dag           (event-driven scheduling)
            │     • task_recycling.recycle_task_ids (color budget mgmt)
            │     • prune_unused_fields
            │     • benchmarking.cycle_counter_instrumentation
            ▼
  Per-rectangle CSL  (one code_<x>_<y>.csl per equivalence class)
  + layout.csl       (set_rectangle, set_tile_code, set_color_config)
  + metadata.json    (host-runtime descriptor)
            │
            ▼   cslc                        (Cerebras toolchain)
            ▼   cs_python spada/runtime/runtime.py
                                            (459 LOC; SdkRuntime wrapper)
            ▼
  SUCCESS!
```

The pass list above is the literal call sequence at
`spada/lowering/spatial_ir_to_csl.py:38-44` (`canonicalize_kernel`) and lines
84-160 (`lower_spatial_ir_to_csl`).

### 2.2 IR design — Spatial IR

44 `@dataclass`-based AST nodes, all subclassing `SpatialNode(BaseNode)`.
The complete class list is at `spada/syntax/spatial_ir/irnodes.py:13-1466`:

| Node category | Classes |
|---|---|
| Literals & expressions | `ConstantLiteral`, `Parameter`, `Identifier`, `UnaryOperator`, `BinaryOperator`, `TernaryOperator`, `MultiplyAccumulateOperator`, `ArraySlice`, `Expression`, `RangeExpression`, `SubgridExpression` |
| Types | `StreamType`, `ArrayType`, `TypedIdentifier` |
| Declarations | `FieldDeclaration`, `RoutingHop`, `RoutingDeclaration`, `RelativeStreamDeclaration`, `MulticastRangeStreamDeclaration`, `ExternStreamDeclaration`, `StreamDeclaration` |
| Block scopes | `PlaceBlock`, `DataflowBlock`, `ComputeBlock`, `Phase`, `MetaForBlock` |
| Statements | `Statement`, `Completion`, `SendStatement`, `ReceiveStatement`, `ReceiveGenerator`, `ForeachStatement`, `MapStatement`, `ForStatement`, `AsyncBlock`, `AwaitCompletionStatement`, `AssignmentStatement`, `AwaitAllStatement` |
| Top-level | `KernelArgument`, `Kernel` |
| Visitors | `NodeVisitor`, `ScopedNodeVisitor`, `NodeTransformer` |

Each node carries optional `lineinfo: LineInfo` and is hand-validated via
its own `validate()` method. There is **no verifier-backed type system**;
type and access correctness is checked by ad-hoc analysis passes
(`spada/syntax/spatial_ir/analysis.py`). Compared to MLIR's TableGen-driven
verifier traits (`AttrSizedOperandSegments`, `IsolatedFromAbove`, `Symbol`,
`HasParent<>`), SPADA's IR is closer to a typed Python AST: invariants live
in the visitor passes, not the IR core.

The Lark grammar that produces these nodes is `language.lark` (213 lines)
and is the most concise specification of what the SPADA front-end actually
accepts.

### 2.3 Source-language constructs (per `language.lark`)

Quoting from `spada/syntax/spatial_ir/language.lark`:

| Construct | Grammar (excerpt) | Semantics |
|---|---|---|
| `kernel @name<P1, P2>(args) { body }` | line 178: `kernel : "kernel" ("@" bare_id)? parameters arguments kernel_body` | Top-level program with comptime parameters and stream/scalar arguments. |
| `place I, J in [a:b, c:d] { fields }` | line 181: `place_block : "place" typed_vars "in" subgrid_expression_2d place_body` | Declares per-PE fields over a 2D PE rectangle. Each PE in the rectangle gets its own copy of the listed fields (`f32[K] local`). |
| `dataflow I, J in [...] { streams }` | line 182 + 727 (`StreamDeclaration`) | Declares routed inter-PE streams over a PE rectangle. Streams are `relative_stream(dx, dy)` with optional `{ hops, channel }` routing. |
| `compute I, J in [...] { stmts }` | line 183 + 1085 (`ComputeBlock`) | Per-PE imperative body that may use `await receive`, `await send`, `foreach`, `map`, `for`, `async`. |
| `phase { ... }` | line 184 + 1181 (`Phase`) | A barrier-bracketed group of place/dataflow/compute blocks. All work in a phase completes before the next phase starts. |
| `for I in [a:b] { ... }` | line 157 / 186 / 187 — context-dependent; both meta-for (over phases) and runtime-for | Compile-time iteration when at kernel scope (`MetaForBlock`); runtime per-PE loop when inside a compute body (`ForStatement`). |
| `foreach k in [0:K] { ... }` | line 159, `foreach_stmt : prefix "foreach" typed_vars "in" generators compute_body` | Per-PE *parallel* iteration over a generator (range or `receive(stream)`). The fall-through vector idiom; SPADA's auto-vectorizer pattern-matches these to DSDs. |
| `map k in [0:K] { ... }` | line 158 | Per-PE *streaming parallel* iteration. Maps to CSL's `@map` builtin (closure over the body). |
| `await stmt` / `prefix : completion | "await"` | line 142 | Synchronous wait. `await receive(x, stream)` blocks until data arrives. |
| `completion f = map ... { ... }` | line 141; line 794 (`Completion`), 1026 (`AwaitCompletionStatement`) | Names an asynchronous completion handle. `await f` later joins. |
| `async { ... }` | line 160 | Forks the body asynchronously; pairs with `awaitall` (line 161) or per-completion awaits. |
| `relative_stream(dx, dy)` / `relative_stream([a:b], dy)` | line 121 + 645 | Single-target or multicast-range stream. Multicast form drives native CSL multicast routing. |
| `extern_stream(in)` / `extern_stream(out)` | line 122 + 704 | Streams that connect to host (memcpy_h2d/d2h). |
| `routing { hops = [(dx, dy), ...], channel = N }` | line 119 + 567 | Optional explicit routing — `hops = auto` triggers the checkerboard allocator. `channel = auto` lets the allocator pick. |
| `receive(stream)` as foreach generator | line 133 + 886 | Streaming generator that yields one element per arrival. |
| `readonly` / `writeonly` / `compiletime` annotations | line 42 + 1261 | Argument access annotations. `compiletime` arguments are inlined by `inline_metaprogramming`. |

### 2.4 Auto-vectorization in SPADA

Lives in `spada/syntax/csl/dsd_ops.py` (579 LOC) and is invoked from
`generate_rectangle()` in `spatial_ir_to_csl.py`. The tiered fallback
(SPADA paper §VI.D) is implemented across:

- **Tier 1 (DSD):** `dsd_ops.detect_dsd_ops` walks `foreach k in [0:K]`
  bodies, recognizes element-wise idioms over `f32[K]`-typed place fields,
  and rewrites them to CSL DSD builtins (`@fadds`, `@fmacs`, `@fmuls`,
  `@fmovs`, `@fnegs`, etc.). The legality predicate is "single-block
  body, indexing-only iterator usage, no control flow".
- **Tier 2 (`@map`):** when the body is *pure but non-idiom*, SPADA emits
  CSL's `@map(fn, src_dsd, dst_dsd)` with the loop body lifted to a closure.
- **Tier 3 (wavelet):** uses the CSL wavelet-trigger mechanism for
  small-data cases.
- **Tier 4 (scalar):** any loop that fails the legality predicate falls
  through unchanged. This is the correctness fallback.

Task-level scheduling is in `spada/syntax/csl/tasks.py` (564 LOC,
`build_task_dag`) and `task_recycling.py` (561 LOC, color-budget
optimization). SPADA fuses adjacent compatible tasks before color
assignment.

### 2.5 Color / routing allocation

Two layers:

1. **Stencil-graph placement** (`spada/placement/`, called from
   `stencil_to_spatial_place.py` for GT4Py inputs): builds a `StencilGraph`
   of fields + stencil edges, then `optimizer.best_of_k_placement(graph, k,
   shape)` runs `k` randomized trials of a heuristic placer
   (`mla.py` + `optimizer.py:21`) that picks the best-cost placement under a
   user-provided PE-grid shape. Output is a `PlacedStencilGraph`. Note the
   colored-graph helper at `optimizer.py:11 def color_graph(g)`.

2. **Channel/color assignment** for streams
   (`stencil_to_spatial_routing.py:13-150`): `KernelRouting.split_blocks()`
   first splits each per-PE block into even/odd halves along each
   communicating dimension (the *checkerboard decomposition*). Then
   `generate_routing()` picks colors per stream identifier.
   `_collect_colors_globally()` in `spatial_ir_to_csl.py:132` does the
   final global channel→physical-color rename (the global allocation
   step).

The current SPADA constraint (`stencil_to_spatial_routing.py:41-42`):
`"Only single hop communication is supported."` — multi-hop is achieved by
chaining single-hop relays, not direct multi-hop allocation.

### 2.6 Runtime layer

`spada/runtime/runtime.py` (459 LOC) is a Python module **invoked at host
runtime**, not at compile time. It reads `metadata.json` (written by
`sptlc` from `ProgramMetadata` at runtime.py:54-65) and uses the Cerebras
SDK's `sdkruntimepybind` (or the stub at `cerebras_runtime_stub.py` for
testing) to:

- Translate user numpy arrays to the right shape/dtype using the metadata.
- Issue `runner.memcpy_h2d(...)` for each input stream.
- Call `runner.launch("compute", nonblock=False)` to start the PE program.
- Issue `runner.memcpy_d2h(...)` for each output stream.
- Optionally read benchmarking cycle counters and emit timing.

The compile-time emission of CSL handles routing and placement statically
(`set_rectangle` + `set_tile_code` + `set_color_config` lines in the
emitted `layout` block — see lines 212-256 of `spatial_ir_to_csl.py`); the
runtime side is purely the host driver.

### 2.7 CSL emission — direct text, no intermediate

SPADA emits CSL **as text strings** directly from `spatial_ir_to_csl.py`
(lines 161-280) — there is *no intermediate textual or structural
representation between Spatial IR and CSL*. The emission style is:

- Build a `StringIO` for `layout.csl` (lines 145-280).
- Build per-rectangle `code_<x>_<y>.csl` files via
  `generate_rectangle(...)` (lines 134-142).
- The `CodeFile` dataclass (`spada/syntax/csl/codefile.py`) is just a
  filename + body-string pair; `write_code_to_files()` dumps them to disk.

The CSL helpers in `spada/syntax/csl/` are utilities for building these
strings:
- `statements.py:expr_to_csl` (line 22 of the same file imports it)
  formats expressions.
- `dsd_ops.py` matches DSD idioms and emits `@fmacs` etc.
- `tasks.py` builds the task DAG and emits `task @t() { ... }`.
- `task_recycling.py` rewrites task-id constants to compress the color
  budget.
- `prune_unused_fields.py` removes dead fields from `place` blocks.

This is **directly comparable to mlir-air's Phase-1 `-air-to-csl` text
emitter** (`mlir/lib/Conversion/AIRToCSL/AIRToCSLPass.cpp`). The key
structural difference: mlir-air *also* has a Phase-2 dialect path
(`csl.wafer / csl.program / csl_layout / csl_host`) lowered via
`air-translate --emit-csl`, which gives the dialect a verifier-backed
intermediate that SPADA does not have.

---

## 3. Representative SPADA samples surveyed

The following 7 samples span the difficulty spectrum.

### 3.1 Trivial: `samples/spatial/simple/add.sptl` (16 lines, single-PE rectangle of N×N)

```
kernel @add<N>(stream<f32, 1>[N, N] readonly a, stream<f32, 1>[N, N] readonly b,
               stream<f32, 1>[N, N] writeonly out) {
    place u16 i, u16 j in [0:N, 0:N] {
        f32 local_a;
        f32 local_b;
    }
    compute u16 i, u16 j in [0:N, 0:N] {
        await receive(local_a, a[i, j]);
        await receive(local_b, b[i, j]);
        local_a = local_a + local_b;
        await send(local_a, out[i, j]);
    }
}
```

Constructs: `kernel`, `place`, `compute`, `await receive` (host→PE),
`await send` (PE→host), scalar arithmetic. **Multi-PE** (N×N grid) but no
inter-PE communication.

Semantics: each PE gets one element of `a`, one of `b`, computes
`a[i,j] + b[i,j]`, writes to `out[i,j]`.

### 3.2 BLAS-1: `samples/spatial/blas/axpy.sptl` (22 lines, N×N grid, K-element local)

```
kernel @axpy<N, K>(stream<f32, K>[N, N] readonly x, f32 alpha,
                   stream<f32, K>[N, N] readonly y,
                   stream<f32, K>[N, N] writeonly out) {
    place u16 i, u16 j in [0:N, 0:N] {
        f32[K] local_x;
        f32[K] local_y;
    }
    compute u16 i, u16 j in [0:N, 0:N] {
        await receive(local_x, x[i, j]);
        await receive(local_y, y[i, j]);
        for u16 k in [0:K] {
            local_y[k] = alpha * local_x[k] + local_y[k];
        }
        await send(local_y, out[i, j]);
    }
}
```

Constructs: `kernel<N, K>`, `place` with array fields, `for` loop in
compute body (auto-vectorizer rewrites this to `@fmacs`), scalar
compile-time argument `alpha`. **Multi-PE, no inter-PE comm.**

Semantics: `y ← α·x + y` per PE on K-element vectors.

### 3.3 Reduction: `samples/spatial/simple/forward_sum.sptl` (16 lines)

```
kernel @add<N, K>(stream<f32, K>[N, N] readonly a,
                  stream<f32, K>[N, N] writeonly out) {
    place u16 i, u16 j in [0:N, 0:N] {
        f32[K] local_a;
    }
    compute u16 i, u16 j in [0:N, 0:N] {
        await receive(local_a, a[i, j]);
        for i32 k in [1:K] {
            local_a[k] = local_a[k-1] + local_a[k];
        }
        await send(local_a, out[i, j]);
    }
}
```

Constructs: scan-style `for` with loop-carried dependence (`local_a[k-1]`).
This is **not** auto-vectorizable to a DSD because of the
loop-carried-dependence — SPADA falls through to scalar (Tier 4).
**Multi-PE, no inter-PE comm.**

### 3.4 Stencil: `samples/spatial/stencils/laplacian.sptl` (76 lines)

The first 30 lines:
```
kernel @laplacian<I,J,K> (stream<f32>[I+2, J+2] readonly in_field,
                          stream<f32>[I, J] writeonly lap_field) {
  place i16 i, i16 j in [0:I+2, 0:J+2] { f32[K] local_input; }
  place i16 i, i16 j in [1:I+1, 1:J+1] { f32[K] local_result; }

  phase { compute i16 i, i16 j in [0:I+2, 0:J+2] {
      await receive(local_input, in_field[i, j])
  } }

  phase {
    dataflow i16 i, i16 j in [0:I+2, 0:J+2] {
       stream<f32> eastwards = relative_stream(+1, 0);
       stream<f32> westwards = relative_stream(-1, 0);
       stream<f32> northwards = relative_stream(0, -1);
       stream<f32> southwards = relative_stream(0, +1);
    }
    compute i16 i, i16 j in [0:1, 1:J] {
        await send(local_input, eastwards);
    }
    ...
    compute i16 i, i16 j in [1:I+1, 1:J+1] {
        completion f = map i32 k in [0:K] {
            local_result[k] = local_input[k] * 4;
        }
        await send(local_input, westwards);
        await f;
        await foreach i32 k, f32 x in [0:K], receive(westwards) {
          local_result[k] = local_result[k] - x;
        }
```

Constructs: multiple disjoint `place` blocks (forming the halo); `phase` to
serialize halo-load → halo-exchange → reduce; `dataflow` with 4 cardinal
streams; `relative_stream` (single hop); `completion`/`await f` for fork-join;
`map` (parallel, asynchronous); `foreach … receive(stream)` (streaming
generator). **Multi-PE, dense inter-PE comm in 4 directions.**

### 3.5 Collective broadcast: `samples/spatial/collectives/broadcast_1D.sptl` (70 lines)

(Quoted in §3.5 of source above; see file lines 1-70.) Constructs:
explicit `{ hops = [(1,0)], channel = 0 }` routing on red/blue
checkerboard streams, parity-conditional channel selection (`a if N%2==0
else b`), pipelined ranges with stride 2 (`[1:N-1:2, 0]`). **Multi-PE
linear chain.**

### 3.6 Collective reduce: `samples/spatial/collectives/tree_reduce_1D.sptl` (66 lines)

(Quoted above.) Constructs: `for stage in [0:L]` *meta-for* (compile-time
loop unrolling over phases), variable hops per stage `relative_stream(-(1<<stage), 0)`, dynamic
channel allocation (`channel = stage`), `hops = auto`. **Multi-PE binary
tree.** This is the cleanest example of SPADA's compile-time
metaprogramming.

### 3.7 Hardest: `samples/spatial/blas/gemv.sptl` (132 lines)

(Quoted above.) Constructs: 5 phases on a 2D `[0:PX, 0:PY]` grid, native
**multicast** (`relative_stream(0, [1:PY])` — the bracketed range means
"one-to-many"), *two-color* checkerboard pipelined chain reduction, mixed
scalar+block handling, deep `phase`-based serialization. This kernel uses
nearly every Spatial-IR construct.

### 3.8 (bonus) GT4Py-derived: `samples/spatial/stencils/laplacian_routed.sptl` (313 lines)

The auto-generated output of `gt4py_to_spatial`. Demonstrates that the
GT4Py path emits **versioned identifiers** (`i#3, j#3`, `in_0_0_0`),
**many disjoint place rectangles** (one per halo region), and explicit
routing with `hops = [(...)]`. Lines 6-32 alone declare 9 disjoint place
rectangles tiling a 130×131 fabric.

---

## 4. mlir-air CSL/AIR dialect surface today

### 4.1 CSL dialect ops (file refs below)

From `mlir/include/air/Dialect/CSL/CSLOps.td` — 13 active ops:

| Op | Line | One-liner |
|---|---|---|
| `csl.color` | 38 | Declare a comm color, optional fixed `id`. |
| `csl.route` | 63 | `in(DIR)` → `out(DIR)` direction pair. |
| `csl.func` | 96 | PE-level function (`fn name() void`). |
| `csl.task` | 123 | Event-driven task bound to a color (Wavelet trigger semantics — has the *handle* but no full lowering pattern wired). |
| `csl.return` | 155 | Terminator of `csl.func`/`csl.task`. |
| `csl.var` | 171 | PE-local variable; produces SSA value of memref type. |
| `csl.import_module` | 207 | `@import_module("name", params)`. |
| `csl.get_mem_dsd` | 244 | `mem1d_dsd` (rank-1) / `mem4d_dsd` (rank ≥ 2) from a (sub)memref. |
| `csl.builtin_call` | 293 | Catch-all for `@fadds`, `@fmacs`, `@fmuls`, `@fmovs`, `@fnegs`, plus any imported-module call. |
| `csl.wafer` | 349 | Top container (`arch = "wse3"`). |
| `csl.program` | 379 | Per-PE-template program with `%col: !csl.comptime<T>` block-arg params. |
| `csl.export` | 417 | Mark var/func host-visible; direction inferred by `-csl-infer-exports`. |
| `csl.layout` | 444 | Static placement container. |
| `csl.host` | 474 | Host-runtime container. |

Plus **commented-but-deferred**: `csl.get_fab_dsd`, `csl.mov` (lines
17-18 of CSLOps.td).

### 4.2 CSL types — `mlir/include/air/Dialect/CSL/CSLBase.td`

| Type | Line | Notes |
|---|---|---|
| `!csl.color` | 68 | Opaque color handle. |
| `!csl.dsd` | 74 | Single DSD type with **`mem1d`/`mem2d`/`fabin`/`fabout`** kind enum at line 108-113 — but only `mem1d`/`mem4d` are emittable today; `fabin`/`fabout` enum values exist but no op uses them. |
| `!csl.imported_module` | 80 | Handle for `@import_module`. |
| `!csl.route` | 86 | Direction pair handle. |
| `Direction` enum | 102 | `NORTH/SOUTH/EAST/WEST/RAMP`. |
| `Edge` enum | 130 | `LEFT/RIGHT/TOP/BOTTOM`. |

`!csl.comptime<T>` — referenced in `CSL_ProgramOp` block-arg syntax (line
391) but the type itself is a wrapper used only as a marker for "this
block argument lowers to a `param` declaration" — no full type def found
in `CSLBase.td`; check `CSLDialect.h` for the runtime registration.

### 4.3 csl_layout dialect — `CSLLayoutOps.td`

| Op | Line | Notes |
|---|---|---|
| `csl_layout.place` | 50 | Two forms: point `at (px, py)`, or range `over [lo:hi:stride, lo:hi:stride] vars (...) params {...}`. SPADA-style half-open ranges. |
| `csl_layout.export` | 100 | Re-export PE symbol with optional `kind = "func"`. |

### 4.4 csl_host dialect — `CSLHostOps.td`

| Op | Line |
|---|---|
| `csl_host.memcpy_h2d` | 47 |
| `csl_host.memcpy_d2h` | 85 |
| `csl_host.launch` | 123 |

### 4.5 AIR dialect ops — `mlir/include/air/Dialect/AIR/AIR.td`

| Op | Line | One-liner |
|---|---|---|
| `air.launch` | 22 | Host-level kernel launch, top of the spatial hierarchy. |
| `air.launch_terminator` | 98 | |
| `air.segment` | 111 | L2-scope subgrid (memtile equivalent on AIE; unused on CSL today). |
| `air.segment_terminator` | 219 | |
| `air.herd` | 231 | Compute-tile rectangle — SPADA's `place`+`compute` analogue. |
| `air.herd_terminator` | 328 | |
| `air.dma_memcpy_nd` | 340 | N-d strided DMA. |
| `air.wait_all` | 379 | Barrier-style join on async tokens. |
| `air.channel` | 405 | Symbol-typed inter-PE channel. |
| `air.channel.put` | 489 | |
| `air.channel.get` | 567 | |
| `air.execute` | 647 | Async wrapper around imperative ops. |
| `air.execute_terminator` | 693 | |
| `air.custom` | 712 | Generic op slot. |

### 4.6 Auto-vectorize coverage today

Per `2026-04-21-csl-auto-vectorize-design.md` §1 ("In scope"):

- **Element type:** `f32` only. `i16`/`i32`/`f16` deferred (§9.2).
- **Rank:** 1 (`mem1d_dsd`) and 2 (`mem4d_dsd`).
- **Stride:** unit + constant non-unit via `memref.subview`.
- **Idiom set:** elementwise binary add/sub/mul + FMA fusion + copy/unary
  (mov/neg) + scalar-broadcast operand. Pattern files in
  `mlir/lib/Dialect/CSL/Transforms/Patterns/`: `ElementwisePatterns.cpp`,
  `MovePatterns.cpp`, `Rank2Patterns.cpp`, `ScalarBroadcastPatterns.cpp`.
- **Out:** reductions with `iter_args` (norm_sq, dot), `@map` Tier-2,
  fabric DSDs, non-unit IV step, non-perfect rank-2 nests.

### 4.7 Currently emittable kernels (SUCCESS! today)

Eight `-csl-emit` end-to-end working kernels (per CLAUDE.md note plus
`mlir/test/Targets/CSLEmit/e2e/`):
`fadds, fsubs, fmuls, fmovs, fnegs, fmuls_scalar, fmacs_scalar,
stencil_fadds, switch_basic`. Plus 9 hand-written under
`mlir/test/Targets/CSLEmit/e2e/scientific/`:
`broadcast.mlir, diag.mlir, dot.mlir, even_saxpy.mlir,
matrix_row_col.mlir, norm_sq.mlir, saxpy.mlir, stencil_1d.mlir`.

All currently working kernels are **single-PE** (`width = 1, height = 1`
in the layout) and use only `csl_host.memcpy_h2d`/`memcpy_d2h` for I/O.
None of the working kernels exercises:

- Multi-PE placement with the range form of `csl_layout.place`.
- `csl.task` event-driven semantics (the op exists; no e2e test triggers a
  wavelet-bound task).
- Inter-PE routing via colors and `csl.route`.
- Fabric DSDs (the `fabin`/`fabout` enum values are declared but unused).

The matching `saxpy.mlir` file (lines 14-55) shows the canonical shape
of working input — flat `csl.wafer { csl.program { csl.var, csl.func,
csl.export } csl.layout { csl_layout.place ... at (0,0) } csl.host {
csl_host.memcpy_h2d/d2h, csl_host.launch } }`.

---

## 5. Cross-reference: SPADA construct → mlir-air equivalent

Status legend:
- **covered** — full lowering exists, e2e tested.
- **partial** — type or op exists in dialect, but no full lowering /
  pattern / e2e test.
- **missing** — neither op nor lowering exists.

| # | SPADA construct | Semantics | mlir-air equivalent | Status | Gap |
|---|---|---|---|---|---|
| 1 | `kernel @name<P1, P2>(args)` | Top-level program with comptime params | `csl.wafer { csl.program(%p: !csl.comptime<i16>) }` (CSLOps.td:349, 379) | covered | full single-PE; multi-PE comptime params on `csl_layout.place` exist (CSLLayoutOps.td:73) but no e2e exercising them |
| 2 | `place I, J in [a:b, c:d] { f32 v }` | Per-PE field declaration over rectangle | `csl_layout.place over [a:b, c:d] vars (%i, %j)` + `csl.var @v` inside `csl.program` (CSLLayoutOps.td:50) | partial | range-form parser exists; no e2e test that places per-PE over a non-trivial rectangle |
| 3 | `compute I, J in [...] { stmts }` | Per-PE imperative body | `csl.func @compute { ... }` inside `csl.program`; the per-PE rectangle comes from `csl_layout.place` over a range | partial | no e2e for non-trivial rect placement |
| 4 | `dataflow I, J in [...] { stream s = ... }` | Stream declaration block | **no equivalent op** — currently routes are written via raw `csl.color` + `csl.route` | missing | need a placement-time stream-decl op (or generate routes from a higher-level construct); see Gap-D below |
| 5 | `phase { ... }` | Barrier-bracketed group | **no equivalent op** in CSL dialect | missing | need either a `csl.phase` region op or AIR-level async-token chains as the canonical lowering |
| 6 | `for k in [0:K]` (meta-for, kernel scope) | Compile-time loop unrolling over phases | partially: `csl_layout.place` range form is iterative, but cross-phase meta-for unrolling is unsupported | missing | medium: need a `csl_layout.for` or compile-time evaluator |
| 7 | `for k in [0:K]` (compute-body) | Per-PE runtime loop | `scf.for` (auto-vec rewrites to DSD when legal) | covered | works for f32; integer support deferred per spec §9.2 |
| 8 | `foreach k in [0:K] { body }` (over range) | Per-PE parallel iteration; auto-vec target | `scf.for` + `-csl-auto-vectorize` to DSD | covered (Tier 1 partial) | only elementwise/FMA/copy idioms; no `@map` fallback |
| 9 | `foreach k in receive(stream)` | Streaming generator | **no equivalent** | missing | needs fabric-DSD type + a `csl.foreach_recv` or task-driven lowering |
| 10 | `map k in [0:K] { body }` | Closure over body, sent to `@map` builtin | **no equivalent** (Tier-2 explicitly deferred per §9.2) | missing | medium: new `csl.map` op + closure-lowering pass |
| 11 | `await receive(local, stream[i,j])` (host stream) | Receive from host-attached stream | `csl_host.memcpy_h2d` (CSLHostOps.td:47) | covered | works |
| 12 | `await send(local, out[i,j])` (host stream) | Send to host-attached stream | `csl_host.memcpy_d2h` (CSLHostOps.td:85) | covered | works |
| 13 | `await receive(local, color_stream)` (PE stream) | Receive from inter-PE stream | **no equivalent** | missing | needs fabric DSD + task lowering |
| 14 | `await send(local, color_stream)` | Send to inter-PE stream | **no equivalent** | missing | same |
| 15 | `relative_stream(dx, dy)` | Single-target inter-PE stream | partially: `csl.route` declares a single direction pair (CSLOps.td:63) but no relative-offset op | partial→missing | need a higher-level "stream from PE A to A+(dx,dy)" op that lowers to color+route on each affected PE |
| 16 | `relative_stream([a:b], dy)` | Native multicast | **no equivalent** | missing | new construct; multicast is a separate hardware primitive |
| 17 | `routing { hops = [(dx,dy),...], channel = N }` | Explicit multi-hop + fixed channel | `csl.color {id = N}` partial; multi-hop expansion missing | partial | the *color id* is expressible; the *route per hop along the path* is not auto-generated |
| 18 | `hops = auto` | Compiler picks routing | **no allocator** | missing | large: SPADA has a checkerboard splitter (`stencil_to_spatial_routing.py:KernelRouting.split_blocks`) + global allocator (`_collect_colors_globally`) — we have neither |
| 19 | `extern_stream(in)` / `extern_stream(out)` | Host-attached streams | `csl_host.memcpy_h2d`/`memcpy_d2h` | covered | direction inferred via `-csl-infer-exports` |
| 20 | `readonly` / `writeonly` annotations | Argument access mode | `csl.export {direction = "in"/"out"}` (CSLOps.td:417, set by `-csl-infer-exports`) | covered | works |
| 21 | `compiletime` annotation | Inline arg as comptime parameter | `csl_layout.place params {p = %i : i16}` (CSLLayoutOps.td:73) | partial | range-form binding parsed; no full inlining-pass-equivalent of SPADA's `inline_metaprogramming` |
| 22 | `completion f = map ...` + `await f` | Named async fork-join | `air.execute` produces `!air.async.token` + `air.wait_all` (AIR.td:647, 379) | partial | AIR-level async exists; not yet bridged to CSL dialect / no test of the bridging |
| 23 | `async { ... }` / `awaitall` | Fork bunch + join all | `air.execute` + `air.wait_all` | partial | same as above |
| 24 | `csl.task @t color(%c) { body }` | Wavelet-triggered task | `csl.task` op present (CSLOps.td:123) | partial | op skeleton; no auto-generation lowering, no e2e test |
| 25 | Native `mem1d_dsd` / `mem4d_dsd` | Memory DSD | `csl.get_mem_dsd` (CSLOps.td:244) | covered | works for ranks 1–4 with `memref.subview` |
| 26 | `fabin_dsd` / `fabout_dsd` | Fabric DSD on a color | enum values defined (CSLBase.td:110-111); op deferred (CSLOps.td:17) | partial | type-side acknowledged; no op, no emitter |
| 27 | `MultiplyAccumulateOperator` / FMA | DSD `@fmacs` | `csl.builtin_call "fmacs"` + auto-vec FMA fusion | covered | works (saxpy e2e green) |
| 28 | DSD `@map(fn, src, dst)` | Closure-bodied DSD op | **no equivalent** | missing | covered under #10 |
| 29 | `stream<f32>` (streaming type) | Lazy/buffer-backed stream | only `memref<NxT>` host-side; no streaming type | missing | semantics-only — no runtime "stream type" object exists |
| 30 | `stream<f32, K>[N, M]` (stream argument) | A per-PE stream argument over an N×M grid | partially: host-side memref + `csl_layout.place` range, but no implicit per-PE indexing | partial | in working samples the grid is 1×1 |
| 31 | `set_rectangle(W, H)` (CSL layout) | Bounding box of fabric region | `csl.layout {width, height}` (CSLOps.td:444) | covered | direct lowering; emits `@set_rectangle(W, H)` |
| 32 | `set_tile_code(x, y, "code.csl", params)` | Per-PE program file binding | `csl_layout.place @prog at (x,y)` and the range form | covered | works in single-PE tests |
| 33 | `set_color_config(x, y, color, .{ rx, ry, ramp })` | Per-PE routing config | partially: emitted by `-air-to-csl` for static routes; no dialect op that explicitly models it | partial | the emitter has logic, but the dialect lacks a first-class routing-config op consumable by passes |
| 34 | Checkerboard split (per-PE block bisection) | Splits each block into even/odd to eliminate routing conflicts | **not present** in any pass | missing | large: this is the algorithm in `stencil_to_spatial_routing.py:22-58`; a port would be a new `-csl-checkerboard-split` pass |
| 35 | `_collect_colors_globally` (channel→color allocator) | Global channel rename to physical colors | partially: `csl.color {id = N}` lets the user pin; the allocator itself is missing | partial → missing | medium pass to allocate IDs when not specified |
| 36 | `prune_unused_fields` | Dead-field DCE on place blocks | partially: standard MLIR DCE works on `csl.var` if the SSA result is unused; SPADA's pass works at field-decl level even when textually referenced but never read | partial | minor difference — likely a no-op gap |
| 37 | `task_recycling` (color-budget compaction) | Reuse a small color set across non-overlapping tasks | **not present** | missing | medium pass; depends on `csl.task` being driven |
| 38 | Benchmarking cycle counters | Auto-instrumentation | **not present** | missing | trivial: emit `time.start()`/`time.end()` builtins; medium if we want the metadata side |
| 39 | `metadata.json` (host runtime descriptor) | Output of `sptlc`, consumed by runtime.py | partially: `csl.host` lowers to `run.py` directly (no `metadata.json`) | partial | structural — different runtime model; not strictly a missing feature, but if SPADA samples expect this contract, it's a porting friction |
| 40 | `cs_python` runtime wrapper | Sets up `SdkRuntime`, calls memcpy/launch | mlir-air `air-translate --emit-csl` writes `run.py` directly | covered (different shape) | per CLAUDE.md, run.py works; passes simfab_numthreads=16/32 |

---

## 6. Three SPADA samples — proposed MLIR-AIR sketches

### 6.1 Easy: `samples/spatial/simple/add.sptl`

Re-quoted (16 lines):
```
kernel @add<N>(stream<f32, 1>[N, N] readonly a, ...) {
    place u16 i, u16 j in [0:N, 0:N] { f32 local_a; f32 local_b; }
    compute u16 i, u16 j in [0:N, 0:N] {
        await receive(local_a, a[i, j]);
        await receive(local_b, b[i, j]);
        local_a = local_a + local_b;
        await send(local_a, out[i, j]);
    }
}
```

Sketched MLIR-AIR equivalent (pseudo-MLIR, N = 4 example):
```mlir
csl.wafer @add {arch = "wse3"} {
  csl.program @pe(%i : !csl.comptime<i16>, %j : !csl.comptime<i16>) {
    %local_a = csl.var @local_a : memref<1xf32>
    %local_b = csl.var @local_b : memref<1xf32>
    csl.func @compute {
      %c0 = arith.constant 0 : index
      %a = memref.load %local_a[%c0] : memref<1xf32>
      %b = memref.load %local_b[%c0] : memref<1xf32>
      %s = arith.addf %a, %b : f32
      memref.store %s, %local_a[%c0] : memref<1xf32>
      csl.return
    }
    csl.export @local_a {alias = "a"}
    csl.export @local_b {alias = "b"}
    csl.export @compute {kind = "func"}
  }
  csl.layout {width = 4 : i64, height = 4 : i64} @layout {
    csl_layout.place @pe over [0:4, 0:4] vars (%i, %j)
                     params {i = %i : i16, j = %j : i16}
  }
  csl.host @main(%a_in: memref<16xf32>, %b_in: memref<16xf32>,
                 %out:  memref<16xf32>) {layout = @layout} {
    csl_host.memcpy_h2d %a_in to @layout::@local_a
        {px = 0 : i64, py = 0 : i64, width = 4 : i64, height = 4 : i64}
        : memref<16xf32>
    // ... b, launch, d2h ...
  }
}
```

**Missing pieces to make this run:**
- An e2e test exercising `csl_layout.place` *range* form (Gap #2, partial).
- Memcpy-h2d-to-rect emission (the `width=4, height=4` form distributes
  the host buffer across the 4×4 grid — verify the emitter handles this;
  current working tests are all 1×1).

**Estimate:** small. The dialect supports this; we just lack a
worked-end-to-end test.

### 6.2 Medium: `samples/spatial/blas/axpy.sptl`

Re-quoted: see §3.2.

Sketched MLIR-AIR equivalent:
```mlir
csl.program @pe(%i : !csl.comptime<i16>, %j : !csl.comptime<i16>) {
  %local_x = csl.var @local_x : memref<128xf32>     // K = 128
  %local_y = csl.var @local_y : memref<128xf32>
  %alpha   = csl.var @alpha   : memref<1xf32>
  csl.func @compute {
    %c0 = arith.constant 0 : index
    %a  = memref.load %alpha[%c0] : memref<1xf32>
    %xd = csl.get_mem_dsd %local_x : memref<128xf32> -> !csl.dsd
    %yd = csl.get_mem_dsd %local_y : memref<128xf32> -> !csl.dsd
    csl.builtin_call "fmacs"(%yd, %yd, %xd, %a)
        : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
    csl.return
  }
  // ...
}
csl.layout {width = N, height = N} @layout {
  csl_layout.place @pe over [0:N, 0:N] vars (%i, %j) params {...}
}
```

This is essentially `mlir/test/Targets/CSLEmit/e2e/scientific/saxpy.mlir`
generalized from 1×1 to N×N. **Missing:**
- Multi-PE placement with broadcast of `alpha` to every PE (SPADA inlines
  this scalar; mlir-air would need either a per-PE memcpy-h2d to alpha or
  a comptime parameter binding for alpha).

**Estimate:** small-to-medium. Mainly testing infrastructure; comptime
scalar broadcast is the only borderline-new-feature item.

### 6.3 Hard: `samples/spatial/collectives/tree_reduce_1D.sptl`

Re-quoted: see §3.6.

Sketched MLIR-AIR equivalent (only the structural skeleton — per-stage
lowering not attempted):
```mlir
// Need: for each compile-time stage in [0, L), generate a fresh program +
// route configuration. SPADA does this via meta-for; in MLIR we'd need
// either (a) ahead-of-time unroll into L csl.program clones + L
// csl_layout phases, or (b) a real csl_layout.for op.

csl.wafer @tree_reduce_1d {arch = "wse3"} {
  csl.program @sender_stage_0(%i : !csl.comptime<i16>) {
    %partial = csl.var @partial : memref<128xf32>
    %col_red = csl.color {id = 0 : i32} : !csl.color  // stage = 0
    csl.func @compute {
      // await send(partial, westward_jump)  →  fabout DSD on %col_red
      // ... !!! requires csl.get_fab_dsd, currently DEFERRED
      csl.return
    }
  }
  csl.program @receiver_stage_0(%i : !csl.comptime<i16>) { ... }
  // ... × L stages × {sender, receiver} = 2L programs

  csl.layout {width = 8 : i64, height = 1 : i64} @layout {
    // Stage 0:
    csl_layout.place @sender_stage_0   over [1:8:2, 0]
    csl_layout.place @receiver_stage_0 over [0:8:2, 0]
    // Stage 1:
    csl_layout.place @sender_stage_1   over [2:8:4, 0]
    csl_layout.place @receiver_stage_1 over [0:8:4, 0]
    // ... up to L
    //
    // !!! Missing: explicit route ops connecting senders→receivers per stage.
    // !!! Missing: a way to express "channel = stage" — channel collision
    //     across stages is hand-managed in SPADA via task_recycling.
  }
}
```

**Missing pieces to express this in mlir-air:**
- **Fabric DSDs** (Gap #26, missing): `csl.get_fab_dsd` is commented-out,
  the `fabin`/`fabout` enum values exist but no op consumes them. Without
  fabric DSDs, `await send(partial, color_stream)` has no lowering.
- **Routing-by-relative-offset** (Gap #15, missing): `relative_stream(-(1<<stage), 0)`
  with auto hops needs a multi-hop expansion that today's `csl.route` (a
  single direction pair) cannot express.
- **Color allocator + checkerboard splitter** (Gaps #18, #34, missing):
  the kernel uses `channel = stage` as a clever way to avoid color reuse;
  we'd either need to manually allocate or to port SPADA's allocator.
- **Meta-for at layout scope** (Gap #6, missing): emitting L distinct
  programs from one source needs either a Python-style preprocessor or a
  `csl_layout.for` op.
- **`csl.task` driven by an inter-PE color** (Gap #24, partial): the
  dialect has `csl.task @t color(%c) { ... }` but no e2e test, no
  patterns, no auto-generation.

**Estimate:** large. The kernel doesn't compose cleanly from the current
mlir-air primitives — it needs at minimum: fabric DSDs op, route-by-offset
op, color allocator pass, task-binding lowering, and a layout-scope
unroller. Each of those is medium individually; together, large.

---

## 7. Categorized gap list

### 7.1 Already covered (kernel features the stack handles end-to-end today)

- Single-PE `csl.program` with `csl.var` + `csl.func` + scalar arith.
- `csl.get_mem_dsd` over rank-1 / rank-2 memrefs (with `memref.subview`).
- `csl.builtin_call` invoking DSD builtins (`@fadds, @fsubs, @fmuls,
  @fmacs, @fmovs, @fnegs`) and imported-module members.
- `csl.import_module` for `<memcpy/get_params>` and `<memcpy/memcpy>`.
- `csl_layout.place @prog at (px, py)` (point form, single-PE).
- `csl.export` + `-csl-infer-exports` for direction annotation.
- `csl_host.memcpy_h2d`, `csl_host.memcpy_d2h`, `csl_host.launch`.
- `-csl-auto-vectorize` Tier-1 patterns over f32 (elementwise + FMA + copy
  + scalar broadcast).
- `air-translate --emit-csl` produces `pe.csl`, `layout.csl`, `run.py` with
  `simfab_numthreads=16/32`.
- AIR-level: `air.launch / air.segment / air.herd / air.dma_memcpy_nd /
  air.channel{,.put,.get} / air.execute / air.wait_all` for the GPU
  pipeline; the abstract spatial hierarchy is well-formed.

### 7.2 Partially covered (op/type skeleton present; full lowering or e2e missing)

- **`csl_layout.place` range form** with `vars` and `params`
  (CSLLayoutOps.td:50-92) — parser + verifier exist; no e2e exercising
  multi-PE placement.
- **`csl.task`** (CSLOps.td:123) — op declared with color binding; no
  pattern emits it from higher-level input, no e2e test triggers a
  wavelet-bound task.
- **`csl.color {id}` + `csl.route`** — declarations work; no allocator,
  no relative-offset or multi-hop expansion.
- **`!csl.dsd` of kind `fabin`/`fabout`** — enum values defined
  (CSLBase.td:110-111); no op constructs them, no emitter.
- **AIR `air.channel` + put/get** — full op set exists (AIR.td:405-647);
  no `-air-to-csl` lowering for the channel ops onto inter-PE colors.
- **Async tokens `!air.async.token`** — `air.execute` + `air.wait_all`
  work in the GPU pipeline; no plumbing to drive CSL `csl.task` triggers.
- **Comptime parameter inlining** — `csl_layout.place params {...}` parses
  the binding; no equivalent of SPADA's `canonicalization.inline_metaprogramming`
  pass that substitutes parameter values into the IR before emission.

### 7.3 Genuinely missing (no op, no lowering)

For each item: scope estimate (Trivial / Medium / Large) and where in the
pipeline it would slot.

#### Trivial

- **G1. Cycle-counter benchmarking instrumentation.** SPADA's
  `_add_benchmarking_fields` adds `time.start()`/`time.end()` builtins
  + an output field. mlir-air has no equivalent. Slot: `--csl-emit`
  emitter + a small dialect attribute (`{benchmark = true}` on
  `csl.func`).

#### Medium

- **G2. Inter-PE routing via relative offsets.** A higher-level op that
  expresses *"PE A sends to PE A + (dx,dy) on color C"* and lowers, on
  every affected PE, to the right `csl.route` direction pairs and color
  configs. Today `csl.route` is just a direction-pair handle. Slot: new
  `csl_layout.relative_route` (or similar), emitted before
  `set_color_config`.
- **G3. Multicast routing.** `relative_stream([a:b], dy)` is a hardware
  multicast primitive distinct from a single-target stream. Slot: new
  `csl_layout.multicast_route`.
- **G4. Color/channel allocator pass.** When `csl.color` lacks an `id`,
  pick one. Trivial in a 1-program toy case; medium to make it global
  (consider task overlap, color budget per PE, multi-stage kernels).
  Slot: between `-csl-infer-exports` and `--emit-csl`.
- **G5. `csl.get_fab_dsd` op + emitter for `fabin_dsd`/`fabout_dsd`.** The
  enum values exist (CSLBase.td:110-111) but no op constructs them. Needs:
  op def, verifier, custom assembly format, emitter case in
  `CSLEmitCommon.h`. Slot: `mlir/include/air/Dialect/CSL/CSLOps.td` next
  to `csl.get_mem_dsd` (and the deferred-op comment at line 17 already
  reserves the spot).
- **G6. `csl.task` driver lowering pattern.** Make the existing
  `csl.task` op reachable from higher-level input. Slot:
  `-air-to-csl` for the AIR → CSL path, with `air.channel.get` mapping to
  a task body. (See "wavelet-driven task" in tree_reduce_1D.)
- **G7. Streaming generator (`foreach k in receive(stream)`).** A loop
  whose iteration is driven by wavelet arrivals. Lowers to `csl.task`
  bodies. Depends on G5+G6. Slot: same as G6.
- **G8. Tier-2 `@map` lowering** (per `2026-04-21-csl-auto-vectorize-design.md`
  §9.2). New `csl.map` op + closure-lowering for pure-but-non-idiom
  loops. Slot: extend `-csl-auto-vectorize`.
- **G9. Reductions with `iter_args`** (per same spec §9.2). Ops like
  `@fadds` with a DSR accumulator. Sibling pass
  `-csl-recognize-reductions`. Slot: alongside `-csl-auto-vectorize`.
- **G10. Comptime parameter inlining pass.** SPADA's
  `inline_metaprogramming` substitutes parameter values into the kernel
  AST before lowering. mlir-air has parsed bindings on
  `csl_layout.place` but no pass that propagates them. Slot: between
  `-air-to-csl` and `--emit-csl`.
- **G11. `phase` (barrier-bracketed) construct.** SPADA serializes
  groups of place/dataflow/compute blocks. Today's mlir-air would have
  to express this with hand-managed AIR async tokens. Slot: either a
  `csl_layout.phase` region op or a documented pattern using
  `air.execute` + `air.wait_all`.
- **G12. Layout-scope meta-for** (`csl_layout.for` or compile-time
  unrolling). Without it, kernels with parameterized stage counts (e.g.
  `tree_reduce_1d<L>`) cannot be expressed in pure mlir-air; the user
  must hand-unroll. Slot: `csl_layout` dialect.
- **G13. `csl.host`-side memcpy-to-rect distribution semantics.**
  Currently working tests have width=height=1; for the SPADA samples we
  need a documented contract for how a host memref of shape `[N*K]`
  distributes across an `N`-PE rectangle (column-major vs row-major,
  per-PE chunk size). Likely already mostly works in the emitter — just
  needs spec + a test.

#### Large

- **G14. Checkerboard decomposition + global color allocator.** SPADA's
  `KernelRouting.split_blocks` (`stencil_to_spatial_routing.py:22-58`)
  splits each per-PE block into even/odd halves to make routing
  conflict-free, then `_collect_colors_globally`
  (`spatial_ir_to_csl.py:132`) does the rename. Together they're SPADA's
  *correctness-by-construction* guarantee. Porting this to MLIR is a
  major analysis pass with its own algorithmic content. Slot: new
  `mlir/lib/Dialect/CSL/Transforms/CSLCheckerboardSplit.cpp` +
  `CSLAllocateColors.cpp`.
- **G15. Stencil-graph placer (`spada/placement/`).** Heuristic
  `best_of_k_placement` (`optimizer.py:21`) over a `StencilGraph`. Only
  needed for the GT4Py front-end path (not for hand-written
  `.sptl` samples that already pin coordinates), so this is a *front-end
  feature gap*, not strictly a CSL-backend gap. Slot: would live next
  to a future stencil-IR dialect or as a Python pre-pass.
- **G16. Task DAG builder + task recycling.**
  `spada/syntax/csl/tasks.py` (564 LOC) builds the task DAG;
  `task_recycling.py` (561 LOC) reuses task IDs to fit the WSE color
  budget. Both depend on having a healthy `csl.task` lowering first
  (G6). Combined effort is large. Slot:
  `mlir/lib/Dialect/CSL/Transforms/CSLBuildTaskDAG.cpp` +
  `CSLTaskRecycling.cpp`.
- **G17. Stencil → Spatial-IR lowering pipeline.** The whole
  `spada/lowering/stencil_to_spatial*.py` family (1781 LOC) — would map
  to either a new MLIR `stencil` dialect lowering or to GT4Py-via-Python
  glue. Out of scope for "rewrite samples in MLIR-AIR" since the
  hand-written `.sptl` samples already start from Spatial-IR equivalents,
  but is the long-term path for production-stencil reuse.

### 7.4 Punch-list summary — the top-5 Medium items

(Prioritized for "unlock the most multi-PE samples per item.")

1. **G5 — `csl.get_fab_dsd` + fabric-DSD emitter.** Unblocks every
   sample that does `await send/receive` on a color stream
   (`broadcast_1D`, `tree_reduce_1D`, `chain_reduce_1D`, `allreduce_1D`,
   2D variants, gemv).
2. **G2 — relative-offset routing op + `set_color_config` emitter
   coverage.** Required to express `relative_stream(±1, 0)` in mlir-air.
3. **G6 + G7 — `csl.task` driver lowering & streaming-receive
   `foreach`.** The execution side of fabric DSDs; without this, the
   `await foreach k, x in [0:K], receive(stream)` idiom (used in *every*
   collective) has no semantics.
4. **G4 — color/channel allocator.** Local (per-program) suffices for
   most simple collectives; full global allocator (G14) only for fused
   stencils.
5. **G10 — comptime parameter inlining pass.** Needed for the
   parameterized samples (`<N>`, `<N, K>`, `<L, K>`); without it, the
   user has to hand-instantiate parameter values.

---

## 8. Concerns and caveats

- **SPADA repo is recent (commit 88e79fb, "Prepare for Publication").**
  The README references SDK 1.4 and WSE-2; mlir-air targets WSE-3. SDK
  semantics for fabric DSDs, multicast, and color budget should be
  re-checked against WSE-3 before any porting work.
- **The 2D variants of collectives** (`allreduce_2D`, `tree_reduce_2D`,
  `broadcast_2D_multicast`) were not opened in detail; the 1D forms
  exercise the same construct set, so the gap inventory likely covers
  them.
- **`samples/spst/` was referenced in the README but is empty in the
  cloned repo** (`find samples -type d` shows only the categories
  surveyed). Skipped that bucket.
- **`!csl.comptime<T>` type definition** was not located in the brief
  scan of `CSLBase.td`. It is referenced in `CSLOps.td:391` as the type
  of `csl.program` block-args. Whether it's a runtime-registered type or
  a textual sugar should be confirmed before relying on it for new ops.
- **No SPADA kernel was actually compiled through mlir-air.** Per the
  Step-7 / Step-8 instructions, this is analysis-only; the next-stage
  experiment is to take `samples/spatial/simple/add.sptl` and write its
  MLIR-AIR equivalent by hand.

---

## 9. References (file-line back-pointers)

### SPADA
- Grammar: `/tmp/spada/spada/syntax/spatial_ir/language.lark` (213 lines).
- Spatial-IR nodes: `/tmp/spada/spada/syntax/spatial_ir/irnodes.py` (1466 LOC).
- Lowering driver: `/tmp/spada/spada/lowering/spatial_ir_to_csl.py:38-280`.
- Routing: `/tmp/spada/spada/lowering/stencil_to_spatial_routing.py:13-150`.
- DSD detection: `/tmp/spada/spada/syntax/csl/dsd_ops.py` (579 LOC).
- Tasks: `/tmp/spada/spada/syntax/csl/tasks.py` (564 LOC).
- Task recycling: `/tmp/spada/spada/syntax/csl/task_recycling.py` (561 LOC).
- Placer: `/tmp/spada/spada/placement/optimizer.py:21`.
- Runtime: `/tmp/spada/spada/runtime/runtime.py:54-65` (metadata),
  `/tmp/spada/spada/runtime/runtime.py` overall (459 LOC).
- Sample kernels: `/tmp/spada/samples/spatial/{simple,blas,collectives,stencils}/*.sptl`.

### mlir-air
- CSL base/types: `/home/bricklib_dataflow/air-csl/mlir-air/mlir/include/air/Dialect/CSL/CSLBase.td:68-130`.
- CSL ops: `/home/bricklib_dataflow/air-csl/mlir-air/mlir/include/air/Dialect/CSL/CSLOps.td:38-498`.
- Layout ops: `/home/bricklib_dataflow/air-csl/mlir-air/mlir/include/air/Dialect/CSL/CSLLayoutOps.td:50-122`.
- Host ops: `/home/bricklib_dataflow/air-csl/mlir-air/mlir/include/air/Dialect/CSL/CSLHostOps.td:47-144`.
- AIR ops: `/home/bricklib_dataflow/air-csl/mlir-air/mlir/include/air/Dialect/AIR/AIR.td:22-712`.
- Auto-vectorize design: `/home/bricklib_dataflow/air-csl/mlir-air/docs/superpowers/specs/2026-04-21-csl-auto-vectorize-design.md` (esp. §9.2 deferred work).
- Auto-vec pattern files:
  `/home/bricklib_dataflow/air-csl/mlir-air/mlir/lib/Dialect/CSL/Transforms/Patterns/{Elementwise,Move,Rank2,ScalarBroadcast}Patterns.cpp`.
- Working e2e tests:
  `/home/bricklib_dataflow/air-csl/mlir-air/mlir/test/Targets/CSLEmit/e2e/*.mlir`
  (8 files) and
  `/home/bricklib_dataflow/air-csl/mlir-air/mlir/test/Targets/CSLEmit/e2e/scientific/*.mlir`
  (8 files including `saxpy.mlir` referenced in §6.2).
