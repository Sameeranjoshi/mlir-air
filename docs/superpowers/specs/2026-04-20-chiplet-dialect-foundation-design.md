# Chiplet Dialect — Foundation, Partition Pass, and Cost Model

**Status:** Draft (brainstorm output).
**Scope:** Phases 1–3 of the `raw/idea.md` backlog (dialect + partition-to-chiplet pass + cost model + schedule search). Phase 4 — ROCDL/ROCm lowering and hardware execution — is explicitly deferred to a later spec.
**Prereqs:** Fleet paper (`raw/asplos2027_fleet.pdf`, unpublished — reference by section/figure only, do not export); Mirage MPK source at `/scratch/general/vast/u1418973/mirage`; mlir-air branch `sam-chiplet-dialect`.

---

## 1. Goal

Produce a research proof-of-concept for chiplet-aware compilation: an MLIR dialect (`chiplet`), a pass that partitions a scope-agnostic `linalg.matmul` onto that dialect, and a cost model that picks the best schedule. The acceptance artifact is a single experiment that reproduces Fleet's reported ordering for `gate_up+SiLU` across batch sizes 1, 8, 32, 64 on MI350 — without hardware execution, using analytical prediction calibrated against Fleet Table 4.

Formally, by the end of this spec the following must hold:

1. The dialect can round-trip a hand-written IR expressing Fleet's gate_up+SiLU task graph.
2. A pass converts `linalg.matmul` (parameterized shape) into a `chiplet.launch` whose structure matches the hand-written IR.
3. A cost model scores each candidate schedule, and its top pick agrees with Fleet's measured best at each batch size.

Hardware execution, ROCDL codegen, and end-to-end latency measurement are deferred. This spec produces the research core; the next spec takes it to silicon.

## 2. Relationship to AIR

Chiplet is a **sibling dialect** to AIR, not a lowering target of it.

- **Why:** AIR's native model is AIE async-token scheduling. Fleet's Chiplet-task model is event-counter-based, scoped to a physical L2 partition, with explicit `buffer_wbl2` semantics for cross-XCD writeback. The two abstractions describe different hardware resource hierarchies; routing chiplet through AIR would force distortions in chiplet's type system that serve no research purpose.
- **What we reuse from mlir-air:** LLVM commit pin (builds stay aligned), `libairgpu` (AMDGPU support code, not AIR-dialect-specific), and the out-of-tree dialect scaffolding pattern. None of these require a lowering edge between AIR and chiplet.
- **Borrowed ideas, not code:** symbolic iteration (`air.launch` → `chiplet.launch`), region-bearing task ops (`air.herd`/`air.segment` → `chiplet.task`), and late-binding of worker IDs. `raw/idea.md` §7: "Borrow its ideas but the target hardware, capacity constraints, and sync semantics are different."
- **Open for the next spec:** whether chiplet lowers directly to the upstream GPU dialect → ROCDL, or into a Mirage MPK task-graph descriptor, or borrows AIR's `air-to-gpu` pipeline. We decide when we build Sub-project D with the Mirage MPK source in view and MI300X available for validation.

## 3. Deliverables

### Phase 1: Dialect foundation (idea.md Tasks 1–4)

1. **Repository `chiplet-mlir/`** (new, standalone out-of-tree), LLVM pin matches mlir-air's current pin.
2. **`ChipletDialect` shared library + `chiplet-opt` tool**.
3. **`test/examples/gate_up_silu.mlir`** — hand-written IR that round-trips and expresses Fleet's gate_up+SiLU task graph.
4. **`docs/fleet_requirements.md`** — Fleet's Figure 4a task graph, Figure 5 sync protocol, §4.1 cache-modifier policy transcribed as IR-level requirements. References Fleet by section/figure only; no quotation.
5. **`docs/design_decisions.md`** — one subsection per §4 open question (Q2, Q3, idea.md §4.1–4.6), each with options / tradeoffs / chosen default / what it would take to change.

### Phase 2: Partition-to-chiplet pass (idea.md Task 5)

6. **`ChipletPartitionPass`** — converts `linalg.matmul` to `chiplet.launch` with `N` Chiplet-tasks. Parameterized on:
   - partition axis (`N-split` | `K-split`)
   - traversal order (`M-major` | `N-major`)
   - number of chiplets (from `chiplet.target`)
   - batch size (from operand shape)
7. **Lit test pair** — `test/Transform/matmul_partition.mlir` input, and a FileCheck'd expected output matching the structure of `test/examples/gate_up_silu.mlir` modulo the body.
8. **`chiplet-opt -chiplet-partition`** pipeline command registered.

Pass begins hardcoded for the bs=1, gate_up shape, then extends to the full parameter space. Emits symbolic IR (one `chiplet.task` with `chiplet.partition_id`, not 8 unrolled clones) per §4.1 decision.

### Phase 3: Cost model + schedule search (idea.md Tasks 6–7)

9. **`tools/cost_model.py`** — analysis that takes a `chiplet.launch` (parsed via MLIR Python bindings) and returns per-task working set at each scope, predicted L2 hit rate, predicted HBM bytes per token. Uses the generalized hit-rate formula `L2_Hit = 1 − 1/min(W, m_tiles)` (Fleet Eq. 1), extended to partial residency under the streaming cache modifier.
10. **`tools/schedule_search.py`** — enumerates the action space (partition-axis × traversal-order × cache-modifier-policy = 12 configurations) for gate_up+SiLU at `bs ∈ {1, 8, 32, 64}`, scores each via the cost model, ranks them.
11. **`docs/schedule_search_results.md`** — table of predicted hit rate / HBM bytes / ranked schedules per batch size, alongside Fleet Table 4's measured values. The cost model's top-ranked schedule must match Fleet's measured best at each batch size (M-tile wins at bs≥32; M-tile and M-split tie at bs=1–16).
12. **`docs/cost_model_calibration.md`** — rank correlation (Spearman ρ) between predicted and measured orderings across the 12 configurations at each batch size. Target: ρ ≥ 0.8 per batch size; absolute-accuracy fitting is not a goal of this spec.

## 4. Design decisions (locked)

All decisions below are cheap to revise in a later sub-project.

| Ref | Question | Decision | Rationale |
|---|---|---|---|
| Q2 | Repo location | Standalone out-of-tree `chiplet-mlir`, LLVM pin shared with mlir-air | Independent for fast iteration; shared pin eases later integration |
| Q3 | Scope | Phases 1–3: dialect + pass + cost model; no lowering, no verifiers beyond trivial | Deliver the research PoC; hardware execution is a separate spec |
| §2 | AIR relationship | Sibling dialect, not lowering target | Abstractions describe different hardware models; coupling distorts both |
| §4.2 | Scope granularity | Four conflated levels: `wavefront / cu / chiplet / device` | Matches Fleet Table 3; portability deferred |
| §4.1 | Nominal vs symbolic | Symbolic; `chiplet.partition_id` SSA | Matches AIR idiom; downstream passes emit one op not eight |
| §4.3 | Capacity | Module-level `chiplet.target` attribute | Static-fit verification not an MVP goal |
| §4.4 | Sync ops | Unified `chiplet.fence {scope}` + `chiplet.event.signal`/`.wait` | Fewer ops; scope attribute selects Fleet's four sync patterns |
| §4.5 | Task body | SPMD with `chiplet.worker_id` | Matches GPU idiom and Fleet's per-worker execution |
| §4.6 | Mirage stance | Extend-first documented; no code here | Runtime decision belongs with lowering (Sub-project D) |

## 5. Repository structure

```
chiplet-mlir/
├── CMakeLists.txt
├── README.md                          # LLVM pin hash, build, test
├── build.sh
├── include/chiplet/
│   ├── Dialect/
│   │   ├── ChipletDialect.td
│   │   ├── ChipletTypes.td            # !chiplet.tensor, !chiplet.event
│   │   ├── ChipletAttrs.td            # ChipletScopeAttr, CacheModifierAttr, ChipletTargetAttr
│   │   └── ChipletOps.td              # 8 ops (see §6.3)
│   └── Transforms/
│       ├── Passes.td                  # ChipletPartitionPass registration
│       └── Passes.h
├── lib/
│   ├── Dialect/Chiplet/               # generated + hand-written op cpp
│   └── Transforms/
│       └── ChipletPartition.cpp       # Phase 2 pass
├── tools/
│   ├── chiplet-opt/                   # registers dialect + Phase 2 pass
│   ├── cost_model.py                  # Phase 3
│   └── schedule_search.py             # Phase 3
├── test/
│   ├── lit.cfg.py
│   ├── Dialect/Chiplet/               # one round-trip .mlir per op
│   ├── Transform/                     # partition pass input/expected
│   └── examples/gate_up_silu.mlir     # Phase 1 deliverable
└── docs/
    ├── fleet_requirements.md
    ├── design_decisions.md
    ├── schedule_search_results.md
    └── cost_model_calibration.md
```

## 6. Phase 1 — Dialect

### 6.1 Attributes

- **`ChipletScopeAttr`** — enum `wavefront / cu / chiplet / device`. Used on `chiplet.task {level}` and `!chiplet.tensor<…, scope>`.
- **`CacheModifierAttr`** — enum `cache_all / streaming / non_temporal`. Covers Fleet §4.1's three-tier policy; lowered to CDNA3/4 `sc1`/`sc0`/`nt` bits in Sub-project D.
- **`ChipletTargetAttr`** — struct `{num_chiplets: i64, workers_per_chiplet: i64, l2_capacity_bytes: i64}`. Attached to the top-level module.

### 6.2 Types

- **`!chiplet.tensor<shape, elem, scope>`** — shape + element type as in builtin `tensor`; `scope: ChipletScopeAttr`. Strided layout consistent with `memref`.
- **`!chiplet.event`** — opaque handle for cross-task dependencies.

### 6.3 Ops (eight)

| Op | Form | Role |
|---|---|---|
| `chiplet.launch` | region-bearing; `num_chiplets` attr; operand tensors; result tensors via terminator | top-level; binds task graph to a device; introduces `chiplet.partition_id` into its body |
| `chiplet.task` | region-bearing; `level: ChipletScopeAttr` | one task at the given level; SPMD across workers |
| `chiplet.partition_id` | `() -> index` | SSA index in `[0, num_chiplets)`; valid inside `chiplet.launch` body |
| `chiplet.worker_id` | `() -> index` | SSA index of the current worker within the enclosing task |
| `chiplet.copy` | `(src) { src_scope, dst_scope, modifier }` | data movement between scopes; strided-slice operands derived from `%pid`/`%wid` |
| `chiplet.fence` | `() { scope }` | synchronization at the named scope |
| `chiplet.event.signal` | `(!chiplet.event) -> ()` | producer side |
| `chiplet.event.wait` | `(!chiplet.event) -> ()` | consumer side |

### 6.4 Trivial verifiers only

- Scope/level attr values valid.
- `chiplet.partition_id` lexically inside a `chiplet.launch`; `chiplet.worker_id` lexically inside a `chiplet.task`.
- `event.signal` and `event.wait` reference a value of type `!chiplet.event`.

Nesting rules, scope compatibility on `chiplet.copy`, capacity fit, event lifetime — **not enforced here**; deferred to a later verifier-hardening sub-project.

### 6.5 Example IR — `test/examples/gate_up_silu.mlir` (sketch)

```mlir
module attributes {
  chiplet.target = #chiplet.target<num_chiplets = 8,
                                   workers_per_chiplet = 31,
                                   l2_capacity_bytes = 4194304>
} {
  func.func @gate_up_silu(
      %x     : !chiplet.tensor<1x4096xbf16, scope = device>,
      %w_gu  : !chiplet.tensor<4096x24576xbf16, scope = device>,
      %ev_in : !chiplet.event)
      -> (!chiplet.tensor<1x24576xbf16, scope = device>, !chiplet.event) {

    %out, %ev_out = chiplet.launch num_chiplets = 8 -> (...) {
      %pid = chiplet.partition_id : index

      chiplet.task { level = #chiplet.scope<chiplet> } {
        chiplet.event.wait %ev_in : !chiplet.event

        %w_slice = chiplet.copy %w_gu [ offsets derived from %pid ]
                   { src_scope = #chiplet.scope<device>,
                     dst_scope = #chiplet.scope<chiplet>,
                     modifier  = #chiplet.cache<streaming> }
          : !chiplet.tensor<4096x3072xbf16, scope = chiplet>

        // MFMA inner loop — opaque region body in this spec

        chiplet.task { level = #chiplet.scope<wavefront> } {
          // Fused SiLU element-wise body
        }

        chiplet.fence { scope = #chiplet.scope<chiplet> }
      }

      chiplet.event.signal %ev_out : !chiplet.event
    }

    return %out, %ev_out
  }
}
```

Deliberately under-specified: exact `chiplet.copy` slice syntax (strided offsets/sizes, `memref.subview`-style), the MFMA inner body (opaque or embedded linalg — we pick embedded linalg), and exact operand-plumbing on `chiplet.launch` (terminator-yield convention).

## 7. Phase 2 — Partition-to-chiplet pass

### 7.1 Input / Output

- **Input.** A module with a `chiplet.target` attribute and a `linalg.matmul` (or `linalg.batch_matmul`) whose operands are `tensor<…>` (no scope). Shapes can be static or dynamic.
- **Output.** The `linalg.matmul` replaced by a `chiplet.launch` containing one `chiplet.task {level = chiplet}` whose body expresses the chosen partition/traversal schedule. The body uses symbolic `partition_id`/`worker_id` to describe per-worker work.

### 7.2 Pass parameters

Passed via pass options or annotation on the `linalg.matmul`:

- `partition-axis ∈ {N, K}` (default: `N`, matches Fleet default for small batch)
- `traversal-order ∈ {M-major, N-major}` (default: `M-major`)
- `weight-cache-policy ∈ {cache_all, streaming, non_temporal}` (default: `streaming`)

### 7.3 Implementation stages

Concrete phasing to keep the pass tractable:

1. **Hardcoded path.** Accept only `bs=1`, the gate_up shape `[1, 4096] × [4096, 24576]`, and produce the exact IR in `test/examples/gate_up_silu.mlir` (SiLU excluded for now — the pass produces the GEMM Chiplet-task; SiLU fusion is a later refinement).
2. **Parameterized on shape.** Accept arbitrary `[M, K] × [K, N]` with N divisible by `num_chiplets`.
3. **Parameterized on partition axis.** Add K-split support (emits a `chiplet.task {level = device}` wrapper that reduces partial results).
4. **Parameterized on traversal order.** Emit the inner worker loop in M-major or N-major form.

Each stage has its own input/expected lit test pair. Stage 1 is the gate before Phase 3 begins; stages 2–4 can complete in parallel with Phase 3 since the cost model only reads the IR the pass emits.

### 7.4 What the pass does not do

- No fusion (SiLU stays as a separate `linalg.generic` for now; fusion is a follow-up).
- No autotuning — parameters are explicit inputs. Phase 3's schedule search is what picks them.
- No lowering (`chiplet.launch` stays as IR).
- No handling of operators other than matmul.

## 8. Phase 3 — Cost model + schedule search

### 8.1 Cost model (`tools/cost_model.py`)

**Input.** A module containing a `chiplet.launch`, parsed via MLIR Python bindings.

**Output.** Per Chiplet-task:

- Working set at each scope (`wavefront / cu / chiplet / device`), in bytes, computed from tensor shapes and the partition.
- Predicted L2 hit rate using `L2_Hit = 1 − 1/min(W, m_tiles)` (Fleet Eq. 1). Extended: when `weight-cache-policy = streaming`, lines are evicted after their reuse window, so the formula applies only within the `min(W, m_tiles)` tile group — no contribution from tile groups outside the active window.
- Predicted HBM bytes per token: `weight_bytes × (1 − L2_Hit) + activation_bytes + output_bytes`.

**Non-goals.** Absolute-accuracy fitting; modelling register pressure, MFMA latency, attention, or MoE; anything that requires runtime profiling.

**Language: Python.** Cost model iteration is analysis, not compilation; Python is faster to iterate than C++ and integrates cleanly with MLIR Python bindings. If this becomes a perf bottleneck later, port to a C++ analysis pass.

### 8.2 Schedule search (`tools/schedule_search.py`)

Enumerates the Cartesian product:

- `partition-axis ∈ {N, K}` — 2
- `traversal-order ∈ {M-major, N-major}` — 2
- `weight-cache-policy ∈ {cache_all, streaming, non_temporal}` — 3

= 12 schedules for each batch size. Evaluates for `bs ∈ {1, 8, 32, 64}` — 48 evaluations total. For each, runs the Phase 2 pass to produce IR, runs the cost model, tabulates predictions.

**Output:** `docs/schedule_search_results.md` containing:

- Full table of 48 entries (batch × schedule): predicted L2 hit rate, predicted HBM bytes, rank within that batch size.
- Comparison column against Fleet Table 4's measured ordering.
- One sentence per batch size explaining the top pick.

### 8.3 Calibration artifact

`docs/cost_model_calibration.md`:

- For each batch size, compute Spearman rank correlation between predicted ordering and Fleet Table 4's measured ordering.
- Target: ρ ≥ 0.8 per batch size. If the target is missed at a given batch size, document the mismatch, which formula term is responsible, and what it would take to fix (without fixing it in this spec — that's for a cost-model-hardening follow-up).
- Explicit statement: absolute accuracy (e.g., "predicted TPOT within 8% of measured") is **not a goal of this spec** and is deferred to a post-silicon sub-project.

## 9. Testing

- **Round-trip tests** (Phase 1): one `.mlir` per op under `test/Dialect/Chiplet/`; `chiplet-opt %s | FileCheck %s`.
- **Example round-trip** (Phase 1): `test/examples/gate_up_silu.mlir` round-trips.
- **Partition tests** (Phase 2): one input/expected pair per implementation stage (§7.3) under `test/Transform/`.
- **Cost model unit tests** (Phase 3): Python unit tests on synthetic `chiplet.launch` IR covering edge cases (`R=1`, `R=W`, `R=m_tiles`).
- **Integration test** (Phase 3): run schedule search, assert top pick per batch size matches Fleet Table 4.
- **No C++ unit tests.** Lit + Python is sufficient for this spec.

## 10. Success criteria (acceptance)

A reviewer can check all six:

1. `chiplet-opt test/examples/gate_up_silu.mlir` round-trips.
2. `docs/fleet_requirements.md` enumerates every op/attr in §6 with the Fleet construct it represents, plus a list of deliberately omitted constructs.
3. `docs/design_decisions.md` covers Q2, Q3, §2 (AIR relationship), and idea.md §4.1–4.6.
4. `chiplet-opt -chiplet-partition` transforms `linalg.matmul` (bs=1 gate_up shape) into an IR whose `chiplet.launch` structure matches the GEMM Chiplet-task portion of `test/examples/gate_up_silu.mlir` (SiLU fusion excluded — see §7.3). The expected output lives at `test/Transform/gate_up_partition_expected.mlir`.
5. `tools/schedule_search.py` produces a ranking whose top pick at `bs ∈ {1, 8, 32, 64}` matches Fleet Table 4's best (M-tile wins at bs ≥ 32; M-tile and M-split tie at bs = 1–16).
6. `docs/cost_model_calibration.md` reports Spearman ρ ≥ 0.8 at each batch size.

## 11. Out of scope (explicit)

Deferred to later specs:

- **ROCDL/AMDGPU lowering, `chiplet-translate`, hardware execution** — the next spec (Sub-project D) with MI300X/MI350X.
- **Mirage MPK runtime integration** — same.
- **AIR↔chiplet lowering edges** — evaluated in Sub-project D when we have concrete integration motivation.
- **Capacity-fit verification** and full nesting-rule enforcement — verifier-hardening spec, after Phase 2 gives realistic IR.
- **Attention, MoE, norm operators** — after the matmul pipeline is solid.
- **Portability to NVIDIA Blackwell or Cerebras WSE** — the stretch goal in `raw/idea.md` §1; takes its own spec.
- **Operator fusion** (e.g., gate_up + SiLU fused in one Chiplet-task) — deliberate follow-up; the MVP handles SiLU as a separate wavefront-task.
- **K-split cross-XCD reduction, `buffer_wbl2` emission** — Phase 2 stage 3 adds structural support; actual lowering of the reduction is Sub-project D.
- **Absolute-accuracy cost-model fitting** — post-silicon calibration.

## 12. Phasing for implementation

Phases are serial but each produces reviewable artifacts:

- **Phase 1** (~5–7 days): scaffolding + ODS + round-trip + docs.
- **Phase 2** (~5–7 days): partition pass, stage 1 gating Phase 3, stages 2–4 parallel.
- **Phase 3** (~5–7 days): cost model + schedule search + calibration.

Total ~15–20 days, one implementation plan. The implementation plan (next step, via `writing-plans` skill) may subdivide into three sub-plans if that makes review cleaner.

## 13. Open items (resolved during implementation)

Small enough to settle in-flight; do not require a spec revision:

- Exact printed syntax for `ChipletScopeAttr` / `CacheModifierAttr` (keyword vs parametric attribute form — pick whichever round-trips most readably).
- Whether `chiplet.task` returns tensor values via a terminator or via explicit result operands (sketch uses terminator).
- Lit scaffolding: reuse mlir-air's lit config or upstream's `standalone-opt` example.
- Whether the cost model parses IR via MLIR Python bindings or via a small C++ analysis entry point exposed to Python — bindings first; fall back only if they cause pain.

---

*End of spec.*
