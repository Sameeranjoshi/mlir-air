# Chiplet Dialect — Design Decisions (Phase 1)

Resolves the open questions in `raw/idea.md` §4 and the brainstorm questions
(Q2, Q3) raised during the Sub-project A design review. One section per
decision: question, options, tradeoffs, chosen default, and what it would
take to revise.

## Q2 — Repository location

**Question.** Is `chiplet-mlir` a standalone repository or integrated into
mlir-air?

**Options considered.**

- **In-tree** under `mlir-air/mlir/Dialect/Chiplet/`.
- **Standalone out-of-tree repo** with its own LLVM pin.
- **Standalone repo that depends on mlir-air** as a consumed dependency.

**Tradeoffs.** In-tree is fastest but couples `chiplet`'s release cadence to
mlir-air; standalone-independent is cleanest but forces rebuilding AMDGPU
support code that mlir-air already has in `libairgpu`; standalone-with-dep
gets both but requires integration work upfront.

**Chosen default.** Standalone out-of-tree, living at
`mlir-air/chiplet-mlir/` in this repo for the research phase, building
against `mlir-air/llvm/install/`. The dialect depends on no mlir-air code in
Phase 1; mlir-air's `libairgpu` becomes relevant only in Sub-project D
(lowering).

**To revise.** Extract to its own git repository when the LLVM pin
decouples from mlir-air or when Sub-project D's integration surface is
stable.

## Q3 — Definition of done for Phase 1

**Question.** What is the acceptance line for the dialect foundation?

**Options considered.**

- **(a)** Minimal round-trip only, no verifiers beyond trivial, no passes.
- **(b)** Round-trip + scope-nesting and scope-compatibility verifiers + one
  canonicalization.
- **(c)** (a) + the Phase 2 partition pass's stage-1 (hardcoded) output.

**Tradeoffs.** (a) is the smallest coherent deliverable; (b) adds real work
whose value only appears once we have Phase 2 test inputs; (c) drifts into
Phase 2 scope.

**Chosen default.** (a). See `fleet_requirements.md` for the exact
requirements; see the spec for acceptance criteria.

**To revise.** Re-open when Phase 2 has emitted realistic IR and we have a
punchlist of real verifier violations; a "verifier hardening" spec is the
right container.

## §2 (of spec) — Relationship to AIR

**Question.** Is `chiplet` lowered from AIR, lowered to AIR, or a sibling
dialect?

**Options considered.**

- AIR → chiplet → ROCDL (chiplet as an intermediate level).
- chiplet → AIR GPU ops → ROCDL (chiplet reuses AIR's GPU backend).
- Sibling dialects (no lowering edges).

**Tradeoffs.** Routing through AIR would reuse AIR's async-token
infrastructure but force `chiplet`'s event-counter semantics into
async-token shape — a distortion. Sibling gives independence at the cost
of later integration work.

**Chosen default.** Sibling. Borrow AIR's idioms (symbolic iteration, late
binding, region-bearing task ops); share mlir-air's build infrastructure and
`libairgpu`; do not share dialects. The final lowering choice (direct to
ROCDL, or via Mirage MPK task descriptors, or via AIR's GPU pipeline)
remains open for Sub-project D.

**To revise.** When Sub-project D has concrete integration evidence —
either a Mirage MPK integration path that closes the deal, or empirical
pain that makes AIR's ACDG reuse compelling.

## §4.1 — Nominal vs symbolic scope

**Question.** Does a chiplet-scoped task carry a concrete partition id
(`xcd=3`) or a symbolic parameter (`%pid`)?

**Options considered.** Nominal (N ops, each with attr); symbolic (one op,
ssa partition_id); hybrid (nominal outer, symbolic inner).

**Tradeoffs.** Nominal is easy to verify and unambiguous but inflates IR by
the chiplet count and makes generic rewrites harder. Symbolic matches AIR's
`air.launch`/`air.herd` idiom and keeps the IR compact. Hybrid trades off
clarity for nothing in an MVP.

**Chosen default.** Symbolic. `chiplet.launch num_chiplets = 8` declares
the iteration domain; `chiplet.partition_id` returns the SSA value. Fleet's
scheduler maps partition ids to physical XCDs at runtime via `HW_ID`; the
IR never encodes this mapping.

**To revise.** Adding a `--chiplet-specialize` pass later is cheap if we
ever need nominal IR for debugging. The symbolic form is strictly more
general.

## §4.2 — Scope granularity

**Question.** How many levels in the scope enum, and do we conflate task
level with memory scope?

**Options considered.**

- Four levels, conflated (wavefront / cu / chiplet / device).
- Four task levels + separate memory-scope enum.
- Extended enum (add `mall`, `cluster`, `fabric_region`) for portability.

**Tradeoffs.** Four-conflated matches Fleet Table 3 exactly and keeps the
type system light. Separating task and memory is cleaner but forces
verifier complexity we're not building in the MVP. Extending the enum
commits to semantics we can't validate without NVIDIA Blackwell or
Cerebras hardware.

**Chosen default.** Four levels, conflated. `ChipletScopeAttr` with
`wavefront / cu / chiplet / device`. `chiplet.task`'s level uses this enum;
so does `#chiplet.scope<…>` as tensor encoding.

**To revise.** If portability work (NVIDIA / Cerebras) happens, add
`cluster` / `fabric_region` as separate enum cases alongside existing ones,
with a dedicated spec to validate the semantics. MALL as a memory-level
scope can be added the same way if a cost-model case ever depends on it.

## §4.3 — Capacity on types vs on passes

**Question.** Does `l2_capacity_bytes` live on each tensor type, or on the
module?

**Options considered.**

- On the type: `tensor<…, scope = chiplet, capacity = 4MB>`.
- On the module: `chiplet.target<…, l2_capacity_bytes = 4194304>`.
- Both via a view.

**Tradeoffs.** Type-level enables static "fits in capacity" verification —
compelling for a paper figure. But heavy on rewrite boilerplate, and the
verifier that would justify the extra bytes is not in Phase 1 scope.

**Chosen default.** Module-level. `chiplet.target` is the single source of
truth. Passes read it; types don't carry it.

**To revise.** Add a `--chiplet-verify-capacity` pass in a later
verifier-hardening spec. If empirically we need capacity to flow through
pattern rewrites, promote it onto the type behind a flag.

## §4.4 — Synchronization ops

**Question.** Distinct ops per sync pattern, or one op with a scope attr?

**Options considered.**

- Four distinct ops (`chiplet.sync.intra_xcd`, `chiplet.sync.cross_xcd`, …).
- One `chiplet.fence` with a `scope` attribute.
- Both, via an alias pass that specializes for codegen.

**Tradeoffs.** Distinct ops are easier to match in pattern rewrites that
target one pattern only; unified lets passes enumerate a single op type.
For Phase 1 we have no pattern rewrites, so the case for distinct ops
evaporates.

**Chosen default.** One `chiplet.fence` + `ChipletScopeAttr`. Plus a pair
of event ops (`chiplet.event.signal` / `chiplet.event.wait`) using an
opaque `!chiplet.event` type, modeled after Fleet's event counters.

**To revise.** If Sub-project D codegen shows one pattern dominates and a
distinct op shortens rewrites, split off a specific form (e.g.
`chiplet.fence.device` as a distinct op) while keeping the general form.

## §4.5 — Task body representation

**Question.** SPMD body with per-worker rank, or explicit broadcast/scatter
graph?

**Options considered.**

- SPMD, one body executed by all workers at the level, `chiplet.worker_id`
  gives rank.
- Explicit graph: a body that a runtime op expands into per-worker subtasks.
- SPMD at the IR level, lowered to explicit form before codegen.

**Tradeoffs.** SPMD is more compact and matches GPU programming; explicit
form is easier to optimize across workers (fusion, reordering). For
compiler-driven scheduling, the analysis is cleaner on SPMD bodies.

**Chosen default.** SPMD. `chiplet.worker_id : () -> index` gives rank in
the enclosing task. Explicit-form lowering, if useful, belongs in
Sub-project D.

**To revise.** If a mid-level analysis in Phase 2 or 3 wants per-worker IR
to reason about shared vs distinct tile loads, add an optional "expand
workers" pass that unrolls the SPMD body into a graph of per-worker ops.

## §4.6 — Mirage extension stance

**Question.** Extend the Mirage MPK runtime or build standalone?

**Options considered.**

- Extend Mirage: emit task graphs in Mirage's descriptor format, reuse its
  scheduler and code generation.
- Standalone: emit our own persistent-kernel runtime.

**Tradeoffs.** Extending Mirage is the fastest path to a working demo and
has already been validated by Fleet. Standalone gives us more control and
is better for multi-architecture portability, at a large cost.

**Chosen default.** Extend-first. Documented here, not coded in Phase 1.
The decision is revisited when Sub-project D starts, with Mirage source
open at `../mirage/`.

**To revise.** Sub-project D's kickoff. If Mirage's descriptor format
proves too narrow (e.g. it can't express K-split reductions or second-arch
primitives), fork to standalone and carry Mirage compatibility as a
test-only path.
