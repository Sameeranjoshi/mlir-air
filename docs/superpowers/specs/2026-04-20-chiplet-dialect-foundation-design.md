# Chiplet Dialect — Foundation (Sub-project A) Design

**Status:** Draft (brainstorm output).
**Scope:** Sub-project A of the backlog in `raw/idea.md`. Later sub-projects (partition-to-chiplet pass, cost model, ROCDL lowering) are out of scope and will have their own specs.
**Prereqs:** Fleet paper (`raw/asplos2027_fleet.pdf`, unpublished — do not export); Mirage MPK source at `/scratch/general/vast/u1418973/mirage`; mlir-air branch `sam-chiplet-dialect`.

---

## 1. Goal

Produce a standalone out-of-tree MLIR dialect, `chiplet`, that can **round-trip a hand-written IR expressing Fleet's gate_up+SiLU task graph** at batch size 1 on AMD MI350. That is the single, falsifiable acceptance criterion for this sub-project.

Everything else — partition-to-chiplet transformation, cost-model-driven schedule search, ROCDL lowering via Mirage MPK, capacity-fit verification, attention, MoE, a second architecture target — belongs to later sub-projects and is explicitly out of scope here.

## 2. Deliverables

1. **Repository `chiplet-mlir/`** (new, out-of-tree), independent of mlir-air's tree, but pinned to mlir-air's LLVM commit so downstream lowering (Sub-project D) can consume `libairgpu` and AIR's ROCm path without rebasing.
2. **`ChipletDialect` shared library + `chiplet-opt` tool** — parses, prints, round-trips. No `chiplet-translate`.
3. **`test/examples/gate_up_silu.mlir`** — hand-written IR that round-trips and expresses Fleet's gate_up+SiLU task graph: one `chiplet.launch`, eight Chiplet-tasks parameterized symbolically on `partition_id`, a fused SiLU wavefront body, a cross-XCD fence, and event-signal edges to the neighbouring tasks.
4. **`docs/fleet_requirements.md`** — transcription of Fleet's Figure 4a task graph, Figure 5 sync protocol, and §4.1 cache-modifier policy into IR-level requirements (which op/attribute represents which Fleet construct, and which Fleet constructs this MVP deliberately omits). Drives the ODS schema.
5. **`docs/design_decisions.md`** — one subsection per open question in `raw/idea.md` §4, each covering (options considered, tradeoffs, chosen default, what it would take to change). Captures the decision table in §3 below.

## 3. Design decisions

All decisions below are the defaults for this MVP. Each is cheap to revise in a later sub-project — the guiding principle is *express the minimum that makes Fleet's gate_up+SiLU representable, and no more*.

| Ref | Question | Decision | Rationale (one line) |
|---|---|---|---|
| Q2 | Repo location | Standalone out-of-tree `chiplet-mlir`, depends on mlir-air install | Keeps dialect independent for fast iteration; preserves access to AIR's GPU/ROCm backend for Sub-project D |
| Q3 | Done-definition | Minimal round-trip only | MVP scope discipline; verifiers + passes are later specs |
| §4.2 | Scope granularity | Four conflated levels: `wavefront / cu / chiplet / device` | Matches Fleet Table 3 exactly; MALL / cluster / Cerebras deferred to portability work |
| §4.1 | Nominal vs symbolic binding | Symbolic; `chiplet.partition_id` as SSA (AIR-style) | Matches `air.launch` idiom; keeps IR compact; downstream passes emit one op not eight |
| §4.3 | Capacity in type vs pass | Module-level `chiplet.target` attribute | Static-verification benefit requires verifiers this MVP is not building; keeps type strings readable |
| §4.4 | Sync-op shape | Unified `chiplet.fence {scope = …}` + `chiplet.event.signal` / `chiplet.event.wait` | Fewer ops; scope attribute is sufficient to distinguish Fleet's four sync patterns at lowering time |
| §4.5 | Task body form | SPMD; `chiplet.worker_id` SSA inside each task | Matches GPU idiom and Fleet's per-worker execution model; explicit broadcast form is a later lowering concern |
| §4.6 | Mirage extension stance | Extend-first documented; no code in this sub-project | Runtime decision deferred until Sub-project D |

## 4. Repository structure

```
chiplet-mlir/
├── CMakeLists.txt                     # Standalone: finds installed MLIR by convention.
├── README.md                          # LLVM pin hash (must match mlir-air), build, test.
├── build.sh                           # Linux build.
├── include/chiplet/Dialect/
│   ├── ChipletDialect.td
│   ├── ChipletTypes.td                # !chiplet.tensor, !chiplet.event
│   ├── ChipletAttrs.td                # ChipletScopeAttr, CacheModifierAttr, ChipletTargetAttr
│   └── ChipletOps.td                  # ops in §5
├── lib/Dialect/Chiplet/               # generated + hand-written op implementations
├── tools/chiplet-opt/                 # registers Chiplet + required upstream dialects; no passes
├── test/
│   ├── lit.cfg.py                     # minimal lit harness
│   ├── Dialect/Chiplet/               # one round-trip .mlir per op
│   └── examples/gate_up_silu.mlir     # deliverable 3
└── docs/
    ├── fleet_requirements.md          # deliverable 4
    └── design_decisions.md            # deliverable 5
```

**LLVM pin.** The README records the exact LLVM commit hash used by mlir-air's `sam-chiplet-dialect` branch at the time this spec is implemented. The implementation plan will capture the specific hash; this spec does not hardcode it because it will drift.

**Build.** `chiplet-mlir` is a standard MLIR out-of-tree layout: the user provides an installed MLIR via `-DMLIR_DIR=…`; CMake finds it, pulls `AddMLIR`, and builds the dialect library + `chiplet-opt` + the lit test suite. No dependency on mlir-air at build time for the MVP (the mlir-air integration surface is a Sub-project D concern).

## 5. ODS schema

### 5.1 Attributes

- **`ChipletScopeAttr`** — enum with cases `wavefront`, `cu`, `chiplet`, `device`. Used on both `chiplet.task {level = …}` and `!chiplet.tensor<…, scope = …>`.
- **`CacheModifierAttr`** — enum with cases `cache_all`, `streaming`, `non_temporal`. Covers the three-tier policy in Fleet §4.1 (default; weights; activations/cross-XCD polling). Lowered to CDNA3/4 `sc1`/`sc0`/`nt` bits in a later sub-project.
- **`ChipletTargetAttr`** — structured attribute with fields `num_chiplets : i64`, `workers_per_chiplet : i64`, `l2_capacity_bytes : i64`. Attached to the top-level module. Read by any future pass; never queried by MVP verifiers.

### 5.2 Types

- **`!chiplet.tensor<shape, elem, scope>`** — shape and element-type as in builtin `tensor`; `scope` is a `ChipletScopeAttr`. Interior representation uses a strided layout consistent with `memref`. Chosen over `memref` because the dialect is meant to read close to high-level tensor programs; converting to `memref` is a lowering concern.
- **`!chiplet.event`** — opaque handle for cross-task dependencies. Corresponds to a Fleet event counter at lowering time.

### 5.3 Ops

Eight ops in total.

- **`chiplet.launch`** — region-bearing; takes `num_chiplets` as an attribute and the tensors the task graph reads/writes as explicit operands; yields result tensors via a terminator. Introduces the `chiplet.partition_id` symbolic index into its body's scope. By convention the single entry point for a chiplet task graph (not enforced in the MVP).
- **`chiplet.task`** — region-bearing; `level : ChipletScopeAttr` attribute. Body is SPMD across workers at that level. May nest inside another `chiplet.task` whose level is strictly coarser (enforced only as a lint in the MVP, not a hard verifier — see §6).
- **`chiplet.partition_id : () -> index`** — SSA value in `[0, num_chiplets)`. Valid only lexically inside a `chiplet.launch` region.
- **`chiplet.worker_id : () -> index`** — SSA value in `[0, workers_at_level)`. Valid only lexically inside a `chiplet.task` region.
- **`chiplet.copy (%src) { src_scope, dst_scope, modifier }`** — data movement between scopes. Result is a tensor at `dst_scope`. The strided-slice offsets/sizes are operand-captured from the enclosing region (typically derived from `%pid`/`%wid`).
- **`chiplet.fence { scope : ChipletScopeAttr }`** — synchronization at the named scope. The scope attribute disambiguates Fleet's four sync patterns at lowering.
- **`chiplet.event.signal %e : !chiplet.event`** — producer side.
- **`chiplet.event.wait %e : !chiplet.event`** — consumer side.

### 5.4 Trivial verifiers (the only verifiers in this MVP)

- Scope/level attribute values must be one of the four enum cases.
- `chiplet.partition_id` must appear lexically inside a `chiplet.launch` body; `chiplet.worker_id` must appear lexically inside a `chiplet.task` body.
- `chiplet.event.signal` and `chiplet.event.wait` must reference a value of type `!chiplet.event`.

Explicitly **not** enforced here: nesting rules (chiplet-task cannot contain a device-task), scope compatibility on `chiplet.copy` (source/destination must be reachable), capacity fit, event lifetime. These belong in a dedicated "verifier-hardening" sub-project that can be done cheaply once the partition-to-chiplet pass gives us realistic test inputs.

## 6. Example IR (driver artifact)

`test/examples/gate_up_silu.mlir` is the concrete round-trip target and the acceptance test for this sub-project. A sketch:

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

        // MFMA inner loop — opaque region body in MVP (no codegen here).

        chiplet.task { level = #chiplet.scope<wavefront> } {
          // Fused SiLU element-wise body.
        }

        chiplet.fence { scope = #chiplet.scope<chiplet> }
      }

      chiplet.event.signal %ev_out : !chiplet.event
    }

    return %out, %ev_out
  }
}
```

Deliberate imprecisions in this sketch that the ODS implementation will pin down:

- Exact slice syntax on `chiplet.copy` (likely strided offsets/sizes as operands, following the `memref.subview` pattern).
- Operand plumbing into and out of `chiplet.launch` (explicit operands with capture-like binding into the region's entry block).
- The MFMA inner body — represented as an opaque block or as embedded linalg; the ODS must accept either. We pick embedded linalg for the example, since linalg dialect is already available upstream.

The point of this file is that *writing it by hand and round-tripping it is the deliverable*. If the ODS cannot express it, the ODS is wrong and this spec has failed.

## 7. Documentation artifacts

### 7.1 `docs/fleet_requirements.md`

Section-by-section transcription of Fleet's constructs into IR requirements:

- **Figure 4a (one-transformer-layer task graph, bs=1)**: for each operator (RMSNorm, QKV proj, Attention, O-Proj+Res, RMSNorm, Gate+Up+SiLU, Down+Res), record the task level, worker count, memory scope, and the ops/attrs used to represent it. Flag which operators are in-MVP-scope (gate_up+SiLU plus its immediate predecessors/successors) and which are deferred (attention body, final down-proj reduction).
- **Figure 5 (hierarchical sync protocol)**: for each of the four patterns (task-queue read-only; scheduler→worker L2-local; worker→worker L2-local; XCD→global `buffer_wbl2`), record how it lowers from the chiplet dialect — specifically the `chiplet.fence` scope attribute that selects it, and the event shape.
- **§4.1 cache-modifier policy**: map the three policy tiers (weight streaming; activation non-temporal; cross-XCD polling) onto `CacheModifierAttr` values.
- **Deliberately out of scope**: K-split / reductions, MALL residency, non-GEMM operator bodies (attention, layer-norm internals).

Note: this document may cite the paper by section/figure number only. It must not quote, paraphrase at length, or reproduce unpublished content. Fleet is not yet published.

### 7.2 `docs/design_decisions.md`

One subsection per row of the decision table in §3 (Q2, Q3, §4.1–4.6). Each subsection has:

1. The question as stated in `raw/idea.md`.
2. Options considered (minimum two; typically three).
3. Tradeoffs for each.
4. Chosen default (matching §3 of this spec).
5. What it would take to revise the decision — specifically, which sub-project would reopen it, and what tests would catch the regression.

Keep each subsection to roughly one page.

## 8. Testing

- **`test/Dialect/Chiplet/`** — one `.mlir` per op; each runs `chiplet-opt %s | FileCheck %s` to verify parse + print + round-trip identity.
- **`test/examples/gate_up_silu.mlir`** — round-trips end-to-end.
- **No C++ unit tests.** lit is sufficient for the MVP; C++ tests appear when we have verifiers and passes worth exercising.

## 9. Success criteria (acceptance)

A reviewer can check all four in under fifteen minutes:

1. `chiplet-opt test/examples/gate_up_silu.mlir` produces output that parses back identically (round-trip).
2. `docs/fleet_requirements.md` enumerates every op and attribute in the §5 schema and identifies the Fleet construct it represents, plus an explicit list of Fleet constructs this MVP does not represent.
3. `docs/design_decisions.md` covers Q2, Q3, and §4.1–4.6 with the structure in §7.2.
4. `build.sh` succeeds on Linux against an installed MLIR at mlir-air's current LLVM pin, and lit tests pass.

## 10. Out of scope (explicit)

To be revised by a later sub-project with its own spec:

- `linalg.matmul` → `chiplet.launch` transformation (Sub-project B).
- Cost model and schedule search (Sub-project C).
- Lowering to ROCDL/AMDGPU via Mirage MPK (Sub-project D).
- Capacity-fit verification and full nesting-rule enforcement (post-MVP verifier hardening).
- `chiplet-translate`.
- Attention / MoE operator representations.
- Portability to NVIDIA Blackwell, Cerebras WSE.

## 11. Open items (spec implementer resolves during Task 4 ODS work)

These are small enough to settle during implementation and do not need a spec revision:

- Exact printed syntax for `ChipletScopeAttr` and `CacheModifierAttr` (the examples above use `#chiplet.scope<...>`; this may become `#chiplet<scope chiplet>` or a keyword form — pick whichever round-trips most readably).
- Whether `chiplet.task` returns values (tensor yields from a region) or mutates explicit result operands. The sketch uses region yields.
- Lit test scaffolding choice: use mlir-air's lit config as a template, or upstream's minimal `standalone-opt` example.

---

*End of spec.*
