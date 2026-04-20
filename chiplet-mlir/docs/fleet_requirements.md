# Fleet → Chiplet Dialect Requirements

**Status:** Phase 1 requirements driver document.
**Audience:** implementers of the `chiplet` dialect.
**Source:** Fleet paper (ASPLOS 2027 submission, `../raw/asplos2027_fleet.pdf`). Unpublished — do not reproduce content in committed files; refer by section/figure/table number only.

This document maps each Fleet construct that Phase 1 must represent onto a
concrete op, attribute, or type in the `chiplet` dialect, and flags Fleet
constructs that Phase 1 deliberately does not represent.

## 1. Task hierarchy (Fleet Table 3)

Fleet defines four task levels with hardware scopes:

| Fleet level | Hardware scope | Typical memory | Typical op | Represented in chiplet dialect as |
|---|---|---|---|---|
| wavefront-task | 1 wavefront | regs, LDS | SiLU, residual add | `chiplet.task level = <wavefront>` |
| cu-task | 1 workgroup | LDS, L2 | attention, RMSNorm | `chiplet.task level = <cu>` |
| Chiplet-task | 1 XCD | L2, HBM | GEMM partition | `chiplet.task level = <chiplet>` |
| device-task | 8 XCDs | global HBM | full GEMM/attn | `chiplet.task level = <device>` |

Level is expressed through the single `ChipletScopeAttr` enum, attached to
`chiplet.task` via the `level` operand and carried on `!tensor` encodings via
`#chiplet.scope<…>`. Conflation of task level and memory scope matches
Fleet's Table 3; see `design_decisions.md` §4.2.

## 2. Task graph (Fleet Figure 4a)

Fleet's per-transformer-layer task graph at bs=1 decomposes one layer into 8
Chiplet-tasks per GEMM plus 1 CU-task per RMSNorm plus 8×8 CU-tasks for
attention. Phase 1 must be able to express the **gate_up+SiLU Chiplet-task
group** (one `chiplet.task level = <chiplet>` fused with a nested
`chiplet.task level = <wavefront>` for SiLU).

| Fleet construct | chiplet dialect representation |
|---|---|
| Per-XCD task dispatch | `chiplet.launch num_chiplets = 8 { … }` |
| Logical partition identity | `%pid = chiplet.partition_id : index` |
| Per-worker rank inside a task | `%wid = chiplet.worker_id : index` |
| Cross-task dependency edge | `!chiplet.event` + `chiplet.event.signal` / `chiplet.event.wait` |
| Operator fusion (SiLU into gate+up) | nested `chiplet.task level = <wavefront>` inside a `chiplet.task level = <chiplet>` |

Phase 1 explicitly does **not** represent:

- The inner MFMA/GEMM body (opaque; Phase 2 partition pass + Sub-project D
  lowering will fill this in).
- K-split / cross-XCD reductions (Fleet §4.1 alternative strategy for bs≥32).
- Per-XCD round-robin scheduler-to-worker dispatch (implementation detail of
  Fleet's runtime, not of the IR).
- Attention Chiplet-/CU-task graphs.

## 3. Synchronization protocol (Fleet Figure 5, §5.2)

Fleet uses four distinct sync patterns scoped to the narrowest level
sufficient for correctness. Phase 1 represents all four through a single
unified op + scope attribute:

| Fleet pattern | Mechanism (CDNA3/4) | chiplet dialect representation |
|---|---|---|
| Task queue (read-only) | no sync | implicit (no op) |
| Scheduler → worker (L2-local) | device-scope atomic, local L2 | `chiplet.fence scope = <cu>` (lowers to intra-XCD atomic) |
| Worker → worker (L2-local) | device-scope atomic, local L2 | `chiplet.fence scope = <chiplet>` |
| XCD → global | `buffer_wbl2` + `threadfence` + `flat_atomic_add sc0 sc1` | `chiplet.fence scope = <device>` + `chiplet.event.signal` |

Phase 1 emits only the IR-level ops. The mapping to CDNA3/4 instructions is
Sub-project D's concern; this table is the authoritative lowering target.

## 4. Cache-modifier policy (Fleet §4.1)

Fleet's three-tier policy uses CDNA3/4 `SC1` / `SC0` / `NT` scope and
non-temporal bits on each memory instruction. Phase 1 encodes the three
policy tiers as `CacheModifierAttr`:

| Fleet tier | Instruction bits | chiplet dialect representation |
|---|---|---|
| Default (wave-scope, LRU) | SC1=0, SC0=0, NT=0 | `#chiplet.cache<cache_all>` |
| Weight streaming | SC1=1, NT=1 | `#chiplet.cache<streaming>` |
| Activation / cross-XCD polling | NT=1 | `#chiplet.cache<non_temporal>` |

Applied as a per-`chiplet.copy` attribute. Lowering to CDNA3/4 bits is
Sub-project D.

## 5. Hardware parameters

Fleet §8 parameterizes the abstraction on X (chiplets), W (workers per
chiplet), and C (L2 capacity per chiplet). Phase 1 carries all three on a
module-level `chiplet.target` attribute:

```mlir
chiplet.target = #chiplet.target<num_chiplets = 8,
                                 workers_per_chiplet = 31,
                                 l2_capacity_bytes = 4194304>
```

Values here correspond to MI350 (8 XCDs × 32 CUs minus 1 scheduler per XCD ×
4 MB private L2). For MI300X the only change is workers_per_chiplet = 37
(38 CUs minus 1). No verifier queries this attribute in Phase 1; it is
future-use metadata for Phase 2 onward.

## 6. Table 4 numerics

Fleet Table 4 reports measured L2 hit rate and HBM traffic per batch size
across three scheduling variants (Mirage, M-tile, M-split). Phase 1 does not
need these numbers — they are the calibration target for Phase 3's cost
model. Kept out of Phase 1 artifacts so the paper's numerics stay off-disk
until Phase 3 needs them.

## 7. Summary of coverage

- **In-Phase-1 scope**: task hierarchy enum, Chiplet-task dispatch
  (`chiplet.launch` + `chiplet.partition_id`), nested task levels, event
  dependencies, unified fence op, cache modifier policy, module-level target
  attribute, gate_up+SiLU round-trip test.
- **Out-of-Phase-1 scope** (requirements captured in spec §11, deferred to
  later specs): K-split reductions, attention/MoE representations, MALL /
  cluster scopes, static capacity verification, Mirage MPK runtime
  integration, lowering to ROCDL/AMDGPU.
