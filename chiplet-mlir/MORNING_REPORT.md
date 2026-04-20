# Morning Report — Phase 1 Chiplet Dialect

**Date:** 2026-04-20 → 2026-04-21 (overnight).
**Status:** ✅ **Phase 1 complete. 8/8 lit tests green. Ready for your review.**
**Branch:** `sam-chiplet-dialect` (unchanged). No new branch created; all changes committed inline.

## TL;DR

The chiplet dialect foundation is built, round-trips Fleet's gate_up+SiLU
task graph, and has full lit test coverage. Located at `chiplet-mlir/`.
Built against the MLIR install in `llvm/install/` (commit
`ebf5d9ef7de29b55fd9e9d504f83689b4013e0de`).

```bash
cd chiplet-mlir/build && ninja check-chiplet
# -- Testing: 8 tests, 8 workers --
# Testing Time: 0.05s
#   Passed: 8
```

Run `chiplet-opt test/examples/gate_up_silu.mlir` to see the Fleet-shaped IR
round-trip.

## What's green

Acceptance criteria from the spec (§9 of
`docs/superpowers/specs/2026-04-20-chiplet-dialect-foundation-design.md`):

1. ✅ `chiplet-opt test/examples/gate_up_silu.mlir` round-trips.
2. ✅ `fleet_requirements.md` enumerates every op/attr and maps it to a Fleet
   construct, plus a list of deliberately omitted constructs.
3. ✅ `design_decisions.md` covers Q2, Q3, §2 (AIR relationship), and
   `raw/idea.md` §4.1–4.6.
4. ✅ Build passes on Linux; lit tests 8/8 green.

## What changed from the spec

**Tensor type: `tensor<…, #chiplet.scope<…>>` instead of `!chiplet.tensor<…>`.**
I switched from a custom tensor type to using MLIR's builtin ranked tensor
with a dialect-encoding attribute. The spec §6.2 had called for a custom
`!chiplet.tensor<shape, elem, scope>`. Reason: the builtin path is
idiomatic, the parser round-trips correctly, and a custom type would
duplicate shape/element parsing for no Phase 1 benefit.

Impact: test files and `fleet_requirements.md` use the encoding form. If
you want the custom type, we can re-introduce it in a later spec; it's a
purely mechanical change.

**Canonical printed form for enum attrs: `level = <chiplet>` not `level = #chiplet.scope<chiplet>`.**
When `ChipletScopeAttr` is a direct op argument (e.g., `chiplet.task`'s
`level`), MLIR prints the short form. When it's an element of an attr-dict
(e.g., `{chiplet.target = #chiplet.target<…>}` on the module) or a tensor
encoding (`tensor<…, #chiplet.scope<…>>`), the full form is printed. Both
are accepted by the parser. Tests use the printed form.

## File manifest

```
chiplet-mlir/
├── CMakeLists.txt
├── README.md                        # LLVM pin hash + build/test commands
├── MORNING_REPORT.md                # this file
├── build.sh                         # convenience wrapper around cmake+ninja
├── .gitignore                       # build/
├── include/chiplet/Dialect/
│   ├── ChipletDialect.td            # dialect registration
│   ├── ChipletAttrs.td              # ScopeAttr, CacheModifierAttr, TargetAttr
│   ├── ChipletTypes.td              # !chiplet.event
│   ├── ChipletOps.td                # 8 ops
│   ├── ChipletDialect.h
│   ├── ChipletAttrs.h
│   ├── ChipletTypes.h
│   ├── ChipletOps.h
│   └── CMakeLists.txt
├── lib/Dialect/Chiplet/
│   ├── ChipletDialect.cpp           # dialect initialize()
│   ├── ChipletAttrs.cpp             # attribute registration
│   ├── ChipletTypes.cpp             # type registration
│   ├── ChipletOps.cpp               # verifiers for partition_id, worker_id
│   └── CMakeLists.txt
├── tools/chiplet-opt/
│   ├── chiplet-opt.cpp              # dialect registration; no passes yet
│   └── CMakeLists.txt
├── test/
│   ├── lit.cfg.py
│   ├── lit.site.cfg.py.in
│   ├── CMakeLists.txt
│   ├── Dialect/Chiplet/
│   │   ├── target.mlir              # module-level target attribute
│   │   ├── launch.mlir              # chiplet.launch + partition_id
│   │   ├── task.mlir                # task at all 4 levels + worker_id
│   │   ├── copy.mlir                # all 3 cache modifiers
│   │   ├── fence.mlir               # all 4 scopes
│   │   ├── event.mlir               # signal/wait
│   │   └── verify_scope.mlir        # trivial-verifier negative + positive
│   └── examples/
│       └── gate_up_silu.mlir        # Phase 1 acceptance test (Fleet Figure 4a)
└── docs/
    ├── fleet_requirements.md        # deliverable 4
    └── design_decisions.md          # deliverable 5
```

## Self-critique (per spec §7 of `raw/idea.md`)

Things I think are weak:

1. **`chiplet.copy` moves whole tensors** — no offset/size operands. The
   gate_up example uses it to "move from `tensor<4096x24576>` to
   `tensor<4096x3072>`" but the ONLY reason the shapes are different is
   that the output type declares a smaller shape. There's no actual slice
   math. In Phase 2 when the partition pass generates these copies, it
   will need a strided-slice form. I deliberately deferred this (MVP), but
   the MVP example hides the fact that we haven't specified *how* the
   per-XCD slice is computed.

2. **Trivial verifiers are too trivial.** `partition_id` inside a launch
   and `worker_id` inside a task — that's it. I did not check:
   - That task-level nesting respects the hierarchy
     (`chiplet.task level = cu` cannot contain `level = chiplet`).
   - That `chiplet.copy`'s source and destination tensor encodings are
     reachable (chiplet → wavefront is invalid).
   - That an event is signalled before it is waited on in SSA order.

   The spec deferred these to a "verifier hardening" sub-project, which is
   defensible, but the dialect is currently permissive enough to write
   clearly nonsensical IR. I'd prioritize adding the nesting-rule verifier
   early in Phase 2.

3. **No `chiplet.yield`.** I used `NoTerminator` on `chiplet.launch` and
   `chiplet.task`. That is the simplest path and works because neither op
   produces results in the MVP. If we later want tensor-valued results
   (e.g. `chiplet.launch → tensor<…>`), we'll need an explicit yield.

4. **No C API.** The standalone example ships a `standalone-capi-test`; I
   intentionally didn't. If you want Python bindings in Phase 3 (cost
   model), the C API is the starting point and will need to be retrofitted.

5. **No `chiplet-translate` tool.** Spec explicitly carved this out, but
   worth stating.

## What I would resolve with you before Phase 2

- **Copy-slice syntax.** How should `chiplet.copy` take offsets/sizes?
  Three options worth considering: (a) `memref.subview`-style operands, (b)
  an affine map on the op, (c) defer entirely and let the enclosing loop
  (or the partition pass) carry the slice math. Phase 2's partition pass
  needs to emit concrete slices, so this has to be resolved then.

- **How strict does Phase 2's expected-output test need to match**
  `gate_up_silu.mlir`? Per the revised spec §10 criterion 4, the pass
  emits the GEMM Chiplet-task only, no SiLU. But we now have a working
  round-trip for the *full* `gate_up_silu.mlir` (GEMM + SiLU). If we want
  the partition pass's golden output to be a subset of that file, the
  subset should be explicit.

- **Scope of the `--chiplet-partition` option surface.** Pass options
  (partition-axis, traversal-order, weight-cache-policy) as CLI flags, or
  as module attributes driven by the cost model? Phase 3 will answer this
  definitively, but Phase 2 stage 1 needs at least a proof-of-life flag.

## Next session

Open the spec — `docs/superpowers/specs/2026-04-20-chiplet-dialect-foundation-design.md` —
and decide if you want Phase 2 + Phase 3 to proceed as-specified or with
changes. Then invoke `superpowers:writing-plans` to produce the Phase 2+3
implementation plan.

If you want to verify Phase 1 yourself first:

```bash
cd chiplet-mlir
./build.sh                                # configure + build
cd build && ninja check-chiplet           # run tests
./bin/chiplet-opt ../test/examples/gate_up_silu.mlir  # see the IR round-trip
```

---

Have a good morning.
