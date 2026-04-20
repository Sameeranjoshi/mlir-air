# chiplet-mlir

Out-of-tree MLIR dialect for compiler-driven chiplet-aware GPU compilation.

**Status:** Phase 1 (dialect foundation) — round-trip only, no transformation passes, no lowering. See `docs/superpowers/specs/2026-04-20-chiplet-dialect-foundation-design.md` in the parent mlir-air repo for the spec.

**Scope in Phase 1:** 8 ops, 2 attribute classes (+ 2 enums), 1 opaque type, 2 trivial verifiers, a hand-written example expressing Fleet's gate_up+SiLU task graph, and a lit test suite.

## Dependencies

- An installed MLIR matching the LLVM commit hash used by the parent mlir-air tree. This repository currently pins to the same commit mlir-air's `sam-chiplet-dialect` branch builds against:

  ```
  ebf5d9ef7de29b55fd9e9d504f83689b4013e0de  (2025-12-05)
  ```

  Recorded in `../utils/clone-llvm.sh`. If you rebuild LLVM, update this hash.

## Build

```bash
./build.sh
```

The script invokes `cmake` pointing at `../llvm/install` (mlir-air's sibling LLVM install), then `ninja`. On a clean tree this takes a few minutes.

## Run tests

```bash
cmake --build build --target check-chiplet
```

This runs the lit test suite under `test/`. Phase 1 acceptance: `test/examples/gate_up_silu.mlir` round-trips through `chiplet-opt` identically.

## Layout

```
chiplet-mlir/
├── CMakeLists.txt
├── include/chiplet/Dialect/   # ODS (.td) + C++ headers
├── lib/Dialect/Chiplet/       # dialect + op C++ implementations
├── tools/chiplet-opt/         # opt tool (dialect registration only)
├── test/
│   ├── Dialect/Chiplet/       # per-op round-trip tests
│   └── examples/              # gate_up_silu.mlir acceptance test
└── docs/
    ├── fleet_requirements.md  # Fleet constructs → IR requirements mapping
    └── design_decisions.md    # §4 open questions resolved
```

## Not here (yet)

- `linalg.matmul → chiplet.launch` partition pass (Phase 2, separate spec).
- Cost model and schedule search (Phase 3, same spec as Phase 2).
- ROCDL lowering, Mirage MPK integration, hardware execution (Sub-project D, not yet spec'd).
- Full nesting-rule and scope-compatibility verifiers (verifier-hardening spec).
