# CSL Dialect Design Evaluation & Roadmap

Based on the recent layout ops stress testing and a review of the Cerebras CSL Language Guide, several design flaws (missing verifiers) and missing language features have been identified. 

## 1. Identified Flaws in Current Layout Ops

When pushing the MLIR ops to edge cases (see `mlir/test/Dialect/CSL/layout_stress.mlir`), the parser/verifier currently accepts invalid spatial configurations. We need to add MLIR C++ verifiers for the following:

*   **Out-of-Bounds Painting**: `csl.paint pe(x, y)` inside a `csl.code_region shape(W, H)` allows `x >= W` or `y >= H`. MLIR should reject this at compile time.
*   **Placement Overlaps**: `csl.place` allows multiple regions to be placed on overlapping absolute WSE coordinates. We need a global layout verifier to check bounding box collisions.
*   **Dataflow Cycle/Type Checking**: `csl.dataflow %src -> %dst` should verify that `%src` is an "output" port, `%dst` is an "input" port, and potentially that their sizes match.

## 2. Missing CSL Language Features

To fully generate the CSL `language_index` and address your comments regarding "TILE SELECTION" and "SCHEDULE/ALGORITHM", our MLIR Dialect needs the following new features:

### A. Advanced Types & Arrays (Bundles)
*   **Arrays of Colors/Routes**: Currently, we pass variadic single colors. CSL heavily relies on arrays of colors for wide channels (e.g., `var colors [3]color`). We need an array/bundle abstraction.
*   **Structs/Enums**: `!csl.struct` and `!csl.enum` types are needed for passing complex configuration payloads to kernels.

### B. Tile Specialization (Compute vs Memory)
*   You mentioned: *"Say make this tile a memory tile only, Say make this a compute only"*. 
*   **Solution**: We can add a `kind` attribute to `csl.code_region` (e.g., `kind = "compute"`, `kind = "memory"`) or introduce specific container ops like `csl.mem_region` vs `csl.compute_region` to enforce what inner operations are valid (e.g., `csl.task` only allowed in compute).

### C. State Machines & Micro-Thread Control Flow
*   You mentioned: *"Wait until task1 is finished, Now perform task2... This can be a state machine maybe? in a loop"*
*   **Solution**: CSL has native `struct fsm` constructs and block/unblock semantics. We need:
    *   `csl.state_machine` / `csl.state` ops.
    *   `csl.block` / `csl.unblock` for task synchronization.
    *   `csl.loop` (or integration with standard `scf.for`) that correctly translates to CSL hardware loop constructs.

## 3. Recommended Next Steps

1.  **Phase 1**: Add C++ verifiers (`OpTrait` or custom `verify()` methods) to `CSLLayoutOps.cpp` to fix the layout flaws (OOB painting, collisions).
2.  **Phase 2**: Introduce Tile Specialization attributes to `csl.code_region`.
3.  **Phase 3**: Introduce State Machine and Control Flow ops to `CSLKernelOps.td`.
