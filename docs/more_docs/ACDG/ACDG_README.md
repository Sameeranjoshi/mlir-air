# ACDG & AIRDependency Pass Learning Materials

This directory contains comprehensive documentation on the MLIR-AIR's Asynchronous Compute Data-flow Graph (ACDG) and the AIRDependency pass that creates it.

## Files in This Package

### 0. **VISUALIZE_DEPENDENCIES.md** (Hands-On Learning)
**Best for:** Running the pass yourself and understanding output

Contains:
- Quick start commands to run the pass
- How to read transformed MLIR code
- Real example before/after with diagrams
- How to manually analyze dependencies
- Python scripts and tools to visualize output
- Debugging tips and validation strategies
- Test cases to try at different difficulty levels

**Read this second** after understanding concepts - it's practical and hands-on!

---

### 1. **ACDG_DEPENDENCY_GUIDE.md** (Start Here!)
**Best for:** Understanding concepts and overall architecture

Contains:
- Overview of what ACDG is and why it matters
- Core concepts (async tokens, dependency edges)
- Three-phase transformation explained
- Operation-by-operation transformations
- Dependency analysis strategy
- Handling loops and branches
- Real-world patterns
- Common pitfalls and insights
- Testing & verification

**Read this first** if you're new to the topic.

---

### 2. **ACDG_VISUAL_EXAMPLES.md** (Learn by Example)
**Best for:** Seeing concrete before/after transformations

Contains:
- 5 detailed real-world examples from the test suite:
  1. Simple matmul (producer-consumer)
  2. Affine if (branching with tokens)
  3. Loop with dependencies (iteration threading)
  4. Parallel herds (independent execution)
  5. Deallocation (synchronization)
- Visual ASCII diagrams for each
- Dependency graphs and execution timelines
- Pattern summary (sequential, join, branch, loops, conditionals)
- Practical debugging tips

**Use this** when you want to see concrete transformations with diagrams.

---

### 3. **ACDG_IMPLEMENTATION_REFERENCE.md** (Deep Technical Dive)
**Best for:** Understanding the C++ code implementation

Contains:
- High-level flow of the pass
- Key data structures (ExecuteNode, ExecuteGraph)
- Phase 1: Creating async operations (with code examples)
- Phase 2: Dependency analysis (algorithm + patterns)
- Phase 3: Applying dependencies (code patterns)
- Utility functions
- Debugging functions
- Common code patterns
- Test case mapping

**Use this** when reading the actual C++ implementation in:
- `mlir/lib/Transform/AIRDependency.cpp`
- `mlir/include/air/Transform/AIRDependency.h`

---

## Quick Navigation Guide

### I want to understand...

**"What is ACDG?"**
→ Read: ACDG_DEPENDENCY_GUIDE.md Part 1 (Core Concepts)

**"How does the transformation work?"**
→ Read: ACDG_DEPENDENCY_GUIDE.md Part 2-3 + ACDG_VISUAL_EXAMPLES.md Example 1

**"How are branches handled?"**
→ Read: ACDG_VISUAL_EXAMPLES.md Example 2 (affine_if)

**"How are loops handled?"**
→ Read: ACDG_VISUAL_EXAMPLES.md Example 3 (scf_for)

**"What's a token and why does it matter?"**
→ Read: ACDG_DEPENDENCY_GUIDE.md Part 1.1-1.2 + ACDG_VISUAL_EXAMPLES.md Pattern Summary

**"How does the C++ code work?"**
→ Read: ACDG_IMPLEMENTATION_REFERENCE.md

**"How do I debug a transformation?"**
→ Read: ACDG_VISUAL_EXAMPLES.md Debugging Tips + ACDG_IMPLEMENTATION_REFERENCE.md Debugging Functions

**"What are common mistakes?"**
→ Read: ACDG_DEPENDENCY_GUIDE.md Part 8 (Pitfalls)

---

## Learning Path

### Beginner (New to the Project)
1. Read ACDG_DEPENDENCY_GUIDE.md Part 1 (Core Concepts) - **5 min**
2. Read VISUALIZE_DEPENDENCIES.md "Understanding the Output" section - **10 min**
3. Run a simple test case:
   ```bash
   cd /scratch/general/vast/u1418973/mlir-air
   ./install/bin/air-opt mlir/test/Transform/AIRDependency/dma_memcpy_nd.mlir -air-dependency
   ```
4. Look at ACDG_VISUAL_EXAMPLES.md Example 1 (Simple matmul) - **10 min**
5. Try the affine_if example:
   ```bash
   ./install/bin/air-opt mlir/test/Transform/AIRDependency/affine_if.mlir -air-dependency
   ```

**Total: ~35 minutes** - You now understand the basic transformation!

### Intermediate (Understand Patterns)
1. Read ACDG_DEPENDENCY_GUIDE.md Parts 2-7 - **30 min**
2. Work through ACDG_VISUAL_EXAMPLES.md Examples 2-5 - **30 min**
3. For each example, use VISUALIZE_DEPENDENCIES.md to:
   - Understand what to look for in output
   - Extract and analyze dependencies
   - Try the debugging tips
4. Run all test cases and compare outputs - **20 min**

**Total: ~80 minutes** - You understand patterns and can analyze code!

### Advanced (Read the Code)
1. Read ACDG_IMPLEMENTATION_REFERENCE.md - **30 min**
2. Open `mlir/lib/Transform/AIRDependency.cpp` - **30 min**
3. Map each section of the code to the reference guide
4. Use VISUALIZE_DEPENDENCIES.md Python scripts to analyze real transformations - **20 min**
5. Modify the pass to add debug output and re-run - **20 min**

**Total: ~120 minutes** - You can read and modify the implementation!

---

## Key Insights

### The Core Insight
Every operation needs to know:
1. **What it depends on** (input tokens)
2. **When it's done** (output token)

This turns synchronous code into an **explicit DAG** that hardware can schedule.

### The Three Phases

| Phase | Input | Output | Purpose |
|-------|-------|--------|---------|
| 1 | Sync ops | Async ops + empty deps | Create async operation structure |
| 2 | Async ops | Dependency graph | Analyze data flow relationships |
| 3 | Deps list | Ops with [tokens] | Apply explicit synchronization |

### Why It Matters

**Without ACDG:**
- Compiler can't parallelize (doesn't know what's independent)
- Hardware can't schedule (doesn't know execution order)
- Memory safety unclear (when is it safe to free?)

**With ACDG:**
- Explicit dependency information
- Can identify parallel operations
- Hardware knows exact scheduling constraints
- Memory operations can be ordered precisely

---

## Test Cases: Quick Reference

Run any test case with:
```bash
cd /scratch/general/vast/u1418973/mlir-air
./install/bin/air-opt mlir/test/Transform/AIRDependency/[FILENAME].mlir -air-dependency
```

### Beginner Tests
- `matmul_nd.mlir` - Basic producer-consumer
- `dma_memcpy_nd.mlir` - Simple DMA async

### Intermediate Tests
- `affine_if.mlir` - Branching with tokens
- `scf_for.mlir` - Loop iteration threading
- `air_channel.mlir` - Channel synchronization

### Advanced Tests
- `parallel_herds.mlir` - Parallel independent execution
- `reshape_dependency.mlir` - Complex memory operations
- `two_layer.mlir` - Multiple herds with dependencies

---

## Common Questions

### Q: What is an async token?
**A:** A value that represents "this operation is done". It's used to serialize operations that depend on each other.

See: ACDG_DEPENDENCY_GUIDE.md Part 1.1

### Q: Why do we need air.execute?
**A:** Operations like `memref.alloc` are synchronous (don't have a "done" signal). We wrap them in `air.execute` to give them a token.

See: ACDG_IMPLEMENTATION_REFERENCE.md Phase 1

### Q: How are loops transformed?
**A:** Loops get an additional `iter_args` that carries the token from one iteration to the next.

See: ACDG_VISUAL_EXAMPLES.md Example 3

### Q: When should I use air.wait_all?
**A:** When you need to synchronize (ensure an op is done) before proceeding.

See: ACDG_DEPENDENCY_GUIDE.md Part 3.5

### Q: Why are channel puts/gets ordered?
**A:** Because they communicate through a shared channel; operations must be sequential.

See: ACDG_IMPLEMENTATION_REFERENCE.md Phase 2 (ChannelOp)

---

## Cheat Sheet

### Async Token Quick Ref
```mlir
// Creating a token
%token, %result = air.execute -> (memref<...>) { ... }

// Using a token
%next_token = air.dma_memcpy_nd async [%token] (...)

// Waiting on a token
%done_token = air.wait_all async [%token]

// In a loop
scf.for %i = ... iter_args(%arg_token = %init) -> (!air.async.token) {
  %t = air.execute [%arg_token] { ... }
  scf.yield %t : !air.async.token
}

// In a branch
%if_token = scf.if %cond -> (!air.async.token) {
  %t1 = air.execute { ... }
  affine.yield %t1 : !air.async.token
} else {
  %t2 = air.execute { ... }
  affine.yield %t2 : !air.async.token
}
```

### Dependency Patterns
```
Sequential:      A → B → C
Join:            A ──┬─→ C
                 B ──┘
Fork:            A ──┬─→ B
                    └─→ C
Loop:            init → for[iter_args] → yield
Branch:          init → if { then: yield } { else: yield }
```

---

## Further Resources

### Official Documentation
- `/scratch/general/vast/u1418973/mlir-air/docs/AIRAsyncConcurrency.md`
- MLIR documentation: https://mlir.llvm.org/

### Key Files to Explore
- **Main pass:** `mlir/lib/Transform/AIRDependency.cpp`
- **Header:** `mlir/include/air/Transform/AIRDependency.h`
- **Utilities:** `mlir/lib/Util/Dependency.cpp`
- **Canonicalize:** `mlir/lib/Transform/AIRDependencyCanonicalize.cpp`
- **Tests:** `mlir/test/Transform/AIRDependency/`

### Related Passes
1. **AIRDependencyCanonicalize** - Removes redundant edges
2. **AIRDependencyScheduleOpt** - Optimizes operation ordering
3. **AIRDmaToChannel** - Converts DMA to channels
4. **AIRToAIE** - Converts to hardware dialect

---

## Document Statistics

| Document | Pages | Focus | Level |
|----------|-------|-------|-------|
| VISUALIZE_DEPENDENCIES.md | ~9 | Hands-on Output Analysis | Beginner+ |
| ACDG_DEPENDENCY_GUIDE.md | ~10 | Concepts & Theory | Beginner |
| ACDG_VISUAL_EXAMPLES.md | ~12 | Examples & Patterns | Intermediate |
| ACDG_IMPLEMENTATION_REFERENCE.md | ~10 | C++ Code | Advanced |

**Total:** ~41 pages of documentation with:
- 30+ diagrams and visual flows
- 20+ code examples
- 5 complete test case walkthroughs
- Python scripts for analysis
- Command references and debugging tips

---

## How to Use These Docs

### Option 1: Linear Reading
Read in order: Guide → Examples → Reference
Best if you have time and want comprehensive understanding.

### Option 2: Topic-Based
Find what you're interested in and jump to that section.
Best if you're debugging or implementing a specific feature.

### Option 3: Active Learning
1. Read a concept in Guide
2. See an example in Visual Examples
3. Look at the code in Reference
4. Run the test case yourself
5. Trace through the output

This interleaving accelerates learning.

---

## Feedback & Updates

These documents are living resources. As you learn and discover:
- Share insights with your team
- Update diagrams if you find better ones
- Add examples if you find interesting patterns
- Report errors or unclear sections

The goal is to make ACDG/AIRDependency understandable to everyone!

---

**Happy learning!** 🚀

For questions, refer to the documents or read the source code with this guide as your reference.

