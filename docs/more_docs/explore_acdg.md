# MLIR-AIR ACDG Workflow: End-to-End Guide

## What is ACDG?

**ACDG** = Asynchronous Compute Data-flow Graph.
It represents asynchronous concurrency explicitly, showing how operations depend on each other via tokens. This is the core abstraction that makes AIR unique.

---

## The 3-Stage ACDG Pipeline

### Stage 1: Extract ACDG from synchronous code (`-air-dependency`)

**Input**: Synchronous AIR operations (`air.herd`, `air.dma_memcpy_nd`)  
**Output**: Async AIR with tokens, `wait_all` operations, `execute` ops

Change directory:
```bash
cd /scratch/general/vast/u1418973/mlir-air
```

**Extract ACDG and print all intermediate IRs after each pass:**
```bash
./install/bin/air-opt \
  mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  -air-dependency \
  -mlir-print-ir-after-all \
  2>&1 | tee /tmp/stage1_extract_acdg.txt
```

**Just extract without verbose output:**
```bash
./install/bin/air-opt \
  mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  -air-dependency > /tmp/stage1_result.mlir
```

> **What happens:**  
> Every operation (`dma_memcpy_nd`, `execute`, etc.) gets an `async` keyword and depends on tokens from predecessor operations. Dependencies are explicit in the operation's `[ %token1, %token2, ... ]` list.

---

### Stage 2: Canonicalize ACDG (`-air-dependency-canonicalize`)

**Input**: Extracted ACDG with potentially redundant edges  
**Output**: Canonical ACDG with only critical edges (transitive reduction)

**Pipeline: extract → canonicalize**
```bash
./install/bin/air-opt \
  mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  -air-dependency \
  -canonicalize \
  -cse \
  -air-dependency-canonicalize \
  -mlir-print-ir-after-all \
  2>&1 | tee /tmp/stage2_canonicalize.txt
```

**Just the canonical form:**
```bash
./install/bin/air-opt \
  mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  -air-dependency \
  -canonicalize -cse \
  -air-dependency-canonicalize > /tmp/stage2_result.mlir
```

> **What happens:**  
> Removes redundant dependency edges.
> For example, if op C depends on both A and B, but B depends on A, then the A→C edge is removed (C only needs to depend on B).

---

### Stage 3: Optimize ACDG (`-air-dependency-schedule-opt`)

**Input**: Canonical ACDG  
**Output**: ACDG with broadcast patterns detected and labeled

**Full pipeline: extract → canonicalize → optimize**
```bash
./install/bin/air-opt \
  mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  -air-dependency \
  -canonicalize -cse \
  -air-dependency-canonicalize \
  -canonicalize -cse \
  -air-dependency-schedule-opt \
  -mlir-print-ir-after-all \
  2>&1 | tee /tmp/stage3_optimize.txt
```

**Just the optimized form:**
```bash
./install/bin/air-opt \
  mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  -air-dependency \
  -canonicalize -cse \
  -air-dependency-canonicalize \
  -canonicalize -cse \
  -air-dependency-schedule-opt > /tmp/stage3_result.mlir
```

> **What happens:**  
> Detects broadcast patterns where one DMA feeds to multiple operations. Labels them with `broadcast_pattern = #set` attributes.

---

## Step-by-Step IR Walkthrough (what to look for)

### Input (original)
```mlir
// Synchronous: operations happen in sequence
air.dma_memcpy_nd (%2[] [] [], %arg0[%c0, %c0] [%c64, %c64] [%c64, %c1])
linalg.matmul ins(%7, %8 : ...) outs(%9 : ...)
air.dma_memcpy_nd (%arg10[...], %9[] [] [])
memref.dealloc %9
```

### Stage 1 Output (`-air-dependency`)
```mlir
// Asynchronous: explicit tokens show dependencies
%4 = air.dma_memcpy_nd async [%async_token_12, %arg12] (...)  // depends on alloc
%async_token_18 = air.execute [%arg12, %5, %6, %4] {          // waits for all DMAs
  linalg.matmul ins(...)
}
%7 = air.dma_memcpy_nd async [%arg12, %async_token_18] (...)  // depends on compute
%async_token_19 = air.execute [%async_token_18] {             // depends on DMA out
  memref.dealloc (...)
}
```

### Stage 2 Output (`-air-dependency-canonicalize`)
```mlir
// Same but with fewer dependency edges (redundant ones removed)
// If %4 depends on A and B, and B depends on A:
//   Before: %compute = air.execute [A, B, %4]
//   After:  %compute = air.execute [B, %4]
//   (A is removed because it's already implied by B→%4→compute)
```

### Stage 3 Output (`-air-dependency-schedule-opt`)
```mlir
// New: broadcast patterns are labeled
%5 = air.dma_memcpy_nd async [%async_token_14, %arg12] (...)
         {broadcast_pattern = #set1, id = 2 : i32}
//       ^^^^^^^^^^^^^^^^^^^^^^
//       This DMA is broadcast to multiple consumers!
```

---

## Visualizing the ACDG (bonus)

**Generate a `.dot` graph file:**
```bash
./install/bin/air-opt \
  mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  -air-dependency \
  -canonicalize -cse \
  -air-dependency-canonicalize \
  -canonicalize -cse \
  -air-dependency-parse-graph \
  -air-dependency-parse-graph-filename=/tmp/acdg.dot
```

**View with graphviz:**
```bash
dot -Tsvg /tmp/acdg.dot -o /tmp/acdg.svg
```

---

## Comparing Input vs Output

**See input:**
```bash
cat mlir/test/Transform/AIRDependency/matmul_nd.mlir | head -80
```

**Compare stage outputs side-by-side:**
```bash
diff -u <(./install/bin/air-opt mlir/test/Transform/AIRDependency/matmul_nd.mlir) \
         <(./install/bin/air-opt mlir/test/Transform/AIRDependency/matmul_nd.mlir -air-dependency)
```

---

## Key Concepts

| Concept                   | Meaning                                                       |
|---------------------------|---------------------------------------------------------------|
| `air.execute`             | Wraps synchronous operations, has input tokens and output token|
| `air.dma_memcpy_nd async` | Asynchronous DMA, depends on tokens in `[...]`                |
| `air.wait_all async`      | Merges multiple token dependencies into one                   |
| `async.token`             | Represents completion of an operation                         |
| Transitive reduction      | Removing redundant edges: if A→B→C, remove A→C               |
| Broadcast pattern         | One DMA feeds to multiple operations                          |

---

## Full Compilation Pipeline (for reference)

From `docs/GEMMCaseStudy.md`:

1. Convert to AIR (`linalg→herd→dma`)
2. Extract ACDG (`-air-dependency`)
3. Canonicalize (`-air-dependency-canonicalize`)
4. Optimize (`-air-dependency-schedule-opt`)
5. Lower DMA→channels (`-air-dma-to-channel`)
6. Convert to AIE dialect
7. Optimize for hardware

The ACDG stages (2-4) are the core compiler work for async concurrency.