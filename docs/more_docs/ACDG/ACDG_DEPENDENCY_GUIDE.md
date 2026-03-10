# AIRDependency Pass: Deep Dive & Learning Guide

## Overview

The **AIRDependency** pass is the heart of MLIR-AIR's asynchronous execution model. It transforms synchronous AIR operations into an **Asynchronous Compute Data-flow Graph (ACDG)** where explicit dependency tokens track the execution order and data dependencies between operations.

**Goal:** Convert this:
```mlir
memref.alloc()
air.dma_memcpy_nd(...)
air.channel.put(...)
```

Into this:
```mlir
%token0, %results = air.execute -> (...) { memref.alloc() }
%token1 = air.dma_memcpy_nd async [%token0] (...)
%token2 = air.channel.put async [%token1] (...)
```

---

## Part 1: Core Concepts

### 1.1 What is an Async Token?

An **async token** (`!air.async.token`) is a *logical timestamp* representing:
- **When an operation completes** in the dependency graph
- **What operations must wait** before executing

```mlir
%token, %result = air.execute -> (memref<...>) {
  %alloc = memref.alloc() : memref<32x32xi32, 2>
  air.execute_terminator %alloc : memref<32x32xi32, 2>
} {id = 1 : i32}
```

- `%token` = "this operation is done"
- `%result` = the allocated memory

### 1.2 Dependency Edges

Operations are connected via their tokens to form a DAG:

```
[alloc: token0] → [dma: token1 depends on token0] → [channel.put: token2 depends on token1]
```

This means:
- DMA cannot start until alloc completes
- channel.put cannot start until DMA completes

### 1.3 Why Tokens Matter

Without tokens, the compiler cannot:
1. **Reorder operations** safely (might violate data dependencies)
2. **Parallelize** operations that don't depend on each other
3. **Schedule** operations optimally across hardware

With tokens, the compiler has explicit information about execution order.

---

## Part 2: The Three-Phase Transformation

### Phase 1: Convert Synchronous Ops → Async Ops (with empty deps)

**Input:**
```mlir
%alloc = memref.alloc() : memref<32x32xi32, 2>
air.dma_memcpy_nd (%alloc[] [] [], %src[...]) {id = 1 : i32}
air.channel.put @ch[] (%alloc[] [] [])
```

**After Phase 1:**
```mlir
%t0, %r0 = air.execute -> (memref<32x32xi32, 2>) {
  %alloc = memref.alloc() : memref<32x32xi32, 2>
  air.execute_terminator %alloc : memref<32x32xi32, 2>
} {id = 1 : i32}

%t1 = air.dma_memcpy_nd async [... empty ...] (%r0[] [] [], %src[...]) {id = 1 : i32}

%t2 = air.channel.put async [... empty ...] @ch[] (%r0[] [] [])
```

**Key Changes:**
- Each op wrapped in `air.execute` or made explicitly async
- Each async op can have input tokens (initially empty)
- Operations return their "done" token

### Phase 2: Trace Data & Control Dependencies

The pass analyzes:
1. **Direct data dependencies**: "This op reads what that op writes"
2. **Memory access patterns**: Which operations touch which memory locations
3. **Control flow**: Branches and loops

**Example tracing:**

```mlir
// alloc produces %r0 (memory)
%t0, %r0 = air.execute -> (memref<32x32xi32, 2>) {
  memref.alloc()
}

// dma reads from %r0 → MUST wait for %t0
%t1 = air.dma_memcpy_nd async [%t0] (%r0[...], ...)

// channel.put reads from %r0 → MUST wait for both %t0 AND %t1
// (because dma also modifies its state)
%t2 = air.channel.put async [%t0, %t1] @ch[] (%r0[] [] [])
```

### Phase 3: Synchronization Points (air.wait_all)

When you need to ensure an operation has completed:

```mlir
%t2 = air.channel.put async [%t0, %t1] @ch[] (...)
%t3 = air.wait_all async [%t2]  // "Wait for channel.put to finish"
affine.yield %t3 : !air.async.token
```

This is essential in:
- **Loop bodies**: Synchronizing each iteration
- **Branches**: Joining tokens from different paths
- **Deallocation**: Ensuring memory isn't freed while still in use

---

## Part 3: Transformation by Operation Type

### 3.1 Memory Allocation & Deallocation

**Input:**
```mlir
%alloc = memref.alloc() : memref<32x32xi32, 2>
```

**Output:**
```mlir
%token, %results = air.execute -> (memref<32x32xi32, 2>) {
  %alloc = memref.alloc() : memref<32x32xi32, 2>
  air.execute_terminator %alloc : memref<32x32xi32, 2>
} {id = 1 : i32}
```

**Why?** Because alloc is an operation with a result; we need a token to track when it's done.

---

### 3.2 DMA Operations

**Input:**
```mlir
air.dma_memcpy_nd (%dst[] [] [], %src[...]) {id = 1 : i32} : (...)
```

**Output:**
```mlir
%token = air.dma_memcpy_nd async [%deps...] (%dst[] [] [], %src[...]) {id = 1 : i32}
```

The pass infers `%deps` by analyzing:
- Which operations write to `%src` (producer)
- Which operations write to `%dst` (producer)

---

### 3.3 Channel Operations (put/get)

**Input:**
```mlir
air.channel.put @channel[] (%data[] [] [])
air.channel.get @channel[] (%data[] [] [])
```

**Output:**
```mlir
%t_put = air.channel.put async [%producer_of_data] @channel[] (...)
%t_get = air.channel.get async [%t_put] @channel[] (...)  // serialized!
```

**Key insight:** Put and get on same channel are automatically ordered:
- If A puts on channel X and B gets from channel X
- B depends on A's put
- A implicitly depends on B's previous operations on that memory

---

### 3.4 Compute Operations (linalg, func.call)

**Input:**
```mlir
linalg.matmul ins(%A, %B : ...) outs(%C : ...)
```

**Output:**
```mlir
%token = air.execute [%deps...] {
  linalg.matmul ins(%A, %B : ...) outs(%C : ...)
} {id = 5 : i32}
```

The pass finds:
- Producers of `%A` and `%B` → input dependencies
- Producers of `%C` → WAW (write-after-write) dependency

---

### 3.5 Hierarchy Operations (air.herd, air.segment, air.launch)

**Input:**
```mlir
air.herd tile(...) in(...) {
  ... body ...
}
```

**Output:**
```mlir
%token = air.herd async [%input_tokens] tile(...) in(...) {
  ... transformed body ...
} attributes {id = 1 : i32}
```

The pass:
1. Recursively transforms the body
2. Collects all input/output dependencies
3. Adds the herd itself as an async operation

---

## Part 4: Complex Example - affine_if.mlir

Let's trace through the actual test case:

### Input (Synchronous):
```mlir
air.herd @herd_0 tile(...) {
  %2 = memref.alloc() : memref<1x1x2048xi32, 2 : i32>

  affine.if affine_set<()[s0] : (s0 == 0)>()[%arg4] {
    air.channel.put @channel_0[%arg4] (%2[] [] [])
  } else {
    affine.if affine_set<()[s0] : (s0 - 1 >= 0, -s0 + 2 >= 0)>()[%arg4] {
      %c1 = arith.constant 1 : index
      %iv_sub1 = arith.subi %arg4, %c1 : index
      air.channel.get @channel_0[%iv_sub1] (%2[] [] [])
      air.channel.put @channel_0[%arg4] (%2[] [] [])
    } else {
      air.channel.get @channel_0[%iv_sub1] (%2[] [] [])
    }
  }
}
```

### Transformation Steps:

**Step 1: Wrap memref.alloc**
```mlir
%async_token, %results = air.execute -> (memref<1x1x2048xi32, 2 : i32>) {
  %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
  air.execute_terminator %alloc : memref<1x1x2048xi32, 2 : i32>
} {id = 1 : i32}
```

**Step 2: Create synchronization point before affine.if**
```mlir
%1 = air.wait_all async [%async_token, %async_token] {id = 4 : i32}
```

**Step 3: Make affine.if return a token**
```mlir
%2 = affine.if #set()[%arg2] -> !air.async.token {
  %3 = air.channel.put async [%1] @channel_0[%arg2] (%results[] [] []) {id = 1 : i32}
  %4 = air.wait_all async [%3] {id = 5 : i32}
  affine.yield %4 : !air.async.token
} else {
  // similar transformation...
  affine.yield %5 : !air.async.token
}
```

**Key Pattern:**
```
affine.if -> affine.yield
  input tokens → operations → output token
```

The if/else becomes a **token-producing operation** that can be passed downstream.

---

## Part 5: Dependency Analysis Strategy

The pass uses a **DirectedAdjacencyMap** to build the dependency graph:

### Key Data Structures:

```cpp
// A node in the graph
struct executeNode {
  std::string asyncEventName;      // "alloc_0", "dma_1", etc
  std::string asyncEventType;       // "execute", "dma", "channel", etc
  unsigned operationId;             // Unique ID
};

// The graph itself
using ExecuteGraph = TypedDirectedAdjacencyMap<executeNode>;
```

### Dependency Detection Algorithm:

1. **For each operation `op2` that reads from a memref `M`:**
   - Find all operations `op1` that **write** to `M`
   - Add edge: `op1 → op2` (op2 depends on op1)

2. **For each operation `op2` that writes to a memref `M`:**
   - Find all operations `op1` that **write** to `M` (WAW dependency)
   - Add edge: `op1 → op2`

3. **For channel operations:**
   - Puts and gets are strictly ordered
   - `put A` → `get A` forms a chain

4. **For loop bodies:**
   - Each iteration depends on the previous
   - Added via `scf.yield` statements

---

## Part 6: Handling Loops and Branches

### For SCF (Structured Control Flow) Loops:

**Input:**
```mlir
scf.for %i = %c0 to %n step %c1 {
  %t_alloc, %result = air.execute -> (...) { memref.alloc() }
  air.dma_memcpy_nd (%result[], ...) {id = 1 : i32}
}
```

**Transformation:**
```mlir
scf.for %i = %c0 to %n step %c1 iter_args(%arg_token = %initial_token) -> (!air.async.token) {
  %t_alloc, %result = air.execute [%arg_token] -> (...) { memref.alloc() }
  %t_dma = air.dma_memcpy_nd async [%t_alloc] (...)
  %t_sync = air.wait_all async [%t_dma] {id = 2 : i32}
  scf.yield %t_sync : !air.async.token
}
```

**Key Changes:**
- `iter_args` added to carry the async token through iterations
- Each iteration starts with the **previous iteration's output token**
- `scf.yield` returns the token for the next iteration

### For Affine Loops (affine.for):

Similar pattern but with `affine.yield`

---

## Part 7: Real-World Patterns

### Pattern 1: Producer-Consumer Chain

```mlir
// Producer: writes data
%t0, %data = air.execute -> (memref<...>) { memref.alloc() }

// Consumer: reads data (depends on producer)
%t1 = air.dma_memcpy_nd async [%t0] (%data[] [] [], ...)

// Another consumer (depends on producer)
%t2 = air.channel.put async [%t0] @ch[] (%data[] [] [])
```

Both `t1` and `t2` depend on `t0`, but **not on each other** → can execute in parallel!

### Pattern 2: Sequential Writes

```mlir
// Write #1
%t0 = air.dma_memcpy_nd async [...] (%buf[] [] [], %src1[...])

// Write #2 (WAW dependency)
%t1 = air.dma_memcpy_nd async [%t0] (%buf[] [] [], %src2[...])

// Read (depends on latest write)
%t2 = air.channel.put async [%t1] @ch[] (%buf[] [] [])
```

Must be sequential because they all touch `%buf`.

### Pattern 3: Parallel Branches

```mlir
scf.if %cond -> (!air.async.token) {
  %t0 = air.execute { ... } {id = 1 : i32}
  affine.yield %t0 : !air.async.token
} else {
  %t1 = air.execute { ... } {id = 2 : i32}
  affine.yield %t1 : !air.async.token
}
// Both branches can execute in parallel on different tiles
```

---

## Part 8: Common Pitfalls & Insights

### 1. Why Alloc Needs Execution Wrapping

```mlir
// ❌ BEFORE (no token):
%alloc = memref.alloc()
use(%alloc)

// ✅ AFTER (with token):
%token, %result = air.execute -> (memref) { memref.alloc() }
use_with_dependency(%result, [%token])
```

Without the token, downstream ops don't know **when** the allocation finishes.

### 2. Why air.execute vs Direct Async

Some ops are already async (`air.dma_memcpy_nd`, `air.channel.put`).
Others need wrapping in `air.execute` (alloc, dealloc, linalg ops).

### 3. Synchronization Points (air.wait_all)

```mlir
%t1 = air.dma_memcpy_nd async [...]
%t2 = air.channel.put async [%t1]
%t_sync = air.wait_all async [%t2]  // "WAIT" - forces synchronization
affine.yield %t_sync
```

Used when:
- **Exiting a scope** (loop, branch, herd) - need to return a completed token
- **Joining multiple paths** - multiple producers → single consumer
- **Memory deallocation** - must ensure memory isn't freed while still in use

### 4. The Role of Operation IDs

```mlir
air.execute { ... } {id = 5 : i32}
air.dma_memcpy_nd async [...] { ..., id = 1 : i32 }
```

IDs help:
- **Tracing** which operation is which
- **Debugging** dependency graphs
- **Reproducibility** across passes

---

## Part 9: Testing & Verification

To verify the pass is working correctly:

### Test the affine_if example:
```bash
cd /scratch/general/vast/u1418973/mlir-air

# Run with just the dependency pass
./install/bin/air-opt test/Transform/AIRDependency/affine_if.mlir \
  -air-dependency | less

# Run with verbose output
./install/bin/air-opt test/Transform/AIRDependency/affine_if.mlir \
  -air-dependency -debug 2>&1 | grep "air-dependency"
```

### Verify with FileCheck:
```bash
cd mlir/test/Transform/AIRDependency

# This runs the FileCheck patterns defined in the test comments
../../.. /install/bin/air-opt affine_if.mlir -air-dependency | FileCheck affine_if.mlir
```

---

## Part 10: Pipeline Overview

After AIRDependency, the next passes are:

1. **AIRDependencyCanonicalize**
   - Removes redundant edges using transitive reduction
   - Cleans up the dependency graph

2. **AIRDependencyScheduleOpt**
   - Detects broadcast patterns
   - Optimizes operation ordering

3. **AIRDmaToChannel**
   - Converts async DMA to `air.channel.put/get`

4. **AIRToAIE**
   - Converts AIR dialect to AIE (hardware) dialect

---

## Summary

| Aspect | Details |
|--------|---------|
| **Input** | Synchronous AIR operations |
| **Output** | Asynchronous operations with explicit tokens |
| **Core Unit** | `!air.async.token` (represents completion) |
| **Graph** | Dependency DAG built from data flow analysis |
| **Key Ops** | `air.execute`, `air.wait_all`, async variants of existing ops |
| **Use Case** | Enable hardware scheduling and parallel execution |

---

## Additional Resources

Key files to explore:
- `mlir/lib/Transform/AIRDependency.cpp` - Main pass logic
- `mlir/lib/Util/Dependency.cpp` - Dependency analysis utilities
- `mlir/test/Transform/AIRDependency/*.mlir` - Test cases with patterns
- `mlir/lib/Transform/AIRDependencyCanonicalize.cpp` - Graph optimization

