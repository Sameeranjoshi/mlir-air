# AIRDependency Pass: Visual Examples & Transformations

This document provides concrete before/after examples from the test suite with visual diagrams.

---

## Example 1: Simple Matmul (matmul_nd.mlir)

### The Core Pattern: Data Producer-Consumer

#### BEFORE (Synchronous):
```mlir
%alloc = memref.alloc() : memref<64x64xi32, 1>
air.dma_memcpy_nd (%alloc[] [] [], %src[...]) {id = 1}
linalg.matmul ins(%alloc, ...) outs(...)
```

#### Dependency Flow:
```
Instruction Order:
┌─────────────────┐
│  memref.alloc   │ ← Creates memory
└────────┬────────┘
         │ (memory is created)
         ▼
┌─────────────────┐
│  air.dma_memcpy │ ← Fills memory
└────────┬────────┘
         │ (memory is filled)
         ▼
┌─────────────────┐
│ linalg.matmul   │ ← Uses memory
└─────────────────┘
```

The compiler sees these ops in **sequence**, but doesn't explicitly know *why*.

---

#### AFTER (Asynchronous with Explicit Dependencies):
```mlir
%t0, %result = air.execute -> (memref<64x64xi32, 1>) {
  %alloc = memref.alloc() : memref<64x64xi32, 1>
  air.execute_terminator %alloc : memref<64x64xi32, 1>
} {id = 1 : i32}

%t1 = air.dma_memcpy_nd async [%t0] (%result[] [] [], %src[...]) {id = 2 : i32}

%t2 = air.execute [%t0, %t1] {
  linalg.matmul ins(%result, ...) outs(...)
} {id = 3 : i32}
```

#### Dependency Graph:
```
GRAPH:
┌─────────────────────────────────────────────────┐
│ Operation ID Graph (Vertices = Operations)      │
├─────────────────────────────────────────────────┤
│                                                 │
│  [Alloc: t0]                                    │
│       │                                         │
│       └──► [DMA: t1]                            │
│            │                                    │
│            └──► [Matmul: t2]                    │
│                                                 │
│  Legend:                                        │
│  t0 = "alloc is done"                           │
│  t1 = "dma is done" (after t0)                  │
│  t2 = "matmul is done" (after t1)               │
└─────────────────────────────────────────────────┘
```

#### Temporal Execution:
```
TIME ──────────────────────────────────────────────►
     │
     ├─ [T0] Allocate (wait_all[])
     │  └─► Done at T_alloc
     │
     ├─ [T1] DMA Wait([T_alloc]), Execute, Done
     │  └─► Done at T_dma
     │
     └─ [T2] Matmul Wait([T_dma]), Execute, Done
        └─► Done at T_matmul
```

---

## Example 2: Affine If (affine_if.mlir)

This shows how **branches** become token-producing operations.

### The Pattern: Conditional Execution

#### BEFORE (Synchronous):
```mlir
%alloc = memref.alloc() : memref<1x1x2048xi32, 2>

affine.if %cond_is_zero() {
  air.channel.put @ch[%tile] (%alloc[] [] [])
} else {
  affine.if %cond_is_one_or_two() {
    air.channel.get @ch[%prev] (%alloc[] [] [])
    air.channel.put @ch[%tile] (%alloc[] [] [])
  } else {
    air.channel.get @ch[%prev] (%alloc[] [] [])
  }
}
```

#### Problem:
No explicit synchronization between:
- When allocation completes
- When branch executes
- When channels operate

#### AFTER (Asynchronous with Branches Returning Tokens):
```mlir
%t0, %results = air.execute -> (memref<1x1x2048xi32, 2 : i32>) {
  %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
  air.execute_terminator %alloc : memref<1x1x2048xi32, 2 : i32>
} {id = 1 : i32}

%t_sync = air.wait_all async [%t0, %t0] {id = 4 : i32}

%t_if = affine.if #set()[%arg2] -> !air.async.token {
  // Branch 1: tile 0 → PUT
  %t_put = air.channel.put async [%t_sync] @channel_0[%arg2]
    (%results[] [] []) {id = 1 : i32}
  %t_wait1 = air.wait_all async [%t_put] {id = 5 : i32}
  affine.yield %t_wait1 : !air.async.token

} else {
  // Branch 2: tile 1-2 → GET then PUT
  %t_sync2 = air.wait_all async [%t_sync, %t_sync] {id = 1 : i32}

  %t_if2 = affine.if #set1()[%arg2] -> !air.async.token {
    // Nested: GET then PUT
    %t_get = air.channel.get async [%t_sync2] @channel_0[%prev]
      (%results[] [] []) {id = 2 : i32}
    %t_put = air.channel.put async [%t_get] @channel_0[%arg2]
      (%results[] [] []) {id = 3 : i32}
    %t_wait2 = air.wait_all async [%t_put] {id = 2 : i32}
    affine.yield %t_wait2 : !air.async.token

  } else {
    // Nested: just GET
    %t_get = air.channel.get async [%t_sync2] @channel_0[%prev]
      (%results[] [] []) {id = 4 : i32}
    %t_wait3 = air.wait_all async [%t_get] {id = 3 : i32}
    affine.yield %t_wait3 : !air.async.token
  }

  %t_sync3 = air.wait_all async [%t_sync2] {id = 6 : i32}
  affine.yield %t_sync3 : !air.async.token
}
```

#### Dependency Visualization:

```
BRANCH EXECUTION WITH TOKENS:

Input: Single alloc producing memory
       %t0 ──┐
             │
        ┌────▼─────┐
        │air.wait_all
        └────┬─────┘
             │
         %t_sync
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌─────────────┐  ┌──────────────┐
│ BRANCH 1    │  │ BRANCH 2     │
│ (cond=zero) │  │ (cond!=zero) │
└────┬────────┘  └──┬───────────┘
     │             │
     │        ┌────▼──────────┐
     │        │ Nested If     │
     │        └────┬──────────┘
     │             │
     ▼             ▼
   PUT-1    GET-then-PUT OR just-GET
     │             │
     └────┬────────┘
          │
       %t_if   (final token from either branch)


Execution:
┌─ ALLOC        [depends on: nothing]
│   └─ done at t0
│
├─ WAIT_ALL    [depends on: t0]
│   └─ done at t_sync
│
├─ BRANCH_IF   [depends on: t_sync]
│   │
│   ├─ THEN: PUT       [depends on: t_sync]
│   │         └─ done at t_put1
│   │         └─ WAIT_ALL [depends on: t_put1]
│   │                   └─ yields t1 from then-branch
│   │
│   └─ ELSE: NESTED_IF [depends on: t_sync]
│           ├─ INNER_THEN: GET → PUT  [sequential]
│           │              └─ yields t2
│           └─ INNER_ELSE: just GET
│                         └─ yields t3
│
└─ Final result: t_if (from whichever branch executed)
```

#### Token Semantics:
```
%t_if represents the "completion token" of the affine.if operation itself:
- If condition true:  t_if = t_wait1 (from then branch)
- If condition false: t_if = result from else branch

Downstream operations can depend on %t_if without needing to know
which branch executed!
```

---

## Example 3: Loop with Dependencies (scf_for.mlir)

This shows how **iterative operations** get threaded through loops.

#### BEFORE (Synchronous Loop):
```mlir
scf.for %i = %c0 to %n step %c1 {
  %alloc = memref.alloc() : memref<4x2xf32, 2>
  air.dma_memcpy_nd (%alloc[] [] [], %src[...]) {id = 1}
  air.dma_memcpy_nd (%dst[...], %alloc[] [] []) {id = 2}
  memref.dealloc %alloc
}
```

Each iteration has **sequential dependencies** but **no explicit synchronization**.

#### AFTER (Asynchronous Loop with Token Threading):
```mlir
%initial_token = air.wait_all async {id = 3 : i32}

%final_token = scf.for %i = %c0 to %n step %c1
    iter_args(%arg_token = %initial_token)
    -> (!air.async.token) {

  // === ITERATION BODY ===

  // Step 1: Alloc (depends on previous iteration)
  %t0, %result = air.execute [%arg_token] -> (memref<4x2xf32, 2>) {
    %alloc = memref.alloc() : memref<4x2xf32, 2>
    air.execute_terminator %alloc : memref<4x2xf32, 2>
  } {id = 1 : i32}

  // Step 2: DMA in (depends on alloc)
  %t1 = air.dma_memcpy_nd async [%arg_token, %t0]
    (%result[] [] [], %src[...]) {id = 1 : i32}

  // Step 3: DMA out (depends on DMA in)
  %t2 = air.dma_memcpy_nd async [%arg_token, %t1]
    (%dst[...], %result[] [] []) {id = 2 : i32}

  // Step 4: Dealloc (depends on all DMAs)
  %t3 = air.execute [%arg_token, %t2] {
    memref.dealloc %result : memref<4x2xf32, 2>
  } {id = 2 : i32}

  // Step 5: Synchronize and pass to next iteration
  %t_iter_done = air.wait_all async [%t2] {id = 2 : i32}

  scf.yield %t_iter_done : !air.async.token

  // === Next iteration starts with t_iter_done as %arg_token ===
}
```

#### Loop Execution Timeline:
```
ITERATION 0:
  t_init = air.wait_all[]
  t0_alloc = air.execute[t_init]
  t0_dma_in = air.dma[t0_alloc]
  t0_dma_out = air.dma[t0_dma_in]
  t0_dealloc = air.execute[t0_dma_out]
  t0_sync = air.wait_all[t0_dma_out]
  → yields t0_sync

ITERATION 1:
  t1_alloc = air.execute[t0_sync]        ← Waits for previous iteration!
  t1_dma_in = air.dma[t1_alloc]
  t1_dma_out = air.dma[t1_dma_in]
  t1_dealloc = air.execute[t1_dma_out]
  t1_sync = air.wait_all[t1_dma_out]
  → yields t1_sync

ITERATION 2:
  ... (similar pattern)

TIME PROGRESSION:
T=0   ├─ [Iter 0] Alloc
      │   └─ Done at T_a0
T=a0  ├─ [Iter 0] DMA_in
      │   └─ Done at T_d0in
T=d0in├─ [Iter 0] DMA_out  ────┬──────────────┐
      │   └─ Done at T_d0out   │              │
T=d0out├─ [Iter 0] Dealloc     │              │
      │                        │              │
      ├─ [Iter 1] Alloc ◄──────┘              │
      │   └─ Done at T_a1                     │
T=a1  ├─ [Iter 1] DMA_in                      │ (overlaps)
      │   └─ Done at T_d1in                   │
T=d1in├─ [Iter 1] DMA_out ◄───────────────────┘
      │   └─ Done at T_d1out
      ...
```

**Key Insight:** Even though iteration 1 must *logically* wait for iteration 0's memory to be deallocated, iteration 1's **allocation** (which is independent) can start as soon as iteration 0's critical path completes.

---

## Example 4: Parallel Herds with No Dependencies (parallel_herds.mlir)

When multiple herds operate on **independent memory**, they execute in parallel.

#### BEFORE:
```mlir
scf.for %arg18 = %c0 to %c1024 step %c64 {
  // Herd 1 at offset %4
  %6 = memref.alloc() : memref<64x64xbf16, 1>
  %7 = memref.alloc() : memref<64x64xbf16, 1>
  %8 = memref.alloc() : memref<64x64xbf16, 1>
  air.dma_memcpy_nd (%6[] [] [], %arg14[%4, %arg18] [%c64, %c64])
  air.dma_memcpy_nd (%7[] [] [], %arg15[%arg18, %3] [%c64, %c64])
  air.dma_memcpy_nd (%8[] [] [], %arg16[%4, %3] [%c64, %c64])
  air.herd tile(...) { ... compute ... }
  air.dma_memcpy_nd (%arg16[%4, %3] [%c64, %c64], %8[] [] [])

  // Herd 2 at offset %5 (different memory!)
  // Same pattern but with %5 instead of %4
}
```

#### CHECK Pattern (from test):
```mlir
// CHECK: %[[EVENT0:.*]] = scf.for
// CHECK: scf.yield
// CHECK-NEXT: }
// CHECK-NOT: %[[EVENT1:.*]] = air.wait_all async [{{.*}}%[[EVENT0]]
// CHECK: %[[EVENT2:.*]] = scf.for
```

The test verifies that the **second scf.for does NOT depend on the first**!

#### AFTER (Parallel Execution):
```mlir
%t_loop1 = scf.for %arg18 = ... iter_args(...) -> (!air.async.token) {
  // All allocs, dmas, herds, deallocations for region 1
  // Yields final token
  scf.yield %t_loop1_done : !air.async.token
}

// ⚠️ NO WAIT_ALL HERE - the next scf.for doesn't depend on the first!

%t_loop2 = scf.for %arg18 = ... iter_args(...) -> (!air.async.token) {
  // All allocs, dmas, herds, deallocations for region 2
  // Yields final token
  scf.yield %t_loop2_done : !air.async.token
}

// Both loops execute INDEPENDENTLY and can run in PARALLEL
```

#### Dependency Graph:
```
PARALLEL EXECUTION:

    ┌─────────────────────┐
    │   scf.for loop 1    │
    │  (%4, memory area)  │
    └──────────┬──────────┘
               │
               └─► t_loop1 (independent)

    ┌─────────────────────┐
    │   scf.for loop 2    │
    │  (%5, memory area)  │
    └──────────┬──────────┘
               │
               └─► t_loop2 (independent)

Timeline:
T=0  ├─ Loop 1 starts
     ├─ Loop 2 starts          ◄── Can start simultaneously!
     │
     │ (both run in parallel on different tiles)
T=t  ├─ Loop 1 finishes
     ├─ Loop 2 finishes        ◄── May finish at different times

Result: PARALLELISM!
```

---

## Example 5: Deallocation with Dependencies (memref_reshape.mlir)

This shows why **deallocation must wait** for all uses.

#### BEFORE:
```mlir
%results = memref.alloc() : memref<16x4x64xi32, 1>
air.dma_memcpy_nd (%results[] [] [], %arg1[...])
air.herd tile(...) args(%arg6=%results) {
  %reshape = memref.reshape %results(...)
  linalg.matmul ins(%..., %reshape : ...) outs(%results : ...)
}
memref.dealloc %results
```

#### Problem:
If dealloc happens before the herd finishes, the reshaped memref is freed while still in use!

#### AFTER:
```mlir
%t_alloc, %results = air.execute -> (memref<16x4x64xi32, 1>) {
  %alloc = memref.alloc() : memref<16x4x64xi32, 1>
  air.execute_terminator %alloc : memref<16x4x64xi32, 1>
} {id = 1 : i32}

%t_dma = air.dma_memcpy_nd async [%t_alloc] (%results[] [] [], %arg1[...]) {id = 1 : i32}

%t_herd = air.herd async [%t_dma] tile(...) args(%arg6=%results) {
  // ... reshape and matmul ...
} {id = 1 : i32}

// ⚠️ Dealloc MUST depend on herd completion!
%t_dealloc = air.execute [%t_herd] {
  memref.dealloc %results : memref<16x4x64xi32, 1>
} {id = 2 : i32}
```

#### Execution Order:
```
%t_alloc
  ↓ (memory created)
%t_dma [%t_alloc]
  ↓ (memory filled)
%t_herd [%t_dma]
  ↓ (computation done)
%t_dealloc [%t_herd]  ◄── Explicit dependency ensures safety!
  ↓ (memory freed)
```

---

## Dependency Pattern Summary

### Pattern 1: Simple Sequential
```
Producer → Consumer
%t1 ──────→ %t2
```

### Pattern 2: Multiple Producers (Join)
```
Producer1 ──┐
            ├──→ Consumer
Producer2 ──┘
```
In code: `%t_cons = op async [%t_prod1, %t_prod2]`

### Pattern 3: Single Producer → Multiple Consumers (Branch)
```
Producer ──┬──→ Consumer1
           ├──→ Consumer2
           └──→ Consumer3
```
In code:
```mlir
%t_prod = op async [...]
%t_cons1 = op1 async [%t_prod]
%t_cons2 = op2 async [%t_prod]
%t_cons3 = op3 async [%t_prod]
```

### Pattern 4: Loops (Threading)
```
Init ──→ Loop[iter_args] ──→ yield
         ├─ Op1[%arg]
         ├─ Op2[%t1]
         └─ Op3[%t2]
```

### Pattern 5: Conditional (Yielding)
```
Init ──→ if ──→ {then: yield t1 | else: yield t2} ──→ Consumer[t_if]
```

---

## Practical Debugging Tips

### 1. Enable Debug Logging
```bash
mlir-air-opt file.mlir -air-dependency -debug 2>&1 | grep "air-dependency"
```

### 2. Visual Inspection
Look for:
- ✅ Every op either returns a token or depends on one
- ✅ No "orphaned" tokens (created but not used)
- ✅ Reasonable dependency chain (not circular!)

### 3. Check for Missing Dependencies
```mlir
// ❌ BAD - alloc and dma are not connected
%t_alloc = air.execute { memref.alloc() }
%t_dma = air.dma_memcpy_nd async [] (...)  // Empty deps!

// ✅ GOOD - dma depends on alloc
%t_dma = air.dma_memcpy_nd async [%t_alloc] (...)
```

### 4. Verify Loop Synchronization
```mlir
// ✅ GOOD - loop carries token through iterations
scf.for ... iter_args(%arg = %init) -> (!air.async.token) {
  ... ops[%arg] ...
  scf.yield %final : !air.async.token
}

// ❌ BAD - loop doesn't carry token
scf.for ... {
  ... ops ...
  scf.yield  // Missing return type!
}
```

---

This should give you a clear visual understanding of how the AIRDependency pass transforms code!

