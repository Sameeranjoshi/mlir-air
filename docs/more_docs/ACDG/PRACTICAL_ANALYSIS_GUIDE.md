# Practical ACDG Analysis Workflow

This guide shows you how to use the `analyze_acdg.py` script to understand dependency transformations in real MLIR code.

## Quick Start

### 1. Transform a test case
```bash
cd /scratch/general/vast/u1418973/mlir-air

# Run the pass on a test file
./install/bin/air-opt mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  -air-dependency > /tmp/output.mlir

# Analyze the result
python3 analyze_acdg.py /tmp/output.mlir
```

### 2. What You'll See

The analysis shows:
- **Total async operations**: Count of all air.execute, air.dma, air.wait_all, etc.
- **Operation breakdown**: How many of each type
- **Detailed list**: Each operation's token, dependencies, and ID
- **Dependency patterns**: Root operations (no deps), join operations (multiple deps)
- **Token usage**: Which tokens are produced but never used (potential issues)

## Real Examples

### Example 1: matmul_nd.mlir (Producer-Consumer Pattern)

```
$ python3 analyze_acdg.py /tmp/matmul_async.mlir

Total async operations: 11

Operation Breakdown:
  air.dma_memcpy_nd              8 operations
  air.herd                       1 operations
  air.wait_all                   2 operations

Root operations (no dependencies): 2
  %3  = air.herd       (id=1)
  %7  = air.wait_all   (id=1)

Join operations (multiple dependencies): 5
  %2  depends on: %async_token_3, %async_token_8
  %9  depends on: %arg11, %async_token_19
  %10 depends on: %arg11, %async_token_21
  ...
```

**What this tells you:**
- Multiple DMAs (8 total) - data movement operations
- 2 root operations with no dependencies (can start immediately)
- 5 join points (operations that wait for multiple other ops)
- This is a **producer-consumer** pattern where data must be copied before computation

### Example 2: affine_if.mlir (Branching Pattern)

```
$ python3 analyze_acdg.py /tmp/affine_if_async.mlir

Total async operations: 11

Operation Breakdown:
  air.channel                    4 operations
  air.wait_all                   6 operations
  air.herd                       1 operations

Root operations (no dependencies): 5
  %0 = air.herd
  %3 = air.channel
  %7 = air.channel
  %8 = air.channel
  %7 = air.channel

Join operations (multiple dependencies): 2
  %1  depends on: %async_token, %async_token
  %3  depends on: %1, %1
```

**What this tells you:**
- 4 channel operations (communication between branches)
- 6 wait_all operations (synchronization points)
- Multiple channel operations can start independently
- Synchronization points (%1, %3) join them back together
- This is a **branching** pattern with explicit synchronization

## Workflow: Debugging a Transformation

### Step 1: Check if transformation happened
```bash
./install/bin/air-opt my_file.mlir -air-dependency > /tmp/output.mlir

# Quick check - should see air.execute, async tokens
grep -c "air.execute\|async\|air.wait_all" /tmp/output.mlir
```

### Step 2: Analyze the dependency structure
```bash
python3 analyze_acdg.py /tmp/output.mlir

# Look for:
# - Are there any async operations? (count should be > 0)
# - Are there root operations? (some should have no deps)
# - Are there join operations? (complex dependencies)
```

### Step 3: Spot problems
The analysis highlights potential issues:

**Problem: Too many unused tokens**
```
Unused tokens:
  %0 %1 %2 %3 %4 %5 %6 %7 %8 %9
```
→ Many tokens are created but never used. This might indicate incomplete dependency analysis.

**Problem: No root operations**
```
Root operations (no dependencies): 0
```
→ Every operation depends on something. Check if there's a circular dependency.

**Problem: Too many join points**
```
Join operations (multiple dependencies): 15
```
→ Many operations synchronize on multiple tokens. This might serialize execution.

### Step 4: Compare before and after

```bash
# Save original
./install/bin/air-opt my_file.mlir --pass-pipeline="builtin.module()" > /tmp/before.mlir

# Transform
./install/bin/air-opt my_file.mlir -air-dependency > /tmp/after.mlir

# Analyze both
python3 analyze_acdg.py /tmp/before.mlir
python3 analyze_acdg.py /tmp/after.mlir

# Compare results
diff <(python3 analyze_acdg.py /tmp/before.mlir) \
     <(python3 analyze_acdg.py /tmp/after.mlir)
```

## Understanding the Output

### Operation Types

| Type | Meaning | Example |
|------|---------|---------|
| `air.execute` | Wraps synchronous ops, produces token | `%t, %r = air.execute { memref.alloc() }` |
| `air.dma_memcpy_nd` | Data movement, depends on alloc | `%t = air.dma async [%alloc_token]` |
| `air.wait_all` | Synchronization point | `%t = air.wait_all async [%t1, %t2]` |
| `air.herd` | Parallel computation | `%t = air.herd async [%dma_tokens]` |
| `air.channel.put` | Send data | `%t = air.channel.put async [%t]` |
| `air.channel.get` | Receive data | `%t = air.channel.get async [%t]` |

### Dependencies Explained

When you see:
```
air.dma_memcpy_nd    %0    1    %async_token_4
```

This means:
- **Token**: `%0` is created by this operation
- **ID**: `1` groups related operations
- **Depends on**: `%async_token_4` (must complete first)

### Root Operations

Operations with `[none]` dependencies can start immediately:
```
Root operations (no dependencies): 2
  %3 = air.herd (id=1)
  %7 = air.wait_all (id=1)
```

These are typically:
- Initial wait_all (empty token)
- Hierarchy operations (herd, segment, launch)
- Operations that don't depend on memory

### Join Operations

Operations with multiple dependencies wait for all inputs:
```
Join operations (multiple dependencies): 5
  %2  depends on: %async_token_3, %async_token_8
```

These represent:
- Synchronization barriers
- Operations that need multiple inputs
- Potential parallelization opportunities (all inputs can happen in parallel)

## Advanced Analysis

### Finding Sequential Bottlenecks

If most operations are joins (depend on multiple tokens), execution becomes sequential:
```bash
python3 analyze_acdg.py /tmp/output.mlir | grep "Join operations"

# Count single-dependency ops
grep "depends on: %[a-z_]*$" /tmp/output.mlir | wc -l

# Count multi-dependency ops
grep "depends on:.*," /tmp/output.mlir | wc -l
```

If multi-dependency >> single-dependency, you have a serial bottleneck.

### Tracing a Specific Token

To understand what a token controls:

```bash
# Find who produces %async_token_4
grep -n "= air.execute" /tmp/output.mlir | grep -B2 "%async_token_4"

# Find who depends on it
grep -n "async \[.*%async_token_4" /tmp/output.mlir
```

### Comparing Canonicalization

See how much the pass removes:
```bash
./install/bin/air-opt test.mlir -air-dependency > /tmp/raw.mlir
./install/bin/air-opt test.mlir \
  --pass-pipeline="builtin.module(air-dependency,air-dependency-canonicalize)" \
  > /tmp/canonical.mlir

echo "Before canonicalization:"
python3 analyze_acdg.py /tmp/raw.mlir | grep "Total\|Unused"

echo "After canonicalization:"
python3 analyze_acdg.py /tmp/canonical.mlir | grep "Total\|Unused"
```

## Test Cases to Try

From most complex to simplest:

```bash
# Hierarchies with multiple nesting levels
./install/bin/air-opt mlir/test/Transform/AIRDependency/air_hierarchy.mlir -air-dependency > /tmp/h.mlir
python3 analyze_acdg.py /tmp/h.mlir

# Complex memory operations
./install/bin/air-opt mlir/test/Transform/AIRDependency/reshape_dependency.mlir -air-dependency > /tmp/r.mlir
python3 analyze_acdg.py /tmp/r.mlir

# Branching
./install/bin/air-opt mlir/test/Transform/AIRDependency/affine_if.mlir -air-dependency > /tmp/a.mlir
python3 analyze_acdg.py /tmp/a.mlir

# Loops
./install/bin/air-opt mlir/test/Transform/AIRDependency/scf_for.mlir -air-dependency > /tmp/f.mlir
python3 analyze_acdg.py /tmp/f.mlir

# Basic matmul
./install/bin/air-opt mlir/test/Transform/AIRDependency/matmul_nd.mlir -air-dependency > /tmp/m.mlir
python3 analyze_acdg.py /tmp/m.mlir

# Simplest DMA
./install/bin/air-opt mlir/test/Transform/AIRDependency/dma_memcpy_nd.mlir -air-dependency > /tmp/d.mlir
python3 analyze_acdg.py /tmp/d.mlir
```

## Summary

Use `analyze_acdg.py` to:
1. ✅ **Verify** that transformation happened
2. ✅ **Understand** operation counts and types
3. ✅ **Identify** parallelization opportunities (independent roots)
4. ✅ **Find** bottlenecks (excessive joins)
5. ✅ **Debug** unexpected token counts
6. ✅ **Compare** before/after transformations

The script gives you immediate insight into the dependency structure without needing to parse the full MLIR manually!
