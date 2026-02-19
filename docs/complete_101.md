# MLIR-AIR Complete Developer Reference

⚡ **AMD AI Engine Compiler Infrastructure**

Complete API documentation, optimization guide, and end-to-end compilation flows for building AIR platforms targeting AMD AI Engine (AIE) devices — including Versal™ and Ryzen™ AI NPUs.

- **Languages:** C++ / MLIR / Python
- **License:** MIT License
- **Target Devices:** NPU1/NPU2
- **Platform:** VCK5000 Platform

## Table of Contents

- [Architecture](#architecture)
- [Memory Model](#memory-model)
- [Hierarchy Operations](#hierarchy-operations)
- [Data Movement Operations](#data-movement-operations)
- [Channel Operations](#channel-operations)
- [Async Operations](#async-operations)
- [Types](#types)
- [Async Concurrency](#async-concurrency)
- [CDFG Passes](#cdfg-passes)
- [Transform Passes](#transform-passes)
- [Conversion Passes](#conversion-passes)
- [E2E Flow](#e2e-flow)
- [aircc.py Driver](#airccpy-driver)
- [Examples](#examples)
- [Tools](#tools)

---

## Architecture

MLIR-AIR is a multi-level compiler stack that progressively lowers high-level parallel programs to bare-metal AIE core configurations. It sits above the AIE dialect (mlir-aie) and below the user's ML framework or linalg operations.

### Key Components

**AIR Dialect**
- High-level spatial IR: `air.launch`, `air.segment`, `air.herd`
- Captures the 3-level hierarchy of compute and memory

**Transform Passes**
- Optimizations: async dependency extraction, ping-pong buffering, channel fusion, broadcast detection, loop tiling, pipelining

**Conversion Passes**
- Lowering: AIR → AIE dialect (tile/DMA/lock allocation), AIR → AIRRt (host control), AIRRt → NPU instructions

**Runtime (AIRRt)**
- Runtime metadata dialect + host library
- Describes segments, herds, DMA allocations for the AIR runtime

**Python Bindings**
- Full Python API via `air.compiler.aircc` and MLIR Python bindings for programmatic IR construction

**air-runner**
- Performance simulator that models concurrent AIE execution without requiring physical hardware

### Compilation Stack

```
Linalg / SCF / Tensor
    ↓
AIR Dialect
    ↓
Optimize (CDFG)
    ↓
air-to-aie
    ↓
AIE Dialect
    ↓
NPU Binary + Host Lib
```

---

## Memory Model

The AIR dialect has an explicit three-level memory hierarchy. Every `memref` in AIR code carries a memory space attribute that determines which level it occupies.

### Three-Level Memory Hierarchy

**L3 — DDR / External DRAM**
- Host-accessible
- Shared across all partitions
- Capacity: GBs
- Memory Space Attr: `(none)`
- Accessible By: Host CPU, Shim DMA

**L2 — Memtile / Shared SRAM**
- Segment-local scratchpad
- Capacity: KBs–MBs
- Memory Space Attr: `1`
- Accessible By: Segment-wide DMAs
- Example: `memref<..., 1>`

**L1 — AIE Tile Memory**
- Core-local scratchpad
- Capacity: KBs only
- Ultra-low latency for SIMD ops
- Memory Space Attr: `2`
- Example: `memref<..., 2>`

> ℹ️ **Note:** Data movement between levels always goes through DMA operations (`air.dma_memcpy_nd` or `air.channel.put/get`). Direct load/store across levels is not legal in the AIR dialect.

---

## Hierarchy Operations

AIR programs are structured by three nested scope operations that mirror the physical hierarchy of the device.

### air.launch

**Host-level parallel launch scope**

Groups segments and L3 allocations that must be co-resident on the device. Optionally defines a parallel iteration space (host-side parallelism, not mapped to tiles).

```mlir
// Syntax
air.launch @name [%async_deps] (%iv0, %iv1) in (%sz0=%bound0, ...)
    args(%arg0=%host_buf) : memref<...>
    attributes {id = 1 : i32} {
  // body: contains air.segment or direct computation
  air.launch_terminator
}

// Async variant (returns a token for dependency tracking)
%tok = air.launch async [%dep] (%iv) in (...) args(...) {
  ...
  air.launch_terminator
}
```

| Attribute | Type | Description |
|-----------|------|-------------|
| async_token | `!air.async.token` | Optional. Return token for dependency chaining. |
| async_dependencies | `variadic<!air.async.token>` | List of tokens this op must wait on before starting. |
| sizes | `variadic<index>` | Upper bounds of the parallel iteration space. |
| ids | `variadic<index>` | Block arguments for iteration indices. |
| operands | variadic | Values passed into the launch body from outside scope. |

### air.segment

**Physically contiguous AIE core grouping**

Represents a physically contiguous grouping of AIE cores, L1 and L2 memory, and controllers sufficient for the enclosed herds. Can define a spatial "stamp-out" iteration space that multiplies physical resources.

```mlir
air.segment @seg_name [%deps] (%iv0, %iv1) in (%sz0, %sz1)
    args(%a=%buf) : memref<256xf32, 1>
    attributes {x_loc = 0 : i32, y_loc = 2 : i32,
                 x_size = 4 : i32, y_size = 4 : i32} {
  // Can alloc L2 memory here
  %l2_buf = memref.alloc() : memref<128xf32, 1>
  // Contains air.herd ops
  air.segment_terminator
}
```

| Placement Attribute | Description |
|-------------------|-------------|
| x_loc, y_loc | Top-left corner of segment on device array (column, row). |
| x_size, y_size | Span of the segment in device grid coordinates. |

### air.herd

**Individual AIE core execution context**

The finest-grain scope. Each herd represents a single AIE core (or small cluster) executing the same function body in SIMD fashion over distributed data.

```mlir
air.herd @herd_name [%deps] (%bx, %by) in (%bsx=%bx_size, %bsy=%by_size)
    args(%b=%l2_buf) : memref<128xf32, 1> {
  // Allocate L1 per-core memory
  %l1_buf = memref.alloc() : memref<32xf32, 2>
  
  // Core compute
  scf.for %i = 0 to 32 {
    %val = memref.load %b[%i] : memref<128xf32, 1>
    memref.store %val, %l1_buf[%i] : memref<32xf32, 2>
  }
  
  air.herd_terminator
}
```

| Attribute | Type | Description |
|-----------|------|-------------|
| block_args | index | Loop variables for herd iteration (typically `%bx, %by`). |
| block_sizes | index | Dimensions of the herd grid. |
| async_token | `!air.async.token` | Optional return token. |

---

## Data Movement Operations

### air.dma_memcpy_nd

**N-dimensional strided DMA copy**

The primary data movement primitive. Copies data between memory levels with optional strides and blocking structure.

```mlir
// Sync variant
air.dma_memcpy_nd (%dst[%dst_off0, %dst_off1] : memref<128x32xf32, 1>,
                    %src[%src_off0, %src_off1] : memref<256x64xf32>)
    attributes {
      id = 1 : i32,
      offset = [0, 0] : i64,
      sizes = [128, 32] : i64,
      strides = [2, 1] : i64
    }

// Async variant
%tok = air.dma_memcpy_nd async [%dep] (...)
```

| Parameter | Description |
|-----------|-------------|
| destination | Output memref with base offset |
| source | Input memref with base offset |
| sizes | Copy size per dimension |
| strides | Stride per dimension (defaults to 1) |
| id | Unique identifier (useful for debugging) |

---

## Channel Operations

### air.channel

Typed, point-to-point channels for communication between herds or segments.

#### air.channel.create

```mlir
%ch:2 = air.channel.create : !air.channel<index>
```

#### air.channel.put / air.channel.get

```mlir
// Producer (put)
air.channel.put %ch[%idx] (%data : f32)

// Consumer (get)
%val = air.channel.get %ch[%idx] : f32
```

---

## Async Operations

All major AIR operations support an `async` variant that returns an `!air.async.token` for explicit dependency management.

```mlir
// Launch async and get token
%tok0 = air.launch async [%dep_tok] (...) {
  ...
  air.launch_terminator
}

// DMA async
%tok1 = air.dma_memcpy_nd async [%tok0] (...)

// Chain dependencies
%tok2 = air.segment async [%tok1] (...) {
  ...
  air.segment_terminator
}
```

---

## Types

### air.async.token

Represents a token for async operation dependency tracking.

```mlir
!air.async.token
```

### air.channel

Typed channel for inter-herd communication.

```mlir
!air.channel<index>      // Indexed channel
!air.channel<f32>        // Value channel
```

---

## Async Concurrency

Use async operations to overlap compute and data movement.

### Example: Ping-Pong Buffering

```mlir
// First DMA
%tok0 = air.dma_memcpy_nd async [%empty] (%buf0[0, 0] : ..., %src[0, 0] : ...)

// Compute on buffer 1 (while buf0 is being fetched)
%tok1 = air.herd async [%tok0] (...) {
  // Process %buf1
  air.herd_terminator
}

// Next DMA (while compute is running)
%tok2 = air.dma_memcpy_nd async [%tok1] (%buf1[0, 0] : ..., %src[128, 0] : ...)
```

---

## CDFG Passes

Control and Data Flow Graph analysis passes for dependency optimization.

- **AsyncDependencyAnalysis:** Extracts implicit dependencies between operations
- **ChannelFusion:** Merges adjacent channel operations
- **BroadcastDetection:** Identifies broadcast patterns for optimization

---

## Transform Passes

High-level program transformation and optimization.

- **LoopTiling:** Divides loops into manageable tiles
- **Pipelining:** Overlaps loop iterations for throughput
- **MemrefToTile:** Maps memrefs to physical tile hierarchy

---

## Conversion Passes

Lower-level conversions to backend dialects.

### air-to-aie

Converts AIR operations to AIE dialect with explicit DMA and lock allocation.

### air-to-airrt

Converts AIR to AIRRt (runtime) dialect for host-side control generation.

### airrt-to-npu

Final conversion to NPU instruction sequences.

---

## E2E Flow

End-to-end compilation from high-level IR to executable binary.

```
Input (Linalg/SCF/Tensor)
    ↓ (Lowering passes)
AIR IR
    ↓ (Transform passes)
Optimized AIR
    ↓ (air-to-aie)
AIE IR
    ↓ (aie-opt)
Scheduled AIE
    ↓ (aie-translate)
NPU Binary
    + Host Control Library
```

---

## aircc.py Driver

Main compilation driver for AIR projects.

```bash
python3 -m air.compiler.aircc \
    --input my_program.mlir \
    --output-dir build/ \
    --target aie2 \
    --num-cores 16
```

**Key Options:**
- `--input`: Source MLIR file
- `--output-dir`: Output directory for binaries
- `--target`: Target device (aie2, npu1, npu2)
- `--num-cores`: Parallel compile threads

---

## Examples

### Basic Memcpy

Copy data from L3 (host) to L2 (segment) to L1 (core).

```mlir
air.launch @copy_kernel {
  %l3_buf = memref.alloc() : memref<256xf32>
  
air.segment @seg {
    %l2_buf = memref.alloc() : memref<256xf32, 1>
    
    // Copy L3 → L2
    air.dma_memcpy_nd (%l2_buf[0] : memref<256xf32, 1>,
                       %l3_buf[0] : memref<256xf32>)
    
air.herd @h {
      %l1_buf = memref.alloc() : memref<32xf32, 2>
      
      // Copy L2 → L1
      air.dma_memcpy_nd (%l1_buf[0] : memref<32xf32, 2>,
                         %l2_buf[0] : memref<256xf32, 1>)
      
air.herd_terminator
    }
    
air.segment_terminator
  }
  
air.launch_terminator
}
```

### Matrix Multiply

Tiled matrix multiplication with async data movement.

```mlir
air.launch @matmul (%M : index, %N : index, %K : index) {
  // Allocate L3 buffers
  %A = memref.alloc(%M, %K) : memref<?x?xf32>
  %B = memref.alloc(%K, %N) : memref<?x?xf32>
  %C = memref.alloc(%M, %N) : memref<?x?xf32>
  
air.segment @seg {
    // Allocate L2 tile buffers
    %A_tile = memref.alloc() : memref<16x16xf32, 1>
    %B_tile = memref.alloc() : memref<16x16xf32, 1>
    %C_tile = memref.alloc() : memref<16x16xf32, 1>
    
    // Tile loops
    scf.for %i = 0 to %M step 16 {
      scf.for %j = 0 to %N step 16 {
        scf.for %k = 0 to %K step 16 {
          // Async DMAs
          %tok_a = air.dma_memcpy_nd async [%empty] (%A_tile[0, 0], %A[%i, %k])
          %tok_b = air.dma_memcpy_nd async [%empty] (%B_tile[0, 0], %B[%k, %j])
          
          // Compute
          %tok_c = air.herd async [%tok_a, %tok_b] {
            // Tile gemm computation
            air.herd_terminator
          }
          
          // Writeback
          air.dma_memcpy_nd async [%tok_c] (%C[%i, %j], %C_tile[0, 0])
        }
      }
    }
    
air.segment_terminator
  }
  
air.launch_terminator
}
```

### Pipelined Tile

Overlap computation with data movement using async tokens.

```mlir
air.herd @pipeline {
  %buf0 = memref.alloc() : memref<32xf32, 2>
  %buf1 = memref.alloc() : memref<32xf32, 2>
  
  // Stage 0: Fetch first tile
  %tok_fetch_0 = air.dma_memcpy_nd async [%empty] (%buf0[0], %src[0])
  
  scf.for %i = 0 to 100 step 1 {
    // Compute on previous tile
    %tok_compute = air.do_something async [%tok_fetch_0] (%buf0)
    
    // Fetch next tile (async, overlaps with compute)
    %next_idx = arith.addi %i, 1 : index
    %tok_fetch_1 = air.dma_memcpy_nd async [%empty] (%buf1[0], %src[%next_idx])
    
    // Swap buffers for next iteration
    ...
  }
  
air.herd_terminator
}
```

### Broadcasting

Broadcast a single value to multiple consumers.

```mlir
air.herd @broadcast {
  %ch:2 = air.channel.create : !air.channel<f32>
  
  // Producer
  scf.parallel (%i) = (0) to (16) {
    %val = ... : f32
    air.channel.put %ch[%i] (%val : f32)
  }
  
  // Consumers
  scf.parallel (%j) = (0) to (16) {
    %data = air.channel.get %ch[%j] : f32
    // Use data
  }
  
air.herd_terminator
}
```

---

## Tools

### AIR Runner

Performance simulator for AIR programs without physical hardware.

```bash
air-runner --mlir-file program.mlir --timeline timeline.txt
```

### AIRRt Dialect

Runtime metadata dialect describing segment and herd allocation for the AIR runtime system.

### Python API

Full programmatic IR construction via MLIR Python bindings.

```python
from air.compiler import aircc
from mlir import ir

# Create context and module
ctx = ir.Context()
module = ir.Module.create()

# Build AIR program programmatically
with ir.InsertionPoint(module.body):
    launch_op = aircc.create_air_launch(...)

# Compile
aircc.compile(module, output_dir="build/")
```

---

**Last Updated:** 2026-02-19
**Repository:** Sameeranjoshi/mlir-air
**License:** MIT