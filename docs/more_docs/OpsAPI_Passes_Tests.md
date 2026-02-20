# MLIR-AIR API Reference
### Operations · Passes · Examples · Optimization Diffs · Test Suite

> **Naming convention used throughout:** `air.mnemonic (air::OpNameOp)` — the MLIR text form followed by the C++ class in parentheses.

---

## Table of Contents

1. [Hardware Mental Model](#hardware-mental-model)
2. [Memory Spaces](#memory-spaces)
3. [Hierarchy Operations](#hierarchy-operations)
   - [air.launch (LaunchOp)](#airlaunch--airlaunchop)
   - [air.segment (SegmentOp)](#airsegment--airsegmentop)
   - [air.herd (HerdOp)](#airherd--airherdop)
4. [Data Movement Operations](#data-movement-operations)
   - [air.dma_memcpy_nd (DmaMemcpyNdOp)](#airdma_memcpy_nd--airdmamemcpyndop)
   - [air.channel (ChannelOp)](#airchannel--airchannelop)
   - [air.channel.put (ChannelPutOp)](#airchannelput--airchannelputop)
   - [air.channel.get (ChannelGetOp)](#airchannelget--airchannelgetop)
5. [Async Operations](#async-operations)
   - [air.execute (ExecuteOp)](#airexecute--airexecuteop)
   - [air.wait\_all (WaitAllOp)](#airwait_all--airwaitallop)
   - [!air.async.token (AsyncTokenType)](#airasynctoken--airasyntokentype)
6. [AIRRt Dialect — Runtime Metadata](#airrt-dialect--runtime-metadata)
7. [Optimization Passes — Before & After](#optimization-passes--before--after)
   - [-air-dependency](#-air-dependency)
   - [-air-dependency-canonicalize](#-air-dependency-canonicalize)
   - [-air-dependency-schedule-opt](#-air-dependency-schedule-opt)
   - [-air-ping-pong-transform](#-air-ping-pong-transform)
   - [-air-fuse-channels](#-air-fuse-channels)
   - [-air-specialize-channel-wrap-and-stride](#-air-specialize-channel-wrap-and-stride)
   - [-air-place-herds](#-air-place-herds)
8. [Conversion Passes](#conversion-passes)
   - [-air-to-aie](#-air-to-aie)
   - [-air-to-std](#-air-to-std)
   - [-airrt-to-npu](#-airrt-to-npu)
   - [Structural Passes](#structural-passes)
9. [Progressive Test Suite](#progressive-test-suite)
   - [T01 — Hello Herd (Beginner)](#t01--hello-herd)
   - [T02 — Strided 2D DMA](#t02--strided-2d-dma)
   - [T03 — Async Parallel Pipeline](#t03--async-parallel-pipeline)
   - [T04 — Channels + Ping-Pong](#t04--channels--ping-pong)
   - [T05 — 4×4 Tiled GEMM](#t05--44-tiled-gemm)
   - [T06 — Broadcast DMA (8×8 herd)](#t06--broadcast-dma-8x8-herd)
   - [T07 — Tiled 1D Convolution](#t07--tiled-1d-convolution)
   - [T08 — Full E2E GEMM Pipeline](#t08--full-e2e-gemm-pipeline)

---

## Hardware Mental Model

Before reading the API, internalize this physical picture. Every AIR abstraction maps directly to one hardware layer.

```
Device Physical Hierarchy:

  🌐  L3 — DDR / Host DRAM          GBs · slow · memref<Nxf32>
      └─  air.launch scope

      📦  L2 — Memtile / URAM        512KB–2MB · medium · memref<Nxf32, 1>
          └─  air.segment scope

          🔲  Tile[0,0] L1 32KB       memref<Nxf32, 2>
          🔲  Tile[1,0] L1 32KB  ──── air.herd scope
          🔲  Tile[2,0] L1 32KB       (body cloned to every tile)
          🔲  Tile[3,0] L1 32KB
```

**Key rule:** You cannot load/store directly across memory levels. All cross-level transfers must use `air.dma_memcpy_nd` or `air.channel.put/get`.

---

## Memory Spaces

| Level | memref syntax | Hardware | Size | Accessible by |
|-------|--------------|----------|------|---------------|
| L3 | `memref<Nxf32>` | DDR / External DRAM | GBs | Host CPU, Shim DMA |
| L2 | `memref<Nxf32, 1>` | Memtile SRAM / URAM | 512KB–2MB | Segment-wide DMAs |
| L1 | `memref<Nxf32, 2>` | AIE tile local memory | 32–128KB | One core + local DMA |

> **Tip:** The most common beginner mistake is forgetting the memory space attribute. Always annotate buffers with `, 1` or `, 2`.

---

## Hierarchy Operations

### `air.launch`  (air::LaunchOp)

**Brief:** Host-level scope; the "job submission" boundary for a device invocation.

**Traits:** `IsolatedFromAbove`, `AffineScope`, `SingleBlock`, `SingleBlockImplicitTerminator<LaunchTerminatorOp>`  
**Interfaces:** `air::AsyncOpInterface`, `air::HierarchyInterface`

#### Synopsis

```mlir
// Scalar (single) launch
air.launch args(%a = %val) : memref<Nxf32> {
  // ... contains air.segment ops ...
  air.launch_terminator
}

// Parallel launch — %ix iterates 0..N-1 on the HOST (not AIE tiles)
%tok = air.launch async [%dep] (%ix) in (%sz = %N)
    args(%a = %buf) : memref<512xf32> {
  // body
  air.launch_terminator
}
```

#### Human explanation

Think of `air.launch` as "calling the GPU kernel". Everything inside is one device invocation. The optional iteration space is host-side batching (like calling the same work N times in a loop) — **not** AIE tile parallelism. Tile parallelism lives inside `air.herd`. All live-in values must be passed explicitly through `args()` because the body is `IsolatedFromAbove`.

#### Example 1 — Minimal scalar launch

```mlir
func.func @example(%x: memref<1024xf32>, %y: memref<1024xf32>) {
  air.launch args(%xa = %x, %ya = %y) : memref<1024xf32>, memref<1024xf32> {
    // ... air.segment lives here ...
    air.launch_terminator
  }
  return
}
```

#### Example 2 — Parallel launch: host-side batching over 4 problem tiles

```mlir
%c4 = arith.constant 4 : index
// %bid ∈ {0, 1, 2, 3} — each iteration processes a 256-element slice
air.launch (%bid) in (%nb = %c4) args(%buf = %global) : memref<1024xf32> {
  %off = affine.apply affine_map<(d0) -> (d0 * 256)>(%bid)
  // air.segment uses %off to select its slice ...
  air.launch_terminator
}
```

---

### `air.segment`  (air::SegmentOp)

**Brief:** Reserves a physically contiguous rectangle of AIE tiles, L2 memory, and the DMA controller.

**Traits:** `IsolatedFromAbove`, `AffineScope`, `SingleBlock`  
**Interfaces:** `air::AsyncOpInterface`, `air::HierarchyInterface`

#### Synopsis

```mlir
air.segment @name args(%a = %v) : type
    attributes { x_loc = 0 : i32, y_loc = 2 : i32,
                 x_size = 4 : i32, y_size = 2 : i32 } {
  %l2 = memref.alloc() : memref<256xf32, 1>   // L2 allocation
  // ... air.herd ops ...
  air.segment_terminator
}
```

#### Placement attributes

| Attribute | Type | Meaning |
|-----------|------|---------|
| `x_loc` | `I32Attr` | Column of top-left corner on the device tile grid |
| `y_loc` | `I32Attr` | Row of top-left corner |
| `x_size` | `I32Attr` | Width in columns (tile columns owned) |
| `y_size` | `I32Attr` | Height in rows |

#### Human explanation

Think of `air.segment` as a "partition lease". It reserves a rectangular block of tiles, the shared L2 memtile, and the loading dock (shim DMA). `‑air-to-aie` converts this into one `aie.device` module with all physical resources allocated. You can allocate L2 buffers directly inside the segment body with `memref.alloc() : memref<N, 1>`.

#### Example — Segment staging L3 data into L2, then feeding a herd

```mlir
air.segment @my_seg args(%inp = %l3_buf) : memref<1024xf32>
    attributes { x_loc = 0 : i32, y_loc = 2 : i32,
                 x_size = 4 : i32, y_size = 1 : i32 } {

  // Stage into L2 for lower-latency tile access
  %l2 = memref.alloc() : memref<1024xf32, 1>
  air.dma_memcpy_nd (%l2[] [] [], %inp[] [] []) {id = 1 : i32}
      : (memref<1024xf32, 1>, memref<1024xf32>)

  %c4   = arith.constant 4   : index
  %c256 = arith.constant 256 : index
  %c1   = arith.constant 1   : index

  air.herd @worker tile(%tx, %ty) in (%sx = %c4, %sy = %c1)
      args(%l2a = %l2) : memref<1024xf32, 1> {
    %l1  = memref.alloc() : memref<256xf32, 2>
    %off = affine.apply affine_map<(d0) -> (d0 * 256)>(%tx)
    air.dma_memcpy_nd (%l1[] [] [], %l2a[%off][%c256][%c1]) {id = 2 : i32}
        : (memref<256xf32, 2>, memref<1024xf32, 1>)
    air.herd_terminator
  }
  air.segment_terminator
}
```

---

### `air.herd`  (air::HerdOp)

**Brief:** Declares a 1D or 2D array of AIE tile instances. The body is cloned to every core in the array.

**Traits:** `IsolatedFromAbove`, `AffineScope`, `SingleBlock`  
**Interfaces:** `air::AsyncOpInterface`, `air::HierarchyInterface`

#### Synopsis

```mlir
air.herd @name tile(%tx, %ty) in (%sx = %cols, %sy = %rows)
    args(%a = %buf) : type
    attributes { x_loc = 1 : i32, y_loc = 1 : i32, link_with = "kernel.o" } {

  // %tx ∈ [0, cols)   %ty ∈ [0, rows)
  %l1 = memref.alloc() : memref<32xf32, 2>    // L1 per-tile buffer

  // affine.if on tile IDs is folded at lowering time
  affine.if #is_col0[%tx, %ty] { ... } else { ... }

  air.herd_terminator
}
```

#### Human explanation — body is cloned; tile IDs become compile-time constants

When `‑air-to-aie` runs, the herd body is copied once per tile position. At that point `%tx` and `%ty` are substituted with known integer constants (0, 1, 2, 3…). Any `affine.if` on those IDs gets **statically folded** — different tiles receive specialised code paths with zero runtime branching overhead.

#### Example 1 — 2×2 herd: left vs right tiles load different halves

```mlir
#is_left = affine_set<(d0, d1) : (d0 == 0)>
%c2 = arith.constant 2 : index

air.herd @quad tile(%tx, %ty) in (%sx = %c2, %sy = %c2)
    args(%data = %l3_src) : memref<256xf32> {
  %l1 = memref.alloc() : memref<128xf32, 2>

  // After -air-to-aie: tile[0,*] always uses offset=0; tile[1,*] always uses 128
  affine.if #is_left[%tx, %ty] {
    air.dma_memcpy_nd (%l1[] [] [], %data[0][128][1])  {id = 1 : i32} : (...)
  } else {
    air.dma_memcpy_nd (%l1[] [] [], %data[128][128][1]) {id = 2 : i32} : (...)
  }
  air.herd_terminator
}
```

#### Example 2 — Row-based task assignment in a 4×2 herd

```mlir
// %ty selects which row of output this tile computes
#map_row = affine_map<(d0) -> (d0 * 64)>   // ty * 64
%c4 = arith.constant 4 : index
%c2 = arith.constant 2 : index

air.herd @rowwise tile(%tx, %ty) in (%sx = %c4, %sy = %c2)
    args(%out = %result) : memref<128x256xf32> {
  %row_off = affine.apply #map_row(%ty)   // row start = ty * 64
  %col_off = affine.apply affine_map<(d0) -> (d0 * 64)>(%tx)
  // Each tile owns result[ty*64 : ty*64+64, tx*64 : tx*64+64]
  %l1 = memref.alloc() : memref<64x64xf32, 2>
  // ... compute and store ...
  air.herd_terminator
}
```

---

## Data Movement Operations

### `air.dma_memcpy_nd`  (air::DmaMemcpyNdOp)

**Brief:** N-dimensional strided DMA transfer between memory levels. Direction is always `dst ← src`.

**Interfaces:** `air::AsyncOpInterface`  
**Required attribute:** `id` — unique `i32` per herd; used by dependency analysis and DMA allocation.

#### Synopsis

```mlir
// Full form (up to 4D)
air.dma_memcpy_nd
  (%dst[off3,off2,off1,off0][sz3,sz2,sz1,sz0][str3,str2,str1],
   %src[off3,off2,off1,off0][sz3,sz2,sz1,sz0][str3,str2,str1])
  {id = N : i32} : (memref<...>, memref<...>)

// Shorthand: empty [] means "whole buffer, contiguous"
air.dma_memcpy_nd (%dst[] [] [], %src[] [] []) {id = 1 : i32} : (...)

// Async form — returns !air.async.token
%tok = air.dma_memcpy_nd async [%dep] (...) {id = 2 : i32} : (...)
```

#### Parameter table

| Name | Dims | Meaning |
|------|------|---------|
| `dst_offsets` | 0–4 | Starting element per dimension in destination. Empty = 0. |
| `dst_sizes` | 0–4 | Number of elements to access per dimension. Empty = full extent. |
| `dst_strides` | 0–3 | Step between consecutive accesses. Innermost stride is always 1 (implicit). |
| `src_offsets` | 0–4 | Starting element per dimension in source. |
| `src_sizes` | 0–4 | Elements to read per dimension. |
| `src_strides` | 0–3 | Source stride. |
| `id` | — | **Required.** Unique `i32` within the herd. |
| `broadcast_pattern` | — | Optional affine set; set by `-air-dependency-schedule-opt` when broadcast is detected. |

#### Human explanation — reading `[offsets][sizes][strides]`

The three brackets describe a nested loop the DMA engine executes as a buffer-derange(16):          # sizes[0] = 16 rows
    for j in range(32):      # sizes[1] = 32 cols
        dst[i*32 + j] = src[(row_off + i)*128 + j]
        #                     ↑ strides[0]=128   ↑ strides[1]=1 (implicit)
```

#### Example 1 — Full L3 → L1 copy (whole buffer)

```mlir
%l1 = memref.alloc() : memref<1024xi32, 2>
air.dma_memcpy_nd (%l1[] [] [], %l3_vec[] [] [])
    {id = 1 : i32} : (memref<1024xi32, 2>, memref<1024xi32>)
```

#### Example 2 — Strided 2D tile extraction from a large matrix

```mlir
// Extract a 16×32 sub-tile starting at (row_off, 0) from a 128×128 matrix.
// Row stride = 128 (row-major: each row is 128 elements wide).
%tile = memref.alloc() : memref<16x32xf32, 2>
%c16  = arith.constant 16  : index
%c32  = arith.constant 32  : index
%c128 = arith.constant 128 : index
%c1   = arith.constant 1   : index
%c0   = arith.constant 0   : index

air.dma_memcpy_nd
  (%tile[] [] [],
   %matrix[%row_off, %c0] [%c16, %c32] [%c128, %c1])
  {id = 2 : i32}
  : (memref<16x32xf32, 2>, memref<128x128xf32>)
```

#### Example 3 — Async parallel loads (A and B fire simultaneously)

```mlir
// tA and tB have NO dependency on each other → both fire at the same time
%tA = air.dma_memcpy_nd async [%allocA]
    (%l1_A[] [] [], %l3_A[%row, %k][%M, %K][%lda, %c1])
    {id = 3 : i32} : (memref<MxKxf32, 2>, memref<MxKxf32>)

%tB = air.dma_memcpy_nd async [%allocB]
    (%l1_B[] [] [], %l3_B[%k, %col][%K, %N][%ldb, %c1])
    {id = 4 : i32} : (memref<KxNxf32, 2>, memref<KxNxf32>)

// Compute waits for BOTH loads via a barrier join
%both = air.wait_all async [%tA, %tB] {id = 5 : i32}
%mm   = air.execute [%both] { linalg.matmul ... }
```

---

### `air.channel`  (air::ChannelOp)

**Brief:** Module-level producer-consumer data stream declaration.

```mlir
// 1×1: single producer, single consumer
air.channel @my_chan [1, 1]

// 4×2: one sub-channel per tile in a 4-col × 2-row herd
air.channel @weights [4, 2]
```

#### Channels vs DMA — when to use which

| | `air.dma_memcpy_nd` | `air.channel.put/get` |
|--|--|--|
| Model | Explicit point-to-point copy | Implicit FIFO stream |
| Best for | One-shot loads / stores | Pipelined streaming loops |
| Ping-pong | Manual wiring | `-air-ping-pong-transform` automates it |
| Optimization | broadcast detection | channel fusion, wrap-stride |

---

### `air.channel.put`  (air::ChannelPutOp)

**Brief:** Send data into a channel (producer side; typically in segment body or outer loop).

```mlir
// Synchronous
air.channel.put @chan[%col, %c0]
    (%src[%offset][%sz][%c1]) : (memref<1024xf32>)

// Async
%tok = air.channel.put async [%dep] @chan[%ix, %iy]
    (%l2_buf[%off][%sz][%str]) : (memref<Nxf32, 1>)
```

---

### `air.channel.get`  (air::ChannelGetOp)

**Brief:** Receive data from a channel (consumer side; typically inside `air.herd`).

```mlir
// Synchronous — tile [%tx, %ty] receives into its L1 buffer
air.channel.get @chan[%tx, %ty] (%l1_dst[] [] []) : (memref<256xf32, 2>)

// Async
%tok = air.channel.get async [%prev] @chan[%tx, %ty]
    (%l1_dst[] [] []) : (memref<256xf32, 2>)
```

#### Full example — feeder loop in segment body + consumer herd

```mlir
// Declare at module level: 4 sub-channels for a 4×1 herd
air.channel @feed [4, 1]

air.segment @seg args(%inp = %l3) : memref<1024xf32> {
  %c4   = arith.constant 4   : index
  %c256 = arith.constant 256 : index
  %c0   = arith.constant 0   : index
  %c1   = arith.constant 1   : index

  // PRODUCER: segment body pushes 256-element chunks to each tile
  scf.for %col = %c0 to %c4 step %c1 {
    %off = arith.muli %col, %c256 : index
    air.channel.put @feed[%col, %c0]
        (%inp[%off][%c256][%c1]) : (memref<1024xf32>)
  }

  // CONSUMER HERD: each tile receives its own 256 elements
  air.herd @worker tile(%tx, %ty) in (%c4, %c1) {
    %l1 = memref.alloc() : memref<256xf32, 2>
    air.channel.get @feed[%tx, %ty] (%l1[] [] []) : (memref<256xf32, 2>)
    // ... compute on %l1 ...
    air.herd_terminator
  }
  air.segment_terminator
}
```

---

## Async Operations

### `air.execute`  (air::ExecuteOp)

**Brief:** Wraps synchronous ops in an async context so the CDFG passes can reason about their ordering.

#### Synopsis

```mlir
// Compute wrapper (no yielded value)
%tok = air.execute [%depA, %depB] {
  linalg.matmul ins(%A, %B : ...) outs(%C : ...)
} {id = 5 : i32}

// Alloc wrapper — yields a value from inside the region
%tok2, %buf = air.execute [%dep] -> (memref<32xf32, 2>) {
  %alloc = memref.alloc() : memref<32xf32, 2>
  air.execute_terminator %alloc : memref<32xf32, 2>
} {id = 6 : i32}
```

#### Human explanation

Standard ops like `linalg.matmul` or `memref.alloc` are synchronous — they have no token interface. `air.execute` is a thin async wrapper giving them one so that `-air-dependency` can wire tokens in and out. In practice `-air-dependency` inserts `air.execute` wrappers automatically; you only write them manually when you need custom scheduling order.

---

### `air.wait_all`  (air::WaitAllOp)

**Brief:** Barrier join — merge N tokens into one. Nothing downstream starts until all inputs complete.

```mlir
// Async form: produces a new token
%barrier = air.wait_all async [%t1, %t2, %t3] {id = 7 : i32}

// Synchronous form: no output token (used at loop boundaries)
air.wait_all [%t4, %t5]
```

#### Example — synchronise three parallel loads before compute

```mlir
%tA = air.dma_memcpy_nd async (...) {id = 1 : i32} : (...)
%tB = air.dma_memcpy_nd async (...) {id = 2 : i32} : (...)
%tC = air.dma_memcpy_nd async (...) {id = 3 : i32} : (...)

// All three must finish before matmul can start
%all_loaded = air.wait_all async [%tA, %tB, %tC] {id = 4 : i32}

%compute = air.execute [%all_loaded] {
  linalg.matmul ins(%l1_A, %l1_B : ...) outs(%l1_C : ...)
} {id = 5 : i32}
```

---

### `!air.async.token`  (air::AsyncTokenType)

**Brief:** An SSA value representing "this async operation will complete in the future". The edges of the CDFG.

```mlir
!air.async.token
```

**Is:** A compile-time SSA value representing a dependency edge. Consume it as an input dep to say "wait for this operation before starting."

**Is not:** A runtime object. By the time `-air-to-aie` finishes, every token is eliminated and replaced with AIE `lock acquire/release` instructions. Tokens exist only in the IR during compilation.

---

## AIRRt Dialect — Runtime Metadata

These operations are generated by `-air-to-aie` and consumed by `-airrt-to-llvm` / `-airrt-to-npu`. You do not write them by hand.

### `airrt.module_metadata`  (airrt::ModuleMetadataOp)

Top-level container; one per module. Contains `airrt.segment_metadata` ops.

```mlir
airrt.module_metadata {
  // ... airrt.segment_metadata ops ...
}
```

### `airrt.segment_metadata`  (airrt::SegmentMetadataOp)

Runtime metadata for one AIE segment.

```mlir
airrt.segment_metadata attributes { sym_name = "seg_0" } {
  // ... airrt.herd_metadata ops ...
}
```

### `airrt.herd_metadata`  (airrt::HerdMetadataOp)

Per-herd DMA allocation table. The `dma_allocations` array maps DMA `id` values to physical tile/channel assignments.

```mlir
airrt.herd_metadata {
  sym_name = "herd_0",
  dma_allocations = [{id=1:i64, channel=2:i64, col=0:i64, row=0:i64, location=2:i64}]
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `id` | `i64` | Matches the `id` on the source `DmaMemcpyNdOp` |
| `channel` | `i64` | Allocated DMA channel number |
| `col` / `row` | `i64` | Physical tile coordinates |
| `location` | `i64` | 0 = shim, 1 = memtile, 2 = core |

### `airrt.dma_memcpy_nd`  (airrt::DmaMemcpyNdOp)

Host-side runtime DMA call with tile routing metadata. Generated by `-air-to-std`.

```mlir
airrt.dma_memcpy_nd(%id, %x, %y, %memref,
  [%off3,%off2,%off1,%off0], [%len3,%len2,%len1,%len0], [%str3,%str2,%str1])
  {metadata = @airMemcpyId1}
  : (i32, i64, i64, memref<Nxi32>, [i64 x4], [i64 x4], [i64 x3])
```

`%x` and `%y` are the target tile column/row for DMA routing. The `metadata` attribute references the `aie.shim_dma_allocation` symbol generated by `-air-to-aie`.

---

## Optimization Passes — Before & After

### `-air-dependency`

**(AIRDependencyPass)** — Auto-extract the CDFG by injecting async tokens based on data and loop dependencies.

> **Always run first.** All other dependency-related passes require the CDFG to be built.

#### Human explanation

Reads your synchronous code and asks: "which op reads what another op writes?" For every true read-after-write pair A→B it inserts an `!air.async.token` edge. Ops with no dependency on each other get *no* edge — the hardware runs them in parallel automatically.

#### Before / After — synchronous matmul loop → parallel async CDFG

```mlir
// ──────── BEFORE: all ops run serially ────────
scf.for %k = ... {
  %A = memref.alloc()          // step 1
  %B = memref.alloc()          // step 2
  air.dma_memcpy_nd  // load A  step 3
  air.dma_memcpy_nd  // load B  step 4   ← could overlap with step 3!
  linalg.matmul               // step 5
  air.dma_memcpy_nd  // store C step 6
}

// ──────── AFTER: A-load and B-load run in parallel ────────
scf.for ... iter_args(%loop_tok = %init) {
  %tAl, %A = air.execute { memref.alloc() }           // id=1
  %tBl, %B = air.execute { memref.alloc() }           // id=2
  // Both DMAs depend only on their respective allocs → parallel!
  %tA = air.dma_memcpy_nd async [%tAl, %loop_tok] (...) {id=3}
  %tB = air.dma_memcpy_nd async [%tBl, %loop_tok] (...) {id=4}
  %both = air.wait_all async [%tA, %tB] {id=5}
  %mm  = air.execute [%both] { linalg.matmul ... }    {id=6}
  %tSt = air.dma_memcpy_nd async [%mm] (...) {id=7}   // store C
  scf.yield %```

---

### `-air-dependency-canonicalize`

**(AIRDependencyCanonicalizeGraphPass)** — Remove redundant CDFG edges via transitive reduction.

#### Human explanation

If A→B and B→C, then the direct edge A→C is redundant (C will never start before B anyway). Removing it makes the IR smaller, later passes faster, and prevents emitting unnecessary lock operations in AIE code.

#### Before / After

```mlir
// ──────── BEFORE: redundant edge ────────
%dma = air.dma_memcpy_nd async [%loop_tok] (...)
// matmul lists %loop_tok directly AND via %dma (redundant!)
%mm  = air.execute [%loop_tok, %dma] { linalg.matmul ... }

// ──────── AFTER: minimal deps ────────
%dma = air.dma_memcpy_nd async [%loop_tok] (...)
// %loop_tok dropped — matmul waits for %dma, which already waits for %loop_tok
%mm  = air.execute [%dma] { linalg.matmul ... }
// Fewer lock operations in generated AIE code → shorter critical path
```

---

### `-air-dependency-schedule-opt`

**(AIRDependencyScheduleOptPass)** — Detect broadcast DMA patterns; annotate with `broadcast_pattern` affine set.

#### Human explanation

In an 8×8 matmul herd, all 8 tiles in the same row need the *same* row of matrix A. Without this pass: 8 separate DMA reads of identical bytes. With this pass: the compiler detects that tiles sharing `%ty` also share `%row_off`, annotates the DMA with `broadcast_pattern`, and `-air-to-aie` generates a single `aie.flow` fanout to all tiles in that row — **8× DDR bandwidth reduction**.

#### Before / After

```mlir
// ──────── BEFORE: 8 identical DDR loads ────────
%tA = air.dma_memcpy_nd async [...]
  (%l1_A[] [] [], %A[%row_off, %k][%M, %K][...])
  {id = 1 : i32}
// 8 column-tiles each issue this DMA independently → 8 DDR transactions

// ──────── AFTER: single multicast load ────────
// Auto-generated affine set: "same row (d0==s0), any column (d1: 0..7)"
#bcast_A = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 7 >= 0)>

%tA = air.dma_memcpy_nd async [...]
  (%l1_A[] [] [], %A[%row_off, %k][%M, %K][...])
  {id = 1 : i32, broadcast_pattern = #bcast_A}
// → -air-to-aie emits one aie.flow fanout to all 8 row-tiles
// → 8× reduction in DDR bandwidth for matrix-A reads
```

---

### `-air-ping-pong-transform`

**(AIRPingPongTransformPass)** — Double-buffering: overlap DMA load[i+1] with compute[i].

**Prerequisites:** Run `-air-label-scf-for-in-air-segment-ping-pong` first to identify target loops.  
**Option:** `-keep-memref-dealloc` — keep `memref.dealloc` on original buffers after transformation.

#### Human explanation

```
WITHOUT ping-pong (serial):        WITH ping-pong (overlapped):
  t=0  [LOAD  chunk 0]               t=0  [LOAD chunk 0]
  t=1  [COMPUTE chunk 0]             t=1  [LOAD chunk 1]  [COMPUTE chunk 0]
  t=2  [LOAD  chunk 1]               t=2  [LOAD chunk 2]  [COMPUTE chunk 1]
  t=3  [COMPUTE chunk 1]             t=3  [LOAD chunk 3]  [COMPUTE chunk 2]
  ↑ 2T per chunk                     ↑ T per chunk  (2× throughput)
```

The transform allocates two alternating buffers (ping / pong) and builds explicit async dependency chains so DMA always writes into the *idle* buffer while compute reads from the *active* one.

#### Before / After

```mlir
// ──────── BEFORE: single buffer, serial ────────
%buf = memref.alloc() : memref<256xf32, 2>
scf.for %i = ... {
  air.channel.get @chan[0, 0] (%buf[] [] [])
  linalg.generic { ... } ins(%buff32, 2>
// Explicit async dependency chains ensure ping ≠ pong is always respected
scf.for %i = ... iter_args(%prev_load = %init, %prev_compute = %init) {
  // Load ping WHILE compute runs on pong (from previous iteration)
  %get  = air.channel.get async [%prev_compute] @chan[0,0] (%ping[] [] [])
  %exec = air.execute async  [%prev_load]       { linalg.generic ... %pong ... }
  scf.yield %get, %exec
}
// DMA and compute engines run every cycle → 2× throughput
```

---

### `-air-fuse-channels`

**(AIRFuseChannelsPass)** — Merge compatible channel ops to reduce DMA transaction count and channel resource pressure.

**Option:** `-aggressive-mode L1|L2|L3` — enables time-multiplexed fusion across levels.

#### Before / After

```mlir
// ──────── BEFORE: 2 channel transactions per iteration ────────
air.channel @chA [1, 1]
air.channel @chB [1, 1]

scf.for %i = ... {
  air.channel.get @chA[0, 0] (%bufA[] [] [])   // DMA setup #1
  air.channel.get @chB[0, 0] (%bufB[] [] [])   // DMA setup #2
}

// ──────── AFTER: 1 fused transaction ────────
air.channel @chFused [1, 1]   // chA and chB merged into one channel

scf.for %i = ... {
  air.channel.get @chFused[0, 0] (%combined_buf[] [] [])   // single DMA setup
  // Less overhead, fewer channel resources consumed
}
```

---

### `-air-specialize-channel-wrap-and-stride`

**(AIRSpecializeChannelWrapAndStridePass)** — Exploit hardware DMA wrap-and-stride mode to collapse multi-BD chains into a single BD.

#### Human explanation

AIE DMA hardware can encode "repeat this transfer with a stride pattern" natively in a single Buffer Descriptor. Without this pass a 2D tiled access pattern over a loop of N might generate N separate BDs. After this pass it becomes 1 BD with hardware wrap counts — reducing DMA setup latency from microseconds to nanoseconds.

#### Before / After

```mlir
// ──────── BEFORE: loop of 4 separate BDs ────────
scf.for %j = %c0 to %c4 step %c1 {
  %off = arith.muli %j, %c64 : index
  air.channel.get @chan[0, 0] (%buf[%off][%c64][%c1]) : (memref<256xf32, 2>)
  // 4 DMA setup calls per iteration
}

// ──────── AFTER: 1 BD, hardware wraps 4× internally ────────
// wrap=4, stride=64 encoded in a single hardware Buffer Descriptor
air.channel.get @chan[0, 0] (%buf[0, 0][4, 64][64]) : (memref<256xf32, 2>)
// Single DMA setup — hardware loops 4× natively
```

---

### `-air-place-herds`

**(AIRPlaceHerdsPass)** — Assign physical `x_loc`/`y_loc` tile coordinates to each `air.herd`.

**Options:**

| Option | Description |
|--------|-------------|
| `-num-rows <n>` | Total tile rows available in this segment |
| `-num-cols <n>` | Total tile columns available |
| `-anchor-point <r,c>` | Grid origin for placement (default 0,0) |

#### Before / After

```mlir
// ──────── BEFORE: no placement attributes ────────
air.herd @worker tile(%tx, %ty) in (%sx = %c4, %sy = %c2) {
  ...
}
// -air-to-aie falls back to CLI row/col offsets — potentially wrong

// ──────── AFTER: -air-place-herds -num-rows 4 -num-cols 4 ────────
air.herd @worker tile(%tx, %ty) in (%sx = %c4, %sy = %c2)
    attributes { x_loc = 0 : i32, y_loc = 2 : i32 } {
  ...
}
// -air-to-aie maps this to columns 0..3, rows 2..3 — matches device layout
```

---

## Conversion Passes

### `-air-to-aie`

**(AIRToAIEPass)** — Lower `air.segment` + `air.herd` → `aie.device` modules + AIRRt metadata.

**Key options) |
| `-generate-shim-dma` | Emit static shim DMA programs (else AIR runtime programs them) |
| `-row-offset / -col-offset` | Default tile origin for herds without explicit `x_loc`/`y_loc` |

#### Conversion example — 1×1 herd copy → full AIE resource allocation

```mlir
// ──────── INPUT (AIR dialect) ────────
air.herd @h tile(%tx, %ty) in (%c1, %c1)
    args(%a = %src) : memref<1024xi32> {
  %buf = memref.alloc() : memref<1024xi32, 2>
  air.dma_memcpy_nd (%buf[] [] [], %a[] [] []) {id = 1 : i32} : (...)
  air.herd_terminator
}

// ──────── OUTPUT (AIE dialect module) ────────
module @aie.segment_0 {
  %tile_1_1 = aie.tile(1, 1)   // allocated physical tile
  %shim     = aie.tile(2, 0)   // shim DMA tile
  %lock0    = aie.lock(%tile_1_1, 0)
  %buf0     = aie.buffer(%tile_1_1) : memref<1024xi32, 2>
  aie.flow(%shim, DMA : 0, %tile_1_1, DMA : 0)   // data path wired
  %mem  = aie.mem(%tile_1_1)  { /* DMA buffer descriptor program */ }
  %core = aie.core(%tile_1_1) { /* herd body code */               }
}
```

---

### `-air-to-std`

**(AIRToStdPass)** — Lower `air.herd` launches → affine loop nests + `airrt` DMA calls (host control code).

#### Conversion example

```mlir
// ──────── INPUT ────────
air.herd @herd_0 tile(%tx, %ty) in (%c1, %c1)
    args(%a = %src) : memref<1024xi32> {
  air.dma_memcpy_nd (...) {id = 1 : i32} : (...)
  air.herd_terminator
}

// ──────── OUTPUT ────────
%h = airrt.herd_load "herd_0" : i64
affine.for %tx = 0 to 1 {
  affine.for %ty = 0 to 1 {
    airrt.dma_memcpy_nd(%id, %tx_i64, %ty_i64, %src
      [0,0,0,0], [1,1,1,1024], [0,0,0])
      {metadata = @airMemcpyId1}
  }
}
```

---

### `-airrt-to-npu`

**(AIRRtToNpuPass)** — Unroll affine loops → flat `aiex.npu.dma_memcpy_nd` instruction sequence.

**Options:** `-trace-size`, `-trace-offset`, `-output-elf`

#### Conversion example — loop unrolled to static instruction list

```mlir
// ──────── INPUT (airrt affine loops) ────────
affine.for %i = 0 to 4 {       // 4 tiles
  affine.for %j = 0 to 4 {
    airrt.dma_memcpy_nd(...)    // runtime loop overhead
  }
}

// ──────── OUTPUT (static NPU instructions, precomputed offsets) ────────
aiex.npu.dma_memcpy_nd(0, 0, %arg [0, 0,   0, 0]...) {id = 0}
aiex.npu.dma_memcpy_nd(0, 0, %arg [0, 0, 128, 0]...) {id = 1}
aiex.npu.dma_memcpy_nd(0, 0, %arg [0, 0, 256, 0]...) {id = 2}
aiex.npu.dma_memcpy_nd(0, 0, %arg [0, 0, 384, 0]...) {id = 3}
// ... (16 total) — no runtime loop, minimal host-side overhead
```

---

### Structural Passes

These passes lift parallel loops into AIR hierarchy scopes.

| Pass | Input → Output | Key option |
|------|---------------|------------|
| `-air-par-to-herd` | `scf.parallel` → `air.herd` | `-depth`, `-first-dim` |
| `-air-par-to-segment` | `scf.parallel` → `air.segment` | `-depth` |
| `-air-par-to-launch` | `scf.parallel` → `air.launch` | `-has-air-segment` |
| `-air-insert-launch-around-herd` | Wraps orphan `air.herd` in `air.launch` [+ `air.segment`] | `-insert-segment` |
| `-air-copy-to-dma` | `memref.copy` → `air.dma_memcpy_nd` | — |
| `-air-linalg-to-func` | `linalg.*` → function calls into precompiled `.o` | `-link-with <path>` |
| `-air-linalg-to-affine` | `linalg.*` → `affine.for` loops for AIE core codegen | — |

---

## Progressive Test Suite

Tests are ordered from beginner to expert. Each introduces new concepts and builds on the previousgment`, `air.herd`, `air.dma_memcpy_nd`  
**Goal:** Copy a 1024-element integer vector from L3 DDR into L1 tile memory and back out. The simplest possible AIR program — no async, no optimization, just the three hierarchy levels and two DMA transfers.

```mlir
// FILE: test_01_hello_herd.mlir
// RUN: air-opt -air-to-aie -device=npu1_1col -air-to-std %s | FileCheck %s
// CHECK: aie.tile
// CHECK: aie.buffer
// CHECK: aie.flow

func.func @hello_herd(
    %src : memref<1024xi32>,   // L3: input vector in DDR
    %dst : memref<1024xi32>    // L3: output vector in DDR
) {
  %c1 = arith.constant 1 : index

  // ─── Layer 1: Launch — the host job boundary ───
  air.launch args(%la = %src, %lb = %dst)
      : memref<1024xi32>, memref<1024xi32> {

    // ─── Layer 2: Segment — reserves tiles + DMA controller ───
    // x_loc=0, y_loc=2: column 0, starting at row 2 (above shim row)
    air.segment @seg args(%sa = %la, %sb = %lb)
        : memref<1024xi32>, memref<1024xi32>
        attributes { x_loc = 0 : i32, y_loc = 2 : i32,
                     x_size = 1 : i32, y_size = 1 : i32 } {

      // ─── Layer 3: Herd — the 1×1 array of AIE compute tiles ───
      air.herd @copy_tile tile(%tx, %ty) in (%sx = %c1, %sy = %c1)
          args(%ha = %sa, %hb = %sb)
          : memref<1024xi32>, memref<1024xi32> {

        // Allocate L1 tile buffer (memory space 2)
        %l1 = memref.alloc() : memref<1024xi32, 2>

        // DMA #1: L3 → L1  (load)    [] [] [] = whole buffer, contiguous
        air.dma_memcpy_nd (%l1[] [] [], %ha[] [] [])
            {id = 1 : i32} : (memref<1024xi32, 2>, memref<1024xi32>)

        // DMA #2: L1 → L3  (store)
        air.dma_memcpy_nd (%hb[] [] [], %l1[] [] [])
            {id = 2 : i32} : (memref<1024xi32>, memref<1024xi32, 2>)

        memref.dealloc %l1 : memref<1024xi32, 2>
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
```

---

### T02 — Strided 2D DMA

**Level:** Beginner+  
**Concepts:** `dma_memcpy_nd` offsets/sizes/strides, per-tile offset calculation with `affine.apply`  
**Goal:** A 4×1 herd. Each tile extracts its own 16×128 row-slice from a 64×128 L3 matrix into an L1 buffer using a strided 2D DMA.

```mlir
// FILE: test_02_strided_dma.mlir
// RUN: air-opt -air-to-aie -device=npu1_1col %s | FileCheck %s
// CHECK-COUNT-4: aie.tile

// Map: tile column index → starting row in the matrix
#tile_row_offset = affine_map<(d0) -> (d0 * 16)>

func.func @strided_tile_load(%matrix : memref<64x128xf32>) {
  %c4  = arith.constant 4   : index
  %c1  = arith.constant 1   : index
  %c16 = arith.constant 16  : index
  %c128= arith.constant 128 : index
  %c0  = arith.constant 0   : index

  air.launch args(%lmat = %matrix) : memref<64x128xf32> {
    air.segment @seg args(%smat = %lmat) : memref<64x128xf32>
        attributes { x_loc = 0 : i32, y_loc = 2 : i32,
                     x_size = 4 : i32, y_size = 1 : i32 } {

      // 4×1 herd: tiles at (0,0), (1,0), (2,0), (3,0)
      air.herd @loader tile(%tx, %ty) in (%sx = %c4, %sy = %c1)
          args(%hmat = %smat) : memref<64x128xf32> {

        // Each tile owns 16 rows: tile tx → rows [tx*16, tx*16+16)
        %row_start = affine.apply #tile_row_offset(%tx)

        %l1 = memref.alloc() : memref<16x128xf32, 2>

        // Strided 2D DMA:
        //   offsets = [row_start, 0]    → start at row tx*16, col 0
        //   sizes   = [16, 128]         → copy 16 rows × 128 cols
        //   strides = [128, 1]          → row stride = [],
           %hmat[%row_start, %c0][%c16, %c128][%c128, %c1])
          {id = 1 : i32}
          : (memref<16x128xf32, 2>, memref<64x128xf32>)

        memref.dealloc %l1 : memref<16x128xf32, 2>
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
```

---

### T03 — Async Parallel Pipeline

**Level:** Intermediate  
**Concepts:** `air.execute`, `air.wait_all`, `!air.async.token`, manual CDFG wiring  
**Goal:** Load matrices A and B in parallel (no dep between them), then compute matmul only after both arrive. Written manually — in practice `-air-dependency` generates this automatically from synchronous code.

```mlir
// FILE: test_03_async_pipeline.mlir
// RUN: air-opt -air-dependency-canonicalize -air-to-aie -device=npu1_1col %s

func.func @async_matmul(
    %A : memref<16x32xf32>,
    %B : memref<32x64xf32>,
    %C : memref<16x64xf32>
) {
  %c1 = arith.constant 1 : index

  air.launch args(%la = %A, %lb = %B, %lc = %C)
      : memref<16x32xf32>, memref<32x64xf32>, memref<16x64xf32> {
    air.segment @seg args(%sa = %la, %sb = %lb, %sc = %lc)
        : memref<16x32xf32>, memref<32x64xf32>, memref<16x64xf32>
        attributes { x_loc = 0 : i32, y_loc = 2 : i32,
                     x_size = 1 : i32, y_size = 1 : i32 } {

      air.herd @mm tile(%tx, %ty) in (%sx = %c1, %sy = %c1)
          args(%ha = %sa, %hb = %sb, %hc = %sc)
          : memref<16x32xf32>, memref<32x64xf32>, memref<16x64xf32> {

        // Step 1: Allocate three L1 buffers — no dependencies → all start immediately
        %tAl, %l1_A = air.execute -> (memref<16x32xf32, 2>) {
          %a = memref.alloc() : memref<16x32xf32, 2>
          air.execute_terminator %a : memref<16x32xf32, 2>
        } {id = 1 : i32}

        %tBl, %l1_B = air.execute -> (memref<32x64xf32, 2>) {
          %b = memref.alloc() : memref<32x64xf32, 2>
          air.execute_terminator %b : memref<32x64xf32, 2>
        } {id = 2 : i32}

        %tCl, %l1_C = air.execute -> (memref<16x64xf32, 2>) {
          %c = memref.alloc() : memref<16x64xf32, 2>
          air.execute_terminator %c : memref<16x64xf32, 2>
        } {id = 3 : i32}

        // Step 2: Load A and B IN PARALLEL — each depends only on its own alloc
        %tA = air.dma_memcpy_nd async [%tAl]
            (%l1_A[] [] [], %ha[] [] []) {id = 4 : i32}
            : (memref<16x32xf32, 2>, memref<16x32xf32>)

        %tB = air.dma_memcpy_nd async [%tBl]
            (%l1_B[] [] [], %hb[] [] []) {id = 5 : i32}
            : (memref<32x64xf32, 2>, memref<32x64xf32>)

        %tC = air.dma_memcpy_nd async [%tCl]
            (%l1_C[] [] [], %hc[] [] []) {id = 6 : i32}
            : (memref<16x64xf32, 2>, memref<16x64xf32>)

        // Step 3: Barrier — wait for ALL three loads, then compute
        %all_ready = air.wait_all async [%tA, %tB, %tC] {id = 7 : i32}

        %tCompute = air.execute [%all_ready] {
          linalg.matmul
            ins(%l1_A, %l1_B : memref<16x32xf32, 2>, memref<32x64xf32, 2>)
            outs(%l1_C : memref<16x64xf32, 2>)
        } {id = 8 : i32}

        // Step 4: Store result, then dealloc
        %tStore = air.dma_memcpy_nd async [%tCompute]
            (%hc[] [] [], %l1_C[] [] []) {id = 9 : i32}
            : (memref<16x64xf32>, memref<16x64xf32, 2>)

        %tDA = air.execute [%tCompute] { memref.dealloc %l1_A } {id = 10 : i32}
        %tDB = air.execute [%tCompute] { memref.dealloc %l1_B } {id = 11 : i32}
        %tDC = air.execute [%tStore]   { memref.dealloc %l1_C } {id = 12 : i32}

        air.wait_all [%tDA, %tDB,   air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
```

---

### T04 — Channels + Ping-Pong

**Level:** Intermediate  
**Concepts:** `air.channel`, `channel.put/get`, `-air-ping-pong-transform`, streaming loops  
**Goal:** Compute `Y = α·X + Y` (SAXPY) in streaming fashion over 8 chunks using channels. The feeder pushes chunks via channels; each tile receives and computes, with ping-pong overlap applied by the pass.

```mlir
// FILE: test_04_channels_pingpong.mlir
// RUN: air-opt \
//   -air-dependency \
//   -air-label-scf-for-in-air-segment-ping-pong \
//   -air-ping-pong-transform -keep-memref-dealloc \
//   -air-dependency-canonicalize \
//   -air-to-aie -device=npu1_1col %s

// Module-level channel declarations
air.channel @X_stream [1, 1]
air.channel @Y_in     [1, 1]
air.channel @Y_out    [1, 1]

// Affine maps for indexing
#map_id = affine_map<(d0) -> (d0)>

func.func @saxpy_streaming(
    %X     : memref<1024xf32>,   // L3: input X
    %Y     : memref<1024xf32>,   // L3: in-out Y
    %alpha : f32
) {
  %c1   = arith.constant 1   : index
  %c8   = arith.constant 8   : index
  %c128 = arith.constant 128 : index
  %c0   = arith.constant 0   : index

  air.launch args(%lx = %X, %ly = %Y) : memref<1024xf32>, memref<1024xf32> {
    air.segment @seg args(%sx = %lx, %sy = %ly) : memref<1024xf32>, memref<1024xf32>
        attributes { x_loc = 0 : i32, y_loc = 2 : i32,
                     x_size = 1 : i32, y_size = 1 : i32 } {

      // ── FEEDER: segment body streams 8 × 128-element chunks ──
      scf.for %i = %c0 to %c8 step %c1 {
        %off = arith.muli %i, %c128 : index
        // Send X chunk to tile
        air.channel.put @X_stream[%c0, %c0]
            (%sx[%off][%c128][%c1]) : (memref<1024xf32>)
        // Send Y chunk for accumulation
        air.channel.put @Y_in[%c0, %c0]
            (%sy[%off][%c128][%c1]) : (memref<1024xf32>)
        // Receive computed Y chunk
        air.channel.get @Y_out[%c0, %c0]
            (%sy[%off][%c128][%c1]) : (memref<1024xf32>)
      }

      // ── CONSUMER HERD: 1×1 tile computes SAXPY ──
      air.herd @saxpy_tile tile(%tx, %ty) in (%c1, %c1) {
        // Single-buffer form — -air-ping-pong-transform will double these
        %x_buf = memref.alloc() : memref<128xf32, 2>
        %y_buf = memref.alloc() : memref<128xf32, 2>

        // Streaming loop — -air-label-scf-for marks this for ping-pong
        scf.for %i = %c0 to %c8 step %c1 {
          air.channel.get @X_stream[%tx, %ty] (%x_buf[] [] []) : (memref<128xf32, 2>)
          air.channel.get @Y_in    [%tx, %ty] (%y_buf[] [] []) : (memref<128xf32, 2>)

          // SAXPY: y[i] = alpha * x[i] + y[i]
          linalg.generic
            { indexing_maps = [#map_id, #map_id],
              iterator_types = ["parallel"] }
            ins(%x_buf : memref<128xf32, 2>)
            outs(%y_buf : memref<128xf32, 2>) {
            ^bb0(%x_val : f32, %y_val : f32):
              %scaled = arith.mulf %alpha, %x_val : f32
              %result = arith.addf %scaled, %y_val : f32
              linalg.yield %result : f32
          }

          air.channel.put @Y_out[%tx, %ty] (%y_buf[] [] []) : (memref<128xf32, 2>)
        }
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
```

---

### T05 — 4×4 Tiled GEMM

**Level:** Intermediate+  
**Concepts:** 2D herd indexing, per-tile output tile assignment, full parallel GEMM decomposition  
**Goal:** Multiply 128×128 matrices using a 4×4 herd. Eamputes its 32×32 output tile. Shows two-level index calculation (row = ty*32, col = tx*32).

```mlir
// FILE: test_05_4x4_matmul.mlir
// RUN: air-opt -air-dependency -air-dependency-canonicalize \
//             -air-dependency-schedule-opt \
//             -air-place-herds -num-rows 4 -num-cols 4 \
//             -air-to-aie -device=npu1_1col %s

#map_ty32 = affine_map<(d0) -> (d0 * 32)>
#map_tx32 = affine_map<(d0) -> (d0 * 32)>

func.func @tiled_gemm_4x4(
    %A : memref<128x128xf32>,
    %B : memref<128x128xf32>,
    %C : memref<128x128xf32>
) {
  %c4  = arith.constant 4   : index
  %c32 = arith.constant 32  : index
  %c128= arith.constant 128 : index
  %c1  = arith.constant 1   : index
  %c0  = arith.constant 0   : index

  air.launch args(%la = %A, %lb = %B, %lc = %C)
      : memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32> {
    air.segment @seg args(%sa = %la, %sb = %lb, %sc = %lc)
        : memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32>
        attributes { x_loc = 0 : i32, y_loc = 2 : i32,
                     x_size = 4 : i32, y_size = 4 : i32 } {

      // ── 4×4 HERD: 16 tiles, each computes a 32×32 output tile ──
      //   tx ∈ {0,1,2,3}: column  → maps to output C column block
      //   ty ∈ {0,1,2,3}: row     → maps to output C row block
      air.herd @gemm tile(%tx, %ty) in (%sx = %c4, %sy = %c4)
          args(%ha = %sa, %hb = %sb, %hc = %sc)
          : memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32> {

        // This tile owns output block C[row_off:row+32, col_off:col+32]
        %row_off = affine.apply #map_ty32(%ty)   // ty * 32
        %col_off = affine.apply #map_tx32(%tx)   // tx * 32

        %l1_A = memref.alloc() : memref<32x128xf32, 2>   // full A row-block
        %l1_B = memref.alloc() : memref<128x32xf32, 2>   // full B col-block
        %l1_C = memref.alloc() : memref<32x32xf32,  2>   // output tile

        // Load A: rows [row_off, row_off+32), all 128 columns
        //   offsets=[row_off, 0]  sizes=[32, 128]  strides=[128, 1]
        air.dma_memcpy_nd
          (%l1_A[] [] [],
           %ha[%row_off, %c0][%c32, %c128][%c128, %c1])
          {id = 1 : i32}
          : (memref<32x128xf32, 2>, memref<128x128xf32>)

        // Load B: columns [col_off, col_off+32), all 128 rows
        //   offsets=[0, col_off]  sizes=[128, 32]  strides=[128, 1]
        air.dma_memcpy_nd
          (%l1_B[] [] [],
           %hb[%c0, %col_off][%c128, %c32][%c128, %c1])
          {id = 2 : i32}
          : (memref<128x32xf32, 2>, memref<128x128xf32>)

        // Load C tile (for accumulation)
        air.dma_memcpy_nd
          (%l1_C[] [] [],
           %hc[%row_off, %col_off][%c32, %c32][%c128, %c1])
          {id = 3 : i32}
          : (memref<32x32xf32, 2>, memref<128x128xf32>)

        // Compute 32×32 matmul — each tile runs independently
        linalg.matmul
          ins(%l1_A, %l1_B : memref<32x128xf32, 2>, memref<128x32xf32, 2>)
          outs(%l1_C : memref<32x32xf32, 2>)

        // Store result tile back to L3
        air.dma_memcpy_nd
          (%hc[%row_off, %col_off][%c32, %c32][%c128, %c1],
           %l1_C[] [] [])
          {id = 4 : i32}
          : (memref<128x128xf32>, memref<32x32xf32, 2>)

        memref.dealloc %l1_A : memref<32x128xf32, 2>
        memref.dealloc %l1_B : memref<128x32xf32, 2>
        memref.dealloc %l1_C : memref<32x32xf32,  2>
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
```

---

### T06 — Broadcast DMA (8×8 herd)

**Level:** `broadcast_pattern` affine set, `-air-dependency-schedule-opt`, hardware multicast  
**Goal:** In an 8×8 matmul herd, all 8 tiles in the same row share the same row of matrix A. This test shows the broadcast annotation and its affine set semantics.

```mlir
// FILE: test_06_broadcast.mlir
// RUN: air-opt -air-dependency -air-dependency-schedule-opt %s | FileCheck %s
// CHECK: broadcast_pattern
// CHECK: affine_set

// These affine sets are auto-generated by -air-dependency-schedule-opt.
// They are shown here for documentation.
//
// #bcast_A: "all tiles in the same row (d0==s0), any column (d1: 0..7)"
#bcast_A = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 7 >= 0,
                                        s0 >= 0, -s0 + 7 >= 0)>
// #bcast_B: "all tiles in the same column (d1==s0), any row (d0: 0..7)"
#bcast_B = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 7 >= 0, d1 - s0 == 0,
                                        s0 >= 0, -s0 + 7 >= 0)>

#map16 = affine_map<(d0) -> (d0 * 16)>

func.func @broadcast_gemm_8x8(
    %A : memref<128x128xf32>,
    %B : memref<128x128xf32>,
    %C : memref<128x128xf32>
) {
  %c8  = arith.constant 8   : index
  %c16 = arith.constant 16  : index
  %c128= arith.constant 128 : index
  %c1  = arith.constant 1   : index
  %c0  = arith.constant 0   : index

  air.launch args(%la = %A, %lb = %B, %lc = %C) : ... {
    air.segment @seg args(%sa = %la, %sb = %lb, %sc = %lc) : ...
        attributes { x_loc = 0 : i32, y_loc = 2 : i32,
                     x_size = 8 : i32, y_size = 8 : i32 } {

      air.herd @gemm8x8 tile(%tx, %ty) in (%sx = %c8, %sy = %c8)
          args(%ha = %sa, %hb = %sb, %hc = %sc) : ... {

        %row_off = affine.apply #map16(%ty)   // ty * 16
        %col_off = affine.apply #map16(%tx)   // tx * 16

        %l1_A = memref.alloc() : memref<16x128xf32, 2>
        %l1_B = memref.alloc() : memref<128x16xf32, 2>
        %l1_C = memref.alloc() : memref<16x16xf32,  2>

        // -air-dependency-schedule-opt detects: same %ty → same %row_off
        // → annotates with broadcast_pattern = #bcast_A
        // → -air-to-aie emits ONE aie.flow fanning out to all 8 column-tiles in this row
        %tA = air.dma_memcpy_nd async
          (%l1_A[] [] [],
           %ha[%row_off, %c0][%c16, %c128][%c128, %c1])
          {id = 1 : i32, broadcast_pattern = #bcast_A}
          : (memref<16x128xf32, 2>, memref<128x128xf32>)

        // -air-dependency-schedule-opt detects: same %tx → same %col_off
        // → annotates with broadcast_pattern = #bcast_B
        %tB = air.dma_memcpy_nd async
          (%l1_B[] [] [],
           %hb[%c0, %col_off][%c128, %c16][%c128, %c1])
          {id = 2 : i32, broadcast_pattern = #bcast_B}
          : (memref<128x16xf32, 2>, memref<128x128xf32>)

        %all = air.wait_all async [%tA, %tB] {id = 3 : i32}
        %mm  = air.execute [%all] {
          linalg.matmul
            ins(%l1_A, %l1_B : memref<16x128xf32, 2>, memref<128x16xf32, 2>)
            outs(%l1_C : memref<16x16xf32, 2>)
        } {id = 4 : i32}

        %tSt = air.dma_memcpy_nd async [%mm]
          (%hc[%row_off, %col_off][%c16, %c16][%c128, %c1],
           %l1_C[] [] [])
          {id = 5 : i32}
          : (memref<128x128xf32>, memref<16x16xf32, 2>)

        air.wait_all [%tSt]
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
```

---

### T07 — Tiled 1D Convolution

**Level:** Advanced  
**Concepts:** `scf.for` reduction, broadcast channel for weights, auto `-air-dependency`, `-air-fuse-channels`  
**Goolution `output[i] = Σ_k input[i+k] * weight[k]` across 4 tiles. Written synchronously — let `-air-dependency` discover the parallelism.

```mlir
// FILE: test_07_conv1d.mlir
// RUN: air-opt -air-dependency -canonicalize \
//             -air-dependency-canonicalize \
//             -air-fuse-channels \
//             -air-to-aie -device=npu1_1col %s

// Weight broadcast (same weights go to all 4 tiles)
air.channel @weight_bcast [4, 1]
// Per-tile input/output
air.channel @input_tiles  [4, 1]
air.channel @output_tiles [4, 1]

func.func @conv1d_4tiles(
    %input  : memref<256xf32>,    // L3: input signal, length 256
    %weight : memref<16xf32>,     // L3: filter, length 16 (shared by all tiles)
    %output : memref<192xf32>     // L3: output, length 256-64 = 192 (for K_size=64)
) {
  %c4  = arith.constant 4  : index
  %c48 = arith.constant 48 : index   // output per tile = 192 / 4
  %c64 = arith.constant 64 : index   // input per tile = 48 + 16 - 1
  %c16 = arith.constant 16 : index   // filter size
  %c1  = arith.constant 1  : index
  %c0  = arith.constant 0  : index

  air.launch args(%lin = %input, %lw = %weight, %lo = %output)
      : memref<256xf32>, memref<16xf32>, memref<192xf32> {
    air.segment @seg args(%sin = %lin, %sw = %lw, %so = %lo)
        : memref<256xf32>, memref<16xf32>, memref<192xf32>
        attributes { x_loc = 0 : i32, y_loc = 2 : i32,
                     x_size = 4 : i32, y_size = 1 : i32 } {

      // Broadcast the 16-element weight to all 4 tiles (loaded once)
      scf.for %col = %c0 to %c4 step %c1 {
        air.channel.put @weight_bcast[%col, %c0]
            (%sw[] [] []) : (memref<16xf32>)
      }

      // Stream distinct input chunks to each tile; collect output
      scf.for %col = %c0 to %c4 step %c1 {
        %in_off  = arith.muli %col, %c48 : index   // input start: col*48
        %out_off = arith.muli %col, %c48 : index   // output start: col*48
        air.channel.put @input_tiles[%col, %c0]
            (%sin[%in_off][%c64][%c1]) : (memref<256xf32>)
        air.channel.get @output_tiles[%col, %c0]
            (%so[%out_off][%c48][%c1]) : (memref<192xf32>)
      }

      // 4×1 herd: each tile convolves its 64-element input slice
      air.herd @conv_tile tile(%tx, %ty) in (%c4, %c1) {
        %l1_w = memref.alloc() : memref<16xf32,  2>
        %l1_x = memref.alloc() : memref<64xf32,  2>
        %l1_y = memref.alloc() : memref<48xf32,  2>

        air.channel.get @weight_bcast[%tx, %ty] (%l1_w[] [] []) : (memref<16xf32,  2>)
        air.channel.get @input_tiles  [%tx, %ty] (%l1_x[] [] []) : (memref<64xf32,  2>)

        // Conv1D: output[i] = sum_{k=0}^{15} input[i+k] * weight[k]
        linalg.conv_1d
          ins(%l1_x, %l1_w : memref<64xf32, 2>, memref<16xf32, 2>)
          outs(%l1_y : memref<48xf32, 2>)

        air.channel.put @output_tiles[%tx, %ty] (%l1_y[] [] []) : (memref<48xf32, 2>)
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
```

---

### T08 — Full E2E GEMM Pipeline

**Level:** Expert  
**Concepts:** All optimizations combined; complete pass pipeline; parallel K-dimension launch  
**Goal:** 512×512×512 GEMM using every optimization pass, ending in a flat NPU instruction sequence. Demonstrates the full compilation command, module structure, and how passes compose.

```mlir
// FILE: test_08_full_e2e_gemm.mlir
//
// ── COMPLETE PASS PIPELINE ──────────────────────────────────────────────
// air-opt                        ← MLIR cleanup
//   -air-dependency-canonicalize \                     ← transitive reduction
//   -air-dependency-schedule-opt \                     ← broadcast detection
//   -air-label-scf-for-in-air-segment-ping-pong \      ← mark PP candidates
//   -air-ping-pong-transform -keep-memref-dealloc \    ← double-buffer
//   -air-fuse-channels \                               ← reduce channel count
//   -air-specialize-channel-wrap-and-stride \          ← BD optimization
//   -air-place-herds -num-rows 4 -num-cols 4 \         ← physical placement
//   -air-to-aie -device=npu1_1col \                    ← AIR → AIE
//     -use-objectfifo -generate-shim-dma \
//   -air-to-std \                                      ← host control code
//   -airrt-to-npu \                                    ← NPU instructions
//   test_08_full_e2e_gemm.mlir -o lowered.mlir
//
// ── COMPILE ─────────────────────────────────────────────────────────────
// aircc.py lowered.mlir -o gemm.a -num-rows 4 -num-cols 4 --shared

air.channel @A_feed [4, 4]
air.channel @B_feed [4, 4]
air.channel @C_out  [4, 4]

#map128 = affine_map<(d0) -> (d0 * 128)>

func.func @gemm_512x512x512(
    %A : memref<512x512xf32>,
    %B : memref<512x512xf32>,
    %C : memref<512x512xf32>
) {
  %c4  = arith.constant 4   : index
  %c128= arith.constant 128 : index
  %c512= arith.constant 512 : index
  %c1  = arith.constant 1   : index
  %c0  = arith.constant 0   : index

  // Parallel launch over the K dimension: 4 iterations × 128 = 512
  air.launch (%lk) in (%nk = %c4)
      args(%la = %A, %lb = %B, %lc = %C)
      : memref<512x512xf32>, memref<512x512xf32>, memref<512x512xf32> {

    %k_off = affine.apply #map128(%lk)   // lk * 128

    air.segment @seg args(%sa = %la, %sb = %lb, %sc = %lc)
        : memref<512x512xf32>, memref<512x512xf32>, memref<512x512xf32>
        attributes { x_loc = 0 : i32, y_loc = 2 : i32,
                     x_size = 4 : i32, y_size = 4 : i32 } {

      // ── FEEDER: push 128×128 tiles to each of 16 tiles ──────────────
      scf.for %row = %c0 to %c4 step %c1 {
        scf.for %col = %c0 to %c4 step %c1 {
          %r_off = affine.apply #map128(%row)
          %c_off = affine.apply #map128(%col)

          // A[row*128:+128, k_off:+128] → tile[row, col]
          // -air-dependency-schedule-opt: same %row → broadcast_pattern across cols
          air.channel.put @A_feed[%row, %col]
            (%sa[%r_off, %k_off][%c128, %c128][%c512, %c1])
            : (memref<512x512xf32>)

          // B[k_off:+128, col*128:+128] → tile[row, col]
          // -air-dependency-schedule-opt: same %col → broadcast_pattern across rows
          air.channel.put @B_feed[%row, %col]
            (%sb[%k_off, %c_off][%c128, %c128][%c512, %c1])
            : (memref<512x512xf32>)

          air.channel.get @C_out[%row, %col]
            (%sc[%r_off, %c_off][%c128, %c128][%c512, %c1])
            : (memref<512x512xf32>)
        }
      }

      // ── 4×4 HERD: each tile accumulates 128×128×128 partial GEMM ────
      air.herd @gemm_core tile(%tx, %ty) in (%c4, %c4) {

        // Single-buffer allocations — -air-ping-pong-transform will double these
        %l1_A = memref.alloc() : memref<128x128xf32, 2>
        %l1_B = memref.alloc() : memref<128x128xf32, 2>
        %l1_C = memref.alloc() : memref<128x128xf32, 2>

        air.channel.get @A_ linalg.matmul
          ins(%l1_A, %l1_B : memref<128x128xf32, 2>, memref<128x128xf32, 2>)
          outs(%l1_C : memref<128x128xf32, 2>)

        air.channel.put @C_out[%ty, %tx] (%l1_C[] [] []) : (memref<128x128xf32, 2>)

        memref.dealloc %l1_A : memref<128x128xf32, 2>
        memref.dealloc %l1_B : memref<128x128xf32, 2>
        memref.dealloc %l1_C : memref<128x128xf32, 2>
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
```

#### What each pass does to this program

| Pass | Effect on T08 |
|------|--------------|
| `-air-dependency` | Wraps `memref.alloc` in `air.execute`; wires `!air.async.token` edges; `@A_feed put` and `@B_feed put` get no dep on each other → parallel |
| `-air-dependency-canonicalize` | Removes transitive redundant edges, reducing total lock count |
| `-air-dependency-schedule-opt` | Detects same-row `A_feed` puts share `%r_off` → annotates `broadcast_pattern`; same for `B_feed` columns |
| `-air-label-scf-for … -ping-pong` | Marks the inner herd `scf.for` (if streaming) as a ping-pong candidate |
| `-air-ping-pong-transform` | Doubles `%l1_A`, `%l1_B`, `%l1_C` buffers; overlaps DMA[i+1] with GEMM[i] |
| `-air-fuse-channels` | Fuses compatible `@A_feed` and `@B_feed` gets into a single channel where possible |
| `-air-specialize-channel-wrap-and-stride` | Collapses the 4×4 feeder loops into single BDs with hardware wrap counts |
| `-air-place-herds` | Sets `x_loc=0, y_loc=2` on the `air.herd` |
| `-air-to-aie` | Generates 16 `aie.core` modules, 16 `aie.buffer` sets, `aie.flow` routing, lock programs |
| `-air-to-std` | Generates host-side `affine.for` loops + `airrt.dma_memcpy_nd` calls |
| `-airrt-to-npu` | Unrolls all loops into a flat `aiex.npu.dma_memcpy_nd` instruction list |

---

*MLIR-AIR API Reference — generated February 2026. Source: https://github.com/Xilinx/mlir-air*
