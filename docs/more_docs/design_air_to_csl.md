# Design Document: AIR to Spatial Backends -- From CSL Emitter to Tiled Dataflow Dialect

This is a living document tracking the design evolution of lowering AIR dialect programs to spatial dataflow accelerator backends, starting with Cerebras CSL and generalizing toward a multi-target tiled dataflow intermediate representation.

---

## 1. Current State: Direct CSL Text Emitter (v0 -- Implemented)

The initial prototype (`air-to-csl` pass in `AIRToCSLPass.cpp`) walks AIR IR and directly emits three CSL text files:

- `air.launch` -> `run.py` (host runtime)
- `air.segment` -> `layout.csl` (PE grid topology)
- `air.herd` body -> `pe_program.csl` (per-PE kernel code)

This works for simple programs but has fundamental limitations:

- No MLIR-level verification of the generated code structure.
- No composability with optimization passes between AIR lowering and text emission.
- The emitter is a long `dyn_cast` chain that will grow unmanageably as more CSL constructs are needed (colors, routes, tasks, DSDs, fabric DSDs, microthreads).
- Cannot represent or optimize inter-PE communication at the IR level.

---

## 2. The Design Question: What Should Replace It?

Three options were considered:

**Option A: Keep the text emitter, grow it.** Fast to build, but becomes a 3000+ line string-templating function. No analysis or optimization possible on CSL-level structure.

**Option B: Build a CSL-specific MLIR dialect.** Mirror what `mlir-aie` does for AMD AI Engines. Full MLIR verification and optimization, but high engineering cost and tightly coupled to one target.

**Option C: Build a generic Tiled Dataflow (TDF) dialect parameterized by a hardware spec.** A single IR that can lower to CSL, AIE, TT-Metal, and others. Higher up-front design cost, but the abstraction serves multiple backends and is a research contribution in its own right.

**Decision: Option C (phased).** Build toward a generic TDF dialect, but in phases so that each phase produces a working prototype. Phase 1 is the current text emitter. Phase 2 introduces the TDF dialect targeting CSL and AIE. Phase 3 adds the hardware spec parameterization and additional backends.

---

## 3. Related Work: The TL (TileLoom) Framework

The TL paper ([Li et al., arXiv:2512.22168](https://arxiv.org/pdf/2512.22168)) is the closest related work to the TDF dialect idea. TL is an end-to-end compiler that maps tile-based programs (Triton kernels) onto spatial dataflow architectures. Its architecture is directly relevant to our design.

### 3.1 What TL Does

TL solves the problem of compiling a tile-level kernel (e.g., a Triton GEMM) onto a 2D grid of cores connected by an on-chip network. The compilation has three stages:

1. **Front-end**: Takes a tile-level kernel + launch grid, lowers to a dataflow-agnostic MLIR representation (`affine.parallel` over block IDs, `scf.for` over reduction dimensions, `linalg` ops for compute). Memory accesses are affinized.

2. **Dataflow planning**: Decides spatiotemporal mapping (which cores run which tiles, in what order), data reuse (which tiles are broadcast vs. loaded from DRAM), and memory allocation (which buffers hold what data at what loop level). This is the core contribution.

3. **Back-end**: Lowers the dataflow-aware MLIR to a target-specific API (TT-Metalium for Tenstorrent) and generates per-core executables.

### 3.2 TL's Hardware Representation: The `df` Dialect

The key insight from TL is their `df` dialect for hardware description. Rather than baking target knowledge into compiler passes, they parameterize the compiler by a machine description written in MLIR:

```mlir
// Scale-out: 8x8 grid of cores
%x = df.spatial_dim 8
%y = df.spatial_dim 8
%cores = df.core(scaleout = (%x, %y))

// Interconnect: horizontal and vertical rings (torus topology)
%noc_h = df.interconnects %cores, %cores {
  map = affine_map<(d0,d1) -> ((d0+1) mod 8, d1)>,
  bandwidth = 28
}
%noc_v = df.interconnects %cores, %cores {
  map = affine_map<(d0,d1) -> (d0, (d1+1) mod 8)>,
  bandwidth = 28
}

// Memory: per-core scratchpads + shared DRAM
%l1 = df.memory(scaleout = (%x,%y), size = 1499136, bandwidth = 60)
%cores_to_l1 = df.mux %cores, %l1 {
  map = affine_map<(d0,d1) -> (d0,d1)>
}
%dram_id = df.spatial_dim 4
%drams = df.memory(scaleout = %dram_id, size = 12884901888, bandwidth = 247)

// Intra-core compute units
%FPU = df.mat(shape = [32,32,32], throughput = 98)
%SFPU = df.vec(shape = [32], throughput = 3)
%cores = df.core(scaleout = (%x,%y), scalein = (%FPU, %SFPU, [8,1]))
```

This is powerful because the same compiler passes work for different architectures by changing only the `df` description. TL demonstrated this by modeling three configurations (1x8 ring, 4x8 mesh, 8x8 mesh) of the same Tenstorrent chip and also showed how to describe a 1D triple-ring topology (IBM Spyre-like).

### 3.3 What TL Does NOT Do (and We Need)

TL has several limitations that our work should address:

1. **Single front-end assumption**: TL assumes Triton-style tile programs. We want to accept AIR programs that already have spatial structure (`air.herd` with explicit tile IDs).

2. **No persistent IR for the mapped program**: TL's dataflow planning produces annotated loop nests (annotations like `{type="broadcast", resource={%noc_h}}`), not a proper MLIR dialect with ops and verifiers. The annotations are consumed by the performance model and backend but cannot be independently analyzed.

3. **No multi-backend emission**: TL targets only TT-Metalium. The `df` dialect describes hardware but there's no generic "mapped program" dialect that lowers to different backends.

4. **No communication abstraction**: TL reasons about broadcasts and global loads but doesn't model point-to-point channels, tasks, or streaming -- constructs central to CSL and AIE.

---

## 4. Proposed Architecture: The TDF Dialect

### 4.1 Overview

The proposed stack (matching the user's diagram):

```
Front-end DSLs (Triton, AIR, SPADA/IRON)
        |
        V
AIR / unified high-level spatial dialect
        |  (air-to-tdf pass)
        V
TDF (Tiled Dataflow) Dialect  <---- SPEC (hardware description)
        |
        |  (tdf-to-csl, tdf-to-aie, tdf-to-tt, ...)
        V
Backend text/IR emitters
        |
        V
Target: CSL, AIE, TT-Metal, XLA, NKI, ...
```

The TDF dialect sits between the algorithm-level spatial IR (AIR) and the target-specific backends. It captures the *mapped* program: tiles are placed, communication is routed, memory is allocated, but the representation is still target-agnostic.

### 4.2 Common Abstractions Across Targets

By cross-referencing AIE, CSL, and TT-Metal programming models, we can identify what is universal and what diverges:

| Concept | AIE | CSL | TT-Metal | Universal? |
|---|---|---|---|---|
| **Device container** | `aie.device(xcvc1902)` | `layout { @set_rectangle }` | `CreateDevice()` | Yes |
| **2D PE grid** | `aie.tile(col, row)` | `@set_tile_code(x, y)` | `CoreCoord{x,y}` | Yes |
| **Per-PE memory** | `aie.buffer(%tile)` L1=32KB | `var arr: [N]f32` 48KB | L1 SRAM per Tensix | Yes (size varies) |
| **Shared memory** | Memtile L2=512KB | None (PE-only) | DRAM | Partially (not all have L2) |
| **Per-PE compute** | `aie.core` body | `fn compute()` | Kernel on RISC-V | Yes |
| **P2P channel** | `aie.flow(src, dst)` | color + route | NoC send/recv | Yes (mechanism differs) |
| **Broadcast** | multi-dest flow | route with fan-out | NoC multicast | Yes |
| **Synchronization** | `aie.lock` (semaphore) | tasks (event-driven) | Semaphores on CBs | Yes (model differs) |
| **FIFO/streaming** | `aie.objectfifo` | fabric DSDs + queues | Circular buffers | Yes |
| **Host data xfer** | Shim DMA | `memcpy` module / `SdkRuntime` | Host API | Yes |
| **DMA descriptors** | `aie.dma_bd` | DSDs (Data Structure Descriptors) | None (explicit copies) | Partially |

### 4.3 Proposed TDF Ops

Based on the commonality analysis, the TDF dialect needs these op categories:

#### Device and Spatial Structure

```
tdf.device @name { arch = "wse3", grid = [W, H] }
```
Top-level container. The `arch` attribute (or a separate SPEC) parameterizes backend-specific choices. Contains all tiles, channels, and host transfers.

```
tdf.tile @name [col, row]
```
A single processing element at grid coordinates. Contains memory declarations and compute regions. Analogous to `aie.tile` / CSL's `@set_tile_code` / TT's `CoreCoord`.

#### Memory

```
tdf.buffer @name on @tile : memref<NxTy>
```
A buffer allocated on a specific tile's local memory. Analogous to `aie.buffer` / CSL's `var arr: [N]f32`.

```
tdf.global @name : memref<NxTy>
```
A buffer in shared/external memory (DRAM, L2, HBM). Not all targets have L2; the SPEC determines what memory levels exist.

#### Communication

```
tdf.channel @name : memref<NxTy> { depth = 2 }
```
An abstract producer-consumer channel. The `depth` attribute enables double-buffering. This is the key abstraction: it maps to `aie.objectfifo` on AIE, a color+route+task on CSL, and a circular buffer on TT-Metal.

```
tdf.send @channel, %data : memref<NxTy>
tdf.recv @channel -> %buf : memref<NxTy>
```
Send/receive on a channel. Blocking semantics. The backend decides whether this becomes a DMA BD chain, a fabric DSD operation, or a NoC transfer.

```
tdf.connect @src_tile::@ch -> @dst_tile::@ch
```
Declares a point-to-point or multicast connection. On AIE this becomes `aie.flow`. On CSL this becomes a color assignment + routing configuration. On TT this becomes a NoC route.

#### Compute

```
tdf.compute @tile {
  // Standard MLIR: arith, memref, scf, linalg, vector
}
```
The per-PE computation. Uses standard MLIR dialects, not TDF-specific ops. This keeps the compute representation backend-agnostic and reuses existing MLIR infrastructure.

#### Host Interface

```
tdf.host_send @global -> @tile::@buf  // H2D
tdf.host_recv @tile::@buf -> @global  // D2H
```
Host-to-device and device-to-host transfers. On AIE these become shim DMA programs. On CSL these become `memcpy_h2d` / `memcpy_d2h` calls. On TT these become host API calls.

### 4.4 The SPEC: Hardware Description

Inspired by TL's `df` dialect but simpler for a prototype. The SPEC can be:

1. **Attributes on `tdf.device`** (simplest, for prototype):
   ```
   tdf.device @wse3 {
     grid = [750, 994],
     mem_per_tile = 48000,  // bytes
     interconnect = "2d_mesh",
     comm_directions = ["N", "S", "E", "W", "RAMP"],
     host_interface = "memcpy"
   }
   ```

2. **A separate `df`-style dialect** (TL approach, for production):
   Full `df.spatial_dim`, `df.core`, `df.interconnects`, `df.memory` ops with affine topology maps and bandwidth numbers.

For the prototype, option 1 is sufficient. The SPEC drives backend-specific lowering decisions without changing the TDF IR itself.

---

## 5. Comparison with TL's `df` Dialect

| Aspect | TL's `df` dialect | Our TDF dialect |
|---|---|---|
| **Purpose** | Describes hardware (input to compiler) | Describes the mapped program (output of mapping) |
| **Contains** | Cores, memories, interconnects, bandwidths | Tiles, buffers, channels, compute, host xfers |
| **Analogy** | LLVM target description | LLVM IR (target-independent but lowerable) |
| **Used by** | Performance model, mapping search | Backend emitters (tdf-to-csl, tdf-to-aie) |
| **Complementary?** | Yes -- TL's `df` could be our SPEC input | Yes -- TDF is what TL is missing |

The key distinction: TL's `df` dialect describes the *machine*, not the *mapped program*. TL's mapped programs are annotated loop nests (annotations on `affine.for`/`affine.parallel`), not a proper dialect. Our TDF dialect fills exactly this gap: it represents the mapped program in a way that can be verified, optimized, and lowered to multiple backends.

In a full system, the compilation flow would be:

```
AIR program + df SPEC
      |
      | (mapping pass, possibly using TL's algorithms)
      V
TDF program (mapped, target-agnostic)
      |
      | (tdf-to-csl / tdf-to-aie / tdf-to-tt)
      V
Target code (CSL text / AIE dialect / TT-Metalium C++)
```

---

## 6. Phased Implementation Plan

### Phase 1: CSL Text Emitter (DONE)

- `air-to-csl` pass emitting `layout.csl`, `pe_program.csl`, `run.py`
- Handles `air.launch` / `air.segment` / `air.herd` hierarchy
- Supports basic ops: `arith`, `memref.load/store`, `scf.for`
- No inter-PE communication, no optimization
- Branch: `air-to-fire`

### Phase 2: Minimal TDF Dialect + Dual Backend

Goal: Prove that a single IR can target both AIE and CSL.

**Step 2a**: Define TDF TableGen ops (`tdf.device`, `tdf.tile`, `tdf.buffer`, `tdf.compute`, `tdf.channel`, `tdf.connect`, `tdf.send`, `tdf.recv`, `tdf.host_send`, `tdf.host_recv`).

**Step 2b**: Write `air-to-tdf` pass that lowers `air.herd` -> `tdf.tile` + `tdf.compute`, `air.segment` -> `tdf.device`, `air.channel` -> `tdf.channel` + `tdf.connect`.

**Step 2c**: Write `tdf-to-csl` emitter (text emission from TDF ops, replacing the current direct emitter).

**Step 2d**: Write `tdf-to-aie` pass (TDF ops -> AIE dialect ops). This is the key validation: if the same TDF IR can produce working AIE and CSL output, the abstraction is sound.

### Phase 3: Hardware Spec + Optimization

- Add SPEC-based parameterization (hardware description as input).
- Implement mapping/scheduling passes on TDF (spatial placement, channel routing, buffer allocation).
- Add performance model (following TL's approach).
- Add additional backends (TT-Metal, others).

---

## 7. Critique of the Proposed Design

### Strengths

1. **The three-layer separation is well-motivated.** Algorithm (AIR) -> Mapped program (TDF) -> Target code (CSL/AIE/TT) mirrors the universal compiler pattern of high-level IR -> low-level IR -> machine code.

2. **The channel abstraction is the right common denominator.** Every spatial target has some form of producer-consumer channel. Abstracting over the mechanism (locks vs. tasks vs. semaphores) while preserving the structure (endpoints, data types, depth) is the right level.

3. **Compute stays in standard MLIR.** Not reinventing `arith`, `memref`, `scf` inside TDF avoids dialect bloat and lets existing MLIR passes (canonicalization, CSE, loop transformations) work unchanged on the compute inside `tdf.compute`.

4. **TL's `df` dialect validates the SPEC concept.** The fact that TL demonstrated architecture-parameterized compilation on real hardware (Tenstorrent Wormhole) shows the approach works.

### Weaknesses and Risks

1. **Communication model divergence.** The gap between AIE's circuit-switched flows, CSL's color/route/task model, and TT's NoC multicast is significant. A `tdf.channel` that must lower to all three may end up either too abstract (losing optimization opportunities) or requiring backend-specific attributes that defeat the purpose.

   *Mitigation*: Start with the simplest common case (point-to-point, single producer, single consumer, blocking) and add complexity only when needed.

2. **Memory hierarchy depth varies.** AIE has three levels (L1/L2/L3), CSL has two (PE local / host DRAM), TT has two (L1 / DRAM) but with different characteristics. A `tdf.buffer` needs a memory space attribute, but the set of valid spaces is target-dependent.

   *Mitigation*: Use a generic memory space enum (local, shared, external) and let the SPEC map these to physical memories.

3. **Per-tile specialization.** AIE clones the herd body per tile and specializes constants. CSL assigns the same `pe_program.csl` to all tiles but parameterizes via `comptime` params. TT runs the same kernel but with per-core coordinates. The TDF dialect needs to support all three models.

   *Mitigation*: `tdf.compute` takes tile coordinates as block arguments, like AIR's herd body. Backend emitters specialize as needed.

4. **Scope creep.** The "unified top-level DSL" layer in the diagram (merging AIR/SPADA/TL/Triton) is a multi-year project. For a paper or prototype, it's essential to scope down to AIR -> TDF -> {CSL, AIE}.

---

## 8. Open Questions

1. **Should TDF be a new MLIR dialect or an extension of AIR?** Adding ops to AIR avoids creating a new dialect but couples the "mapped" representation to the "algorithmic" one. A separate dialect is cleaner but more engineering.

2. **How to represent collective operations?** CSL has `collectives_2d` (broadcast, reduce, scatter, gather). AIE has no built-in collectives. TT has NoC multicast. Should TDF have `tdf.broadcast`, `tdf.reduce`, or should these be patterns of `tdf.send`/`tdf.recv`?

3. **Task-based execution.** CSL's execution model is fundamentally task/event-driven (tasks bound to colors, activated by wavelets). AIE is sequential within a core. TT is kernel-launch-based. How does TDF represent per-PE control flow?

4. **DSD/bulk memory operations.** CSL's DSDs provide efficient strided/patterned memory access that scalar `memref.load`/`memref.store` cannot capture. Should TDF have a DSD-like abstraction, or should this be a CSL-backend-specific optimization?

---

---

## 9. Revised Plan: CSL-First, Phased Generalization (v2)

After studying the AIE and TT compiler stacks in detail, the grand multi-backend TDF dialect is deferred. The immediate focus is building proper CSL infrastructure, following the proven `*-opt` / `*-translate` pattern used by every mature MLIR accelerator stack.

### 9.1 The `mlir-translate` Pattern (Key Architectural Decision)

Every mature MLIR accelerator stack separates IR transformation from code emission using two binaries:

```
*-opt       : MLIR → MLIR   (passes that transform IR)
*-translate : MLIR → text   (one-shot emission to external format)
```

**AIE**: `aie-opt` transforms `aie.*` IR. `aie-translate --aie-generate-xaiev2` emits C++ (libXAIE calls). `aie-translate --aie-mlir-to-llvm` emits LLVM IR.

**TT**: `ttmlir-opt --convert-ttkernel-to-emitc` transforms IR. `ttmlir-translate --mlir-to-cpp` emits C++ source. `ttmlir-translate --ttnn-to-flatbuffer` emits binary.

Both use MLIR's `TranslateFromMLIRRegistration` API.

**CSL target** (proposed):
```
air-opt --air-to-csl     →  CSL dialect MLIR  (proper MLIR pass)
air-translate --csl-emit  →  layout.csl + pe_program.csl + run.py  (text translation)
```

Phase 1 conflated these: the `air-to-csl` pass both transforms IR and emits text as a side effect. Phase 2 separates them properly.

### 9.2 Revised Phased Plan

**Phase 1: Text emitter (DONE).** Direct AIR → CSL text via a pass. Proves the mapping works. Quick to build.

**Phase 2: CSL dialect + `air-translate --csl-emit`.** Build `csl.*` MLIR ops mirroring CSL language constructs. `air-opt --air-to-csl` becomes a proper MLIR pass (AIR → CSL dialect). `air-translate --csl-emit` is a registered translation that walks `csl.*` ops and writes syntactically valid CSL source. Enables optimization and verification passes on the CSL IR before emission.

**Phase 3: Common backend dialect.** With both `csl.*` and `aie.*` dialects in hand, identify shared ops and factor them into a common `spatial.*` dialect. `spatial.tile`, `spatial.buffer`, `spatial.channel` lower to either `csl.*` or `aie.*`. New backends add a thin lowering from `spatial.*` to their target dialect.

**Phase 4: SPEC + IRDL auto-generation.** Hardware description as input. IRDL generates dialect ops from spec. New backends require only a spec file + translation function.

### 9.3 Phase 2 CSL Dialect: Concrete Op List

Based on CSL language constructs that need to be represented at the MLIR level:

| CSL construct | Proposed op | Role |
|---|---|---|
| `layout { ... }` | `csl.layout { }` | Top-level container for spatial config |
| `@set_rectangle(W, H)` | `csl.set_rectangle [W, H]` | Define PE grid dimensions |
| `@set_tile_code(x, y, ...)` | `csl.set_tile_code [x, y] @module params(...)` | Assign code to PE |
| `@import_module(...)` | `csl.import_module "name" as %ref` | Import CSL module |
| `@export_name(...)` | `csl.export_name "name" : type` | Host-visible symbol in layout |
| `@export_symbol(...)` | `csl.export @sym` | Export from PE module |
| `var x: [N]T` | `csl.var @name : memref<NxT>` | PE-local memory declaration |
| `fn name() void { }` | `csl.func @name { }` | Function definition |
| `task name() void { }` | `csl.task @name color(%c) { }` | Event-driven task bound to color |
| Color declaration | `csl.color @name = %id` | Communication color |
| Route config | `csl.route @color dir(%d)` | Routing direction for color |
| `@get_dsd(...)` | `csl.dsd mem1d(%buf, %len)` | Data structure descriptor |
| `@mov32/16(...)` | `csl.mov %dst_dsd, %src_dsd` | Bulk DSD operation |
| `comptime { }` | `csl.comptime { }` | Compile-time region |
| Module params | `csl.module @name params(...)` | Parameterized PE module |

Compute inside `csl.func` and `csl.task` bodies uses standard MLIR (`arith`, `scf`, `memref`).

### 9.4 Why This Order

1. **Phase 1 → 2 is a refactoring, not a rewrite.** The same CSL knowledge goes from string templates into structured ops. The translation function (`--csl-emit`) is essentially the same emitter code, now walking `csl.*` ops instead of `air.*` ops.

2. **Phase 2 enables CSL-level optimization.** With `csl.*` ops, passes can do color allocation, DSD pattern matching (convert scalar loops to bulk DSD moves), buffer sizing verification — all before text emission.

3. **Phase 3 emerges naturally.** Once `csl.*` and `aie.*` coexist, the shared structure becomes visible empirically rather than designed speculatively.

4. **Phase 4 is the research moonshot.** IRDL-based auto-generation is novel and requires Phases 2+3 to be stable.

---

## Revision History

| Date | Change |
|---|---|
| 2025-02-20 | v0: Initial CSL text emitter implemented (`air-to-csl` pass). |
| 2025-02-21 | v1: Design document created. Analyzed TL paper, compared AIE/CSL/TT models, proposed TDF dialect with phased plan. |
| 2025-02-21 | v2: Revised plan to CSL-first approach. Adopted `mlir-translate` pattern for code emission. Defined concrete CSL dialect op list. Deferred multi-backend TDF to Phase 3+. |
