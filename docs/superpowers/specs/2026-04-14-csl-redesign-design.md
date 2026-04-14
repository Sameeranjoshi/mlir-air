# CSL Dialect Family — Redesign Evaluation

**Date:** 2026-04-14
**Status:** Research + redesign sketches. **Not yet approved.** Read this, red‑pen it tomorrow, and we'll go from sketches to a concrete spec in a follow-up brainstorming pass.
**Predecessor:** [`2026-04-13-air-to-csl-vecadd-design.md`](2026-04-13-air-to-csl-vecadd-design.md) — the vecadd milestone whose implementation is what we are now evaluating.
**Trigger:** `mlir-aie` was added to the workspace as a reference. Hypothesis: the current CSL dialect is mixing *spatial placement / partitioning / routing* with *kernel code* and *host runtime*, and AIE/AIR may show us a cleaner factoring.
**Scope of this document:** research + 2–3 design sketches + recommendation. Not a final spec. No code is written from this doc.

---

## 0. TL;DR

1. **AIE separates four concerns very cleanly:**
   - Structural hardware (`aie.tile`, `aie.buffer`, `aie.lock`) lives at device scope, **outside** the kernel body.
   - The kernel body inside `aie.core { ... }` is **plain MLIR** (`memref/scf/arith/vector`) with only thin AIE wrappers (`aie.use_lock`, `aie.objectfifo.acquire`).
   - Routing is a **separate layer**: either explicit (`aie.flow`/`aie.switchbox`/`aie.connect`) or implicit-and-elaborated (`aie.objectfifo` → buffers + locks + flows by a stateful-transform pass).
   - The host runtime program is its **own region** (`aie.runtime_sequence` containing `aiex.npu.*` ops). It coexists with the device IR but is not interleaved.

2. **Our current CSL gets the high-level idea right but the details are muddled.** Good things: `csl.kernel` body holds no coordinates; `csl.place` separates kernel binding from region; colors and routes are operands of `csl.code_region`, not buried in the kernel. Bad things, in roughly decreasing severity:
   - **Direction is duplicated** between kernel-side `csl.export_symbol` and host-side `csl.export_name`. The emitter has to reach across the IR to reconcile them. Kernels are not reusable across host contexts.
   - **`csl.var` is a symbol-named SSA result**, and the kernel emitter walks SSA→string by hand. Naming is a side channel.
   - **`csl.kernel` carries `source_file`** — that's a codegen artifact bleeding into IR semantics.
   - **Hardcoded 1×1 placement** in `AIRToCSLDialect.cpp` (lines 310/320), hardcoded `a_buf/b_buf/c_buf` arg names (lines 223–230), and **hardcoded test data + validator** in `HostEmitter` (lines 796–880).
   - **`csl.code_region`'s body is documented but never used**, and routes/colors are declared as operands but never lowered into anything that actually wires PEs.
   - The split between `csl.*` and `csl_rt.*` is **not enforced**: `CSLToCSLRuntime` adds `csl_rt.*` ops next to the existing `csl.*` ops without removing or sealing anything. Two emitters then walk a single mixed module.

3. **AIE has the same kind of two-dialect split (`aie` for device, `aiex` for host runtime sequence), and it works because both live inside one `aie.device` symbol table with a *single* terminal emission target — not three separate emitters reading the same module.** This is a key lesson.

4. **Three redesign sketches** are below. The recommended one is **Sketch C: a three-layer split** — `csl_program` (per-PE compute, comptime-parameterized, completely placement-free), `csl_layout` (the rectangle, placements, colors, routes — direct analog of CSL's textual `layout {}` block), and `csl_host` (renamed `csl_rt`, thin host orchestration over `SdkRuntime`). The pipeline becomes one *non-mixed* module per layer, with conversions that **consume** the source layer rather than tattooing onto it.

5. **Optimizations** (placement, BD assignment, lock allocation, objectfifo elaboration, pathfinding) are listed in §11 with notes on which AIE passes are conceptually reusable. We skip them in the redesign for now and revisit once vec_add and a 2×2 example both work end-to-end on the new IR.

---

## 1. Goals and non-goals

### Goals

- **Understand AIE well enough** to know which pieces of its design are deliberate factoring decisions vs. accidents of Xilinx hardware, so we don't blindly copy.
- **Honestly diagnose** where today's CSL/CSLRT design conflates concerns and where it doesn't. (See §8 — there are wins worth keeping, not just losses.)
- **Sketch 2–3 alternative shapes** for the dialect family, with tradeoffs, so we can have a productive conversation tomorrow before committing.
- **Stay focused on structures, ops, and the lowering pipeline.** Optimizations are noted but deferred.
- **Plan for NxN even though vec_add is still 1×1.** The current code hardcodes 1×1 — a redesign that does the same is no redesign at all.

### Non-goals

- **Not a final spec.** No tests, no implementation plan, no migration playbook. Those come *after* you pick a sketch (or a hybrid).
- **Not committing to op names.** Anything in `csl.*` / `csl_rt.*` is renamable; the sketches use suggestive names (`csl_program.*`, `csl_layout.*`, `csl_host.*`) but those are placeholders for discussion.
- **Not changing the textual emit philosophy.** We will keep emitting `.csl` source files plus a `run.py` host runner. The question is *how the IR feeds those emitters*, not whether to switch to LLVM.
- **No optimization design.** §11 is a forward-pointer, not a plan.
- **No DSD/task design yet.** vec_add doesn't need them. They are listed as "must reserve op-space for" and we'll design them when the second example demands it.

---

## 2. Background — AIE dialect family at a glance

mlir-aie is the closest existing toolchain to what we're building. Same shape: high-level region IR (AIR) → tile-level structural IR (AIE) → external compiler/runtime artifacts. The AIE dialect *family* has five dialects, four of which matter to us:

| Dialect | Namespace | Purpose | What we should learn from it |
|---|---|---|---|
| **AIE** | `aie.*` | Structural hardware IR — tiles, cores, mems, buffers, locks, dmas, switchboxes, objectfifos | The clean separation of "where things live" from "what runs there" |
| **AIEX** | `aiex.*` | Host-runtime extensions — `aiex.runtime_sequence`, `aiex.npu.dma_memcpy_nd`, `aiex.npu.dma_wait` | How a host program is encoded *next to* the device program in one symbol table |
| **AIEVec** | `aievec.*` | Vector compute IR, lowering target inside `aie.core` | How to keep kernel ops high-level until the very last lowering |
| **XLLVM** | `xllvm.*` | Xilinx-specific LLVM intrinsics | How to bottom out on hardware-specific intrinsics without polluting AIE |
| ADF | `ADF.*` | Adaptive dataflow graphs | Not directly relevant — orthogonal frontend |

Source: `mlir-aie/include/aie/Dialect/{AIE,AIEX,AIEVec,XLLVM,ADF}/IR/*.td`.

### 2.1 AIE op categories (the parts that matter)

I'm pulling out only the ops that influence our redesign. The full table is enormous.

| Category | Op | What it represents | Structural / Behavioral |
|---|---|---|---|
| Device | `aie.device(<arch>)` | Toplevel symbol-table containing everything for one chip | structural |
| Tile | `aie.tile(col, row)` | A physical processing/mem/shim tile at (col,row) | structural |
| Tile (logical) | `aie.logical_tile` | Unplaced tile, refined later by a placement pass | structural |
| Core | `aie.core(%tile) { ... aie.end }` | Per-tile compute body. Region holds plain MLIR (memref/arith/scf/vector) | structural wrapper, behavioral region |
| Memory module | `aie.mem(%tile) { ... }` | Region holding tile-local DMA channels and BD chains | structural |
| Buffer | `aie.buffer(%tile) : memref<...>` | A static tile-memory allocation | structural |
| Lock | `aie.lock(%tile, id, init)` | A tile-local synchronization counter (binary on AIE1, counting on AIE2) | structural |
| Lock use | `aie.use_lock(%lock, Acquire/Release, val)` | Acquire/release inside core or DMA region | behavioral |
| DMA start | `aie.dma_start(MM2S/S2MM, ch, ^bd, ^end)` | Begins a DMA channel inside `aie.mem` | structural+behavioral mix |
| DMA BD | `aie.dma_bd(%buf, off, len)` | One buffer descriptor in a DMA chain | behavioral |
| Flow | `aie.flow(%src, srcBundle, srcCh, %dst, dstBundle, dstCh)` | A logical point-to-point routed connection between tiles | structural |
| Switchbox | `aie.switchbox(%tile) { aie.connect ... }` | Per-tile switch matrix programming (lower-level than flow) | structural |
| ObjectFIFO (decl) | `aie.objectfifo @name(%producer, {%consumers}, depth) : !aie.objectfifo<memref<...>>` | High-level circular buffer between tiles | structural |
| OF acquire | `aie.objectfifo.acquire @name(Produce/Consume, n)` | Acquire `n` slots, returns subview | behavioral |
| OF release | `aie.objectfifo.release @name(Produce/Consume, n)` | Release `n` slots | behavioral |
| Shim DMA alloc | `aie.shim_dma_allocation @name { tile, dir, ch }` | Symbol that names a shim-DMA channel for runtime programming | structural |
| Runtime sequence | `aie.runtime_sequence(%args...) { aiex.npu.* ... }` | Host-side instruction stream that the NPU controller executes at kernel launch | structural wrapper, runtime body |
| NPU memcpy | `aiex.npu.dma_memcpy_nd(%memref[off][size][stride]) { metadata = @symbol }` | Reprogram a shim DMA at runtime to push n-d data | runtime |
| NPU wait | `aiex.npu.dma_wait { symbol = @... }` | Block until that shim DMA finishes | runtime |

### 2.2 Anatomy of a small AIE program

This is the cleanest explanatory snippet — a passthrough from `mlir-aie/test/aiecc/cpp_aie2p_target.mlir`:

```mlir
aie.device(npu2) {                                       // (1) device
  %tile_0_0 = aie.tile(0, 0)                             // (2) shim tile (col,row)
  %tile_0_2 = aie.tile(0, 2)                             // (2) compute tile

  aie.objectfifo @in (%tile_0_0, {%tile_0_2}, 2 : i32)   // (3) high-level data movement
                : !aie.objectfifo<memref<32xi32>>
  aie.objectfifo @out(%tile_0_2, {%tile_0_0}, 2 : i32)
                : !aie.objectfifo<memref<32xi32>>

  %core = aie.core(%tile_0_2) {                          // (4) kernel body for compute tile
    %sv_in  = aie.objectfifo.acquire @in (Consume, 1)
              : !aie.objectfifosubview<memref<32xi32>>
    %el_in  = aie.objectfifo.subview.access %sv_in[0] : ... -> memref<32xi32>
    %sv_out = aie.objectfifo.acquire @out(Produce, 1) : ...
    %el_out = aie.objectfifo.subview.access %sv_out[0] : ... -> memref<32xi32>

    scf.for %i = %c0 to %c32 step %c1 {                  // (5) plain MLIR — memref/scf/arith
      %v = memref.load %el_in [%i] : memref<32xi32>
      memref.store %v, %el_out[%i] : memref<32xi32>
    }
    aie.objectfifo.release @in (Consume, 1)
    aie.objectfifo.release @out(Produce, 1)
    aie.end
  }

  aie.runtime_sequence(%arg_in : memref<32xi32>,         // (6) host program — separate region
                       %arg_out: memref<32xi32>) {
    aiex.npu.dma_memcpy_nd(%arg_out[%c0,%c0,%c0,%c0]
                                  [%c1,%c1,%c1,%c32]
                                  [%c0,%c0,%c0,%c1])
      { metadata = @out, id = 1 : i64 } : memref<32xi32>
    aiex.npu.dma_memcpy_nd(%arg_in [%c0,%c0,%c0,%c0]
                                  [%c1,%c1,%c1,%c32]
                                  [%c0,%c0,%c0,%c1])
      { metadata = @in,  id = 0 : i64, issue_token = true } : memref<32xi32>
    aiex.npu.dma_wait { symbol = @out }
  }
}
```

This is the critical example for our redesign. Notice:

1. Everything is inside one `aie.device` symbol table.
2. Tiles are SSA values (`%tile_0_2 = aie.tile(0,2)`), and **resources reference tiles via SSA use-def**, not by attribute lookup.
3. The kernel body inside `aie.core` is *almost* plain MLIR — only `aie.objectfifo.{acquire,release,subview.access}` are AIE ops. Even those carry no coordinates: they reference `@in`/`@out` symbols, and the symbol is defined separately at device scope.
4. The host program (`aie.runtime_sequence`) is a *sibling* of the cores, not a sibling of the device. It takes host-side memrefs as block args and references DMA channels by symbol.
5. Routing is *implicit* here — no `aie.flow` or `aie.switchbox` appears. The objectfifo elaboration pass will materialize them later.

### 2.3 Separation of concerns in AIE — the answer to each question

| Concern | Where it lives in AIE |
|---|---|
| Spatial placement (col,row) | `aie.tile(col, row)` — only here |
| Resource→tile binding | SSA operand: every `aie.buffer`/`aie.lock`/`aie.core`/`aie.mem` takes the tile result as an operand |
| Routing (high-level) | `aie.flow`, `aie.objectfifo`, `aie.packet_flow` |
| Routing (low-level) | `aie.switchbox` + `aie.connect`, generated by passes from the above |
| Kernel compute | Inside `aie.core { }` — plain `scf/memref/arith/vector` plus a few thin wrappers |
| Host runtime | `aie.runtime_sequence` region, sibling of cores, with `aiex.npu.*` body |
| Cross-tile sync | `aie.lock` + `aie.use_lock`, or implicitly via `aie.objectfifo.acquire/release` |

The kernel body never names a column, a row, a DMA channel, a switchbox port, a lock ID, or a buffer address. All of those are decided by passes between the user-facing IR and the final config. **This is the discipline we want to copy.**

---

## 3. Background — How AIE leaves MLIR

Important for our design because Cerebras output is *not* LLVM — we emit textual `.csl` and a Python runner. We need to know if AIE faces the same issue and how it solves it.

### 3.1 The two-track exit

AIE has **two distinct exit paths from MLIR** that are joined at the very end by `aiecc.py`:

1. **Kernel core path → LLVM IR → object file.** `aie.core` bodies are extracted by `AIECoreToStandard`, lowered through standard MLIR (memref → llvm, arith → llvm, etc.) and `AIEVecToLLVM`, then translated to LLVM IR and compiled by either `xchesscc` (the proprietary Chess compiler) or `peano` (the open-source LLVM-based Vitis AIE backend) into ELF.
2. **Structural config path → emitted artifacts.** Tiles, locks, buffers, DMAs, switchboxes, flows, objectfifo-elaborated state — these are *not* lowered to LLVM. They are visited by **emitters** in `mlir-aie/lib/Targets/` that produce other artifacts.

The third "thing" — the host runtime sequence — has its own emitter that turns it into a packed binary instruction stream the NPU controller executes.

### 3.2 The targets table

Here are the `aie-translate` targets that matter for our pattern-matching against textual `.csl`:

| Translate target | Output | What it really is |
|---|---|---|
| `--aie-generate-xaie` | `aie_inc.cpp` | **C++ source** containing XAIE driver API calls (`XAie_TileConfigureTile`, `XAie_LockInit`, `XAie_DmaWriteAddressDiffBd`, `XAie_StrmConnCctPktRoute`). This is the closest analog to our textual `.csl` emit — AIE *does* emit human-readable target source code. |
| `--aie-generate-cdo` | binary blob (CDO) | A packed sequence of register writes, conceptually the same content as the C++ above but for direct loading instead of compilation |
| `--aie-npu-to-binary` | binary | The host runtime sequence translated to a packed `uint32_t[]` instruction stream for the NPU controller |
| `--aie-generate-ldscript` | linker script | For linking the per-core ELFs |
| `--aie-generate-corelist` | Python list | Names + ELF paths of cores (consumed by aiecc) |
| `--aie-flows-to-json` | JSON | Flow connectivity for analysis/visualization |
| `--aie-generate-mmap` | text | Per-tile memory map, human readable |

Source: `mlir-aie/lib/Targets/AIETargets.cpp:169-413`, especially `AIETargetXAIEV2.cpp` (the `aie_inc.cpp` emitter).

### 3.3 Lessons for the CSL emit

- **AIE genuinely does emit textual target source code** — the `aie_inc.cpp` from `--aie-generate-xaie` is "C++ that, when compiled, programs the array." This is the same pattern we use for `.csl`. The pattern is *legitimate*, not hacky.
- **One MLIR module → many emitter targets.** Each emitter is a focused walker. AIE has a target for the kernel ELF list, another for the structural config, another for the runtime instruction stream. Today our `CSLRuntimeToPy.cpp` is *one* file that does **all three** (LayoutEmitter, KernelEmitter, HostEmitter) in 933 lines. We can split it.
- **The emitter is fed a clean IR.** By the time `--aie-generate-xaie` runs, all the high-level abstractions (objectfifo, herds, flows) have been *elaborated* into low-level concrete ops (buffers, locks, DMA BDs, switch connections) by passes. The emitter is a near-mechanical printer. This is the inversion of "make a smart emitter" — instead, **make smart passes that produce a dumb-but-rich IR, then have a dumb emitter**. The current CSL emitter is doing way too much work because the IR it consumes is too thin.
- **No Python harness from AIE.** AIE stops at xclbin/PDI + per-core ELF and assumes the user provides their own host runner. We want to keep emitting `run.py` because Cerebras' workflow expects it, but we should treat it as an *optional* downstream emitter, not part of the core IR design.

### 3.4 The seam to external compilers

`aiecc.py` orchestrates: opt passes → translate (multiple targets) → external compile (xchesscc/peano) → link (xbridge/llvm-link) → CDO/PDI assembly → xclbin packaging. The seam where MLIR ends and external tools begin is the *core ELF compile* and the *xclbin packager*.

Our analog: MLIR opt passes → translate (`--emit-csl-rt`) → `cslc` (Cerebras' compiler) → `cs_python run.py` (the runner). The seam is at `cslc`.

---

## 4. Background — The AIR → AIE conversion

This is the most directly relevant prior art for AIRToCSL. The pass we care about is `air-to-aie` in `mlir-air/mlir/lib/Conversion/AIRToAIEPass.cpp`. Pre-conditions and the conversion pipeline:

### 4.1 Pre-AIRToAIE passes

Before `-air-to-aie` runs, the AIR module has typically been through:

1. `air-par-to-launch` / `air-par-to-segment` / `air-par-to-herd` — turn loop nests into the AIR hierarchy
2. `air-copy-to-dma` — turn `memref.copy` into `air.dma_memcpy_nd`
3. (Optional) a placement-hint pass that sets `x_loc`/`y_loc` attrs on `air.herd` ops
4. Affine/SCF canonicalization

### 4.2 What AIRToAIE actually does

| AIR op | Becomes in AIE | Decided where |
|---|---|---|
| `air.launch` | (host-side dispatch context; no direct AIE op) | The launch's iteration space becomes context for segments. There is no `aie.launch`. |
| `air.segment` | An `aie.device` + memtile assignments for L2 buffers | `L2MemrefToMemTileMap` groups L2 memrefs by their owning `air.channel` and round-robins groups across memtiles |
| `air.herd` (shape `[Cx, Ry]`) | `Cx*Ry` `aie.tile` ops + one `aie.core` per tile | `outlineAIECores` (lines 182–440), with placement = `(x + col_offset, y + row_offset)` from herd attrs `x_loc`/`y_loc` or pass options `-col-offset`/`-row-offset` (default 1,1) |
| Herd body | Cloned into each `aie.core`, with herd block args **constant-folded** to that tile's `(x, y)` | `outlineAIECores` builds an `IRMapping` that replaces `%tx`/`%ty` with `arith.constant` and clones each op of the herd body |
| `memref.alloc` (mem space 2, inside herd) | `aie.buffer(%tile)` on the owning herd's tile | `AllocL1BuffersPattern` |
| `memref.alloc` (mem space 1, inside segment) | `aie.buffer(%memtile)` on the assigned memtile | `AllocL2BuffersPattern` |
| `air.channel @ch` | If `use-objectfifo=true`: `aie.objectfifo @air_ch(...)`. Otherwise: `aie.lock` pair + `aie.flow` + DMA BDs | `LowerAIRChannelsPattern::matchAndRewrite` (lines 1117–1280) |
| `air.channel.put @ch` (inside core) | `aie.objectfifo.acquire(Produce)` + ... + `release(Produce)` (or DMA BD + lock with the lock-based path) | same pattern |
| `air.channel.get @ch` (inside core) | symmetric to above for Consume | same pattern |
| `air.dma_memcpy_nd` (L3↔L2 or L2↔L1) | `aie.shim_dma_allocation` + BD chain (or memtile DMA) + `aiex.npu.dma_memcpy_nd` in the runtime sequence | `lowerAIRMemcpyOp` |
| `air.execute { ... }` / `air.wait_all` | inlined / sequentialized | unwraps async tokens |

### 4.3 The two big lessons from AIRToAIE

**(a) Placement is decided inside the conversion, but it is not a *heuristic*** — it's a literal mapping from the herd's logical grid to a physical tile rectangle, governed by the herd's `x_loc/y_loc` attrs (which a *prior* pass can set if it wants smarter placement) plus pass-option offsets. The conversion is dumb; the smarts are upstream and optional. **This is the model we should copy.** Our redesign should not bake "smart placement" into AIRToCSL — it should accept a `csl_layout`-shaped placement directive and lower mechanically.

**(b) Routing is decided implicitly by the choice of objectfifo or lock/flow.** With `use-objectfifo=true`, AIRToAIE produces objectfifo declarations and the *real* routing is materialized later by `AIEObjectFifoStatefulTransformPass`. With the older lock/flow path, AIRToAIE itself produces explicit `aie.flow` ops. **Either way, no router runs inside AIRToAIE itself.** AIRToAIE produces declarative routing, and a *separate, single-purpose pass* materializes physical routes. We should adopt the same "AIRToCSL emits high-level routes; a CSL-internal pass elaborates them" structure rather than letting AIRToCSL be a kitchen sink.

### 4.4 The minimal AIR-to-AIE walked example

Input (AIR):

```mlir
func.func @foo(%arg0: i32) {
  %c1 = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %c1, %size_y = %c1) {
    %src0 = memref.alloc() : memref<1xi32, 2>     // L1 buffer in herd
    %src1 = memref.alloc() : memref<1xi32, 2>
    %dst0 = memref.alloc() : memref<1xi32, 2>
    %z = arith.constant 0 : index
    %0 = memref.load %src0[%z] : memref<1xi32, 2>
    %1 = memref.load %src1[%z] : memref<1xi32, 2>
    %2 = arith.addi %0, %1 : i32
    memref.store %2, %dst0[%z] : memref<1xi32, 2>
  }
  return
}
```

Output (AIE, after `-air-to-aie`):

```mlir
aie.device(xcvc1902) {
  %0 = aie.tile(1, 1)                             // 1×1 herd at default offset (1,1)
  %1 = aie.buffer(%0) {sym_name = "buf0"} : memref<1xi32, 2>
  %2 = aie.buffer(%0) {sym_name = "buf1"} : memref<1xi32, 2>
  %3 = aie.buffer(%0) {sym_name = "buf2"} : memref<1xi32, 2>

  %4 = aie.core(%0) {
    %c0 = arith.constant 0 : index                // %tx, %ty constant-folded to 0
    %5 = memref.load %1[%c0] : memref<1xi32, 2>   // refs the AIE buffer, not the original alloc
    %6 = memref.load %2[%c0] : memref<1xi32, 2>
    %7 = arith.addi %5, %6 : i32
    memref.store %7, %3[%c0] : memref<1xi32, 2>
    aie.end
  } {sym_name = "herd_0"}
}
```

What changed: the herd is gone, replaced by tiles + cores; `memref.alloc` results are replaced by `aie.buffer` results pinned to the tile; herd block args become `arith.constant`s. The *body of the core is otherwise unchanged* — still plain `memref.load`/`arith.addi`/`memref.store`. This is the cleanliness target.

---

## 5. Background — Cerebras CSL programming model

Sourced from <https://sdk.cerebras.net/csl/language_index>, <https://sdk.cerebras.net/csl/tutorials/>, <https://sdk.cerebras.net/csl/tutorials/gemv-01-complete-program/>, <https://sdk.cerebras.net/csl/tutorials/gemv-05-multiple-pes/>, and <https://sdk.cerebras.net/csl/language/dsds>. The deeper routing pages were partially unreachable; details on `@get_color`/`@set_color_config`/route syntax need a follow-up fetch.

### 5.1 Execution model in one paragraph

Cerebras' WSE is a 2D fabric of hundreds of thousands of PEs. Each PE runs **its own program** — there is no implicit SPMD broadcast. PEs that happen to share the same source file behave SPMD-like, but that is a *layout* decision. Per-PE programs are parameterized via comptime params handed in at `@set_tile_code` time. PEs communicate only by sending **wavelets** (32-bit data words) along **colors** that have been wired by the layout. Per-PE memory is small (~48 KB SRAM); there is no shared address space.

### 5.2 The three-file structure

A complete CSL program is **three files**:

1. **`layout.csl`** — declarative spatial top-level. Calls `@set_rectangle(W, H)`, then `@set_tile_code(x, y, "pe_program.csl", .{ ... params ... })` for each PE, plus `@export_name(...)` for host-visible symbols and host-callable RPC entry points.

2. **`pe_program.csl`** — the PE program (one or more files). Imports `<memcpy/memcpy>`, declares `param`s, `var`s, plain `fn`s, and **tasks**. Ends with a `comptime { @export_symbol(...) }` block that publishes symbols to the host.

3. **`run.py`** — the host runner. Uses `cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime`. Calls `runner.load()`, `runner.run()`, `memcpy_h2d`, `launch`, `memcpy_d2h`, `stop`.

This three-file structure is **not** a quirk of the SDK — it is the actual semantic factoring of CSL. Our IR needs to model these three things distinctly, because *every* CSL program has them, and they have very different concerns.

### 5.3 Single-PE GEMV example (what vec_add looks like in real CSL)

`layout.csl` (~10 lines):

```csl
const memcpy = @import_module("<memcpy/get_params>", .{ .width = 1, .height = 1 });

layout {
  @set_rectangle(1, 1);
  @set_tile_code(0, 0, "pe_program.csl",
    .{ .memcpy_params = memcpy.get_params(0) });
  @export_name("y", [*]f32, false);
  @export_name("init_and_compute", fn()void);
}
```

`pe_program.csl` (~25 lines, abbreviated):

```csl
param memcpy_params: comptime_struct;
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

const M: i16 = 4; const N: i16 = 6;
var A: [M*N]f32;  var x: [N]f32;  var b: [M]f32;  var y: [M]f32;
const y_ptr: [*]f32 = &y;

fn gemv() void {
  for (@range(i16, M)) |i| {
    var tmp: f32 = 0.0;
    for (@range(i16, N)) |j| { tmp += A[i*N + j] * x[j]; }
    y[i] = tmp + b[i];
  }
}

fn init_and_compute() void { initialize(); gemv(); sys_mod.unblock_cmd_stream(); }

comptime {
  @export_symbol(y_ptr, "y");
  @export_symbol(init_and_compute);
}
```

`run.py` (~25 lines): standard `SdkRuntime` boilerplate, exactly like the one our `HostEmitter` currently produces.

### 5.4 Key concepts for IR design

| CSL concept | What it means | What an IR has to represent |
|---|---|---|
| Rectangle | The (W, H) sub-grid of PEs the program occupies | A top-level structural attribute on the program |
| `@set_tile_code(x, y, file, params)` | Bind a per-PE source file with comptime params | Placement op: `csl_layout.place(@kernel, x, y, params)` |
| Comptime param | Compile-time constant passed from layout to PE | Attribute on the placement op, not an SSA value |
| `var` / `const` | PE-local memory (SRAM) | A symbol inside the PE program — declarative, not SSA-driven |
| `fn` | Plain CSL function | A function symbol inside the PE program |
| Task | Event-driven handler bound to a color | A function-like symbol with a `bind_color` attribute |
| Color | One of 24 routable IDs for inter-PE wavelets | A finite-numbered resource. IR should **not** assign IDs — that's a pass |
| Route | The path a color takes through switch fabric | A first-class layout-level relation |
| DSD (mem1d/mem4d/fabin/fabout/fifo) | Strided memory or stream descriptor consumed by builtins | A first-class type in the PE-program dialect |
| `@export_symbol` | Make a PE-local symbol host-visible | A symbol-table attribute *on the kernel* |
| `@export_name` | Declare a host-visible name (with [*]T type and mutability) | A symbol *on the layout*, referencing kernel exports |
| `<memcpy/memcpy>` library | Per-PE host-RPC + bulk H2D/D2H | An implicit dependency that the PE program imports — IR doesn't need to model it explicitly, just inject the import in the emitter |
| `SdkRuntime.{load,run,launch,memcpy_h2d,memcpy_d2h,stop}` | Host-side orchestration | The host dialect (`csl_host.*` in the redesign) |

### 5.5 CSL ↔ AIE vocabulary (full table)

| CSL term | Closest AIE concept | Notes |
|---|---|---|
| PE | `aie.tile` + `aie.core` | One-to-one |
| Rectangle | `aie.device` region | The "everything lives here" wrapper |
| `@set_tile_code` | implicit: `aie.core(%tile)` body bound by SSA | AIE has no source-file binding; bodies are inline. CSL splits compile units per PE. |
| Comptime params | Function args / attributes on the core (no exact analog) | This is a *real* difference; AIE has no comptime layer because each core has its own LLVM compile |
| Color | Roughly `aie.flow` source/dest channel | But CSL has only 24 colors and they trigger tasks; AIE flows are basically routing-only |
| Wavelet | One beat on an AIE stream | 32-bit each |
| Task (data/local/control) | No exact analog — closest is DMA-completion-driven core code | This is one of the biggest semantic gaps. We need to reserve op-space. |
| `@bind_data_task` | Connecting a flow input to a core's input port (loose) | |
| DSD | `memref` + `aie.dma_bd` blended into one value | DSD is a *value*, not a setup-only thing. This is closer to a fused `memref + linalg.generic` operand. |
| `<memcpy/memcpy>` | `aie.shim_dma_allocation` + `aiex.runtime_sequence` + `aiex.npu.dma_memcpy_nd` | Host RPC + bulk transfer combined |
| `@export_symbol` | `aie.shim_dma_allocation` (for buffers) — no clean analog for host-callable RPC | |
| `@export_name` | A combination of shim_dma_allocation + a runtime_sequence entry | |
| `SdkRuntime.launch` | `aiex.npu.*` runtime sequence kickoff | RPC into PE function |

### 5.6 Differences worth flagging in the redesign

- **Per-PE source files as a unit of compilation.** AIE's "one device, many cores in one module" doesn't quite map. Each PE in CSL has its own `.csl` file with its own comptime params. The IR should explicitly model "PE program = a module with comptime parameters at its boundary."
- **Comptime parameterization is everywhere.** Width/height, per-PE column index, shapes — these are baked into the source via comptime params, not via runtime SSA values. The IR should support attribute-driven specialization at the PE-module boundary.
- **Colors are a constrained resource (24 IDs).** Unlike AIE flows, color allocation is a real constraint. The IR should declare colors symbolically and have a pass assign physical IDs.
- **Tasks are event-driven, not just data-flow.** Three flavors (data/local/control). AIE has no equivalent abstraction. The IR needs a task op even if vec_add doesn't use it.
- **No global address space.** Cross-PE access is wavelet-only. The IR must not pretend a memref can span PEs.
- **Textual layout file vs. structural MLIR.** This is actually a *match* with how CSL is written — but our IR can be structural and *lower* to the textual form. The key is that the layout level is naturally `scf.for` over symbolic PE coordinates, unrolled at codegen.

---

## 6. The current CSL + CSLRT design

This is what we have today, sourced from `mlir-air/mlir/{include,lib}/air/Dialect/{CSL,CSLRuntime}/`, `mlir/lib/Conversion/{AIRToCSLDialect,CSLToCSLRuntime}/`, and `mlir/lib/Targets/CSLRuntimeToPy.cpp`.

### 6.1 Top-level shape

Two dialects, intentionally split:

| Dialect | Namespace | Purpose | Files |
|---|---|---|---|
| **CSL** | `csl.*` | PE-level kernel compute, spatial placement, routing/colors, data structures | `mlir/include/air/Dialect/CSL/` (`CSLBase.td`, `CSLOps.td`, `CSLDialect.h`, `CSLOps.h`) |
| **CSLRuntime** | `csl_rt.*` | Host-side `SdkLayout`/`SdkRuntime` API operations | `mlir/include/air/Dialect/CSLRuntime/` (`CSLRuntimeBase.td`, `CSLRuntimeOps.td`, ...) |

The split is the right *idea* — "device IR" vs "host IR". The problem is in execution (§7).

### 6.2 CSL dialect ops

| Category | Op | Region | Purpose |
|---|---|---|---|
| Routing | `csl.color` | none | Declare a comm color, optional ID |
| Routing | `csl.route` | none | Declare an input→output direction route |
| Layout | `csl.spatial_placement` | sized region | Top-level container for everything device-side |
| Layout | `csl.code_region` | sized region | A "PE grid region" with width, height, routes, colors. Body is documented to hold per-PE config; **never populated** today. |
| Layout | `csl.place` | none | `place %region %kernel { x, y }` |
| Data movement | `csl.dataflow` | none | Connect two ports (declared but unused in vecadd) |
| PE kernel | `csl.kernel` | sized region (SymbolTable) | PE program container, attrs `source_file` + `params` |
| PE kernel | `csl.func` | any region | A PE function |
| PE kernel | `csl.task` | any region (IsolatedFromAbove) | A PE task bound to a color |
| PE kernel | `csl.return` | none | terminator |
| PE memory | `csl.var` | none | PE-local buffer, **symbol-named SSA result** |
| PE module | `csl.import_module` | none | `@import_module(...)` |
| Host iface | `csl.export_name` | none | Host-visible export (sits in host func, with attrs `direction = "in"/"out"`) |
| Host iface | `csl.export_symbol` | none | `@export_symbol` declaration inside a kernel, with optional alias |
| (Deferred) | `csl.get_mem_dsd`, `csl.get_fab_dsd`, `csl.mov` | — | DSD ops, commented out in `CSLOps.td` |

Custom types: `!csl.color`, `!csl.dsd`, `!csl.imported_module`, `!csl.kernel`, `!csl.route`, `!csl.code_region`, `!csl.port`, `!csl.stream`. Custom enums: `csl.Direction` (NORTH/SOUTH/EAST/WEST/RAMP), `csl.DsdKind` (mem1d/mem2d/fabin/fabout), `csl.Edge` (LEFT/RIGHT/TOP/BOTTOM).

### 6.3 CSLRuntime dialect ops

| Op | Purpose |
|---|---|
| `csl_rt.create_layout` | New `SdkLayout` |
| `csl_rt.create_code_region` | Add region to layout |
| `csl_rt.place` | Place region at (x, y) |
| `csl_rt.set_param_all` | Set comptime param on region |
| `csl_rt.export_name` | Export symbol from layout |
| `csl_rt.compile` | Compile layout → artifacts |
| `csl_rt.runtime_create` | New `SdkRuntime` |
| `csl_rt.load` / `csl_rt.run` / `csl_rt.stop` | runtime lifecycle |
| `csl_rt.get_id` | Look up a symbol id |
| `csl_rt.memcpy_h2d` / `csl_rt.memcpy_d2h` | Bulk transfers |
| `csl_rt.launch` | RPC into device function |

Custom types: `!csl_rt.{layout,code_region,compile_artifacts,runtime,color,routing_position,port,stream}`.

### 6.4 The pipeline today (vec_add)

```
vecadd.mlir (AIR)
  │
  │  -air-to-csl-dialect              (AIRToCSLDialect.cpp, 407 lines)
  ▼
mixed-1: csl.spatial_placement { csl.kernel {...} csl.code_region {} csl.place }
         + host-level csl.export_name ops
  │
  │  -csl-to-csl-rt                   (CSLToCSLRuntime.cpp, 158 lines)
  ▼
mixed-2: same csl.* ops (untouched)
         + new csl_rt.create_layout / compile / runtime_create / memcpy_h2d / launch / memcpy_d2h ops
  │
  │  air-translate --emit-csl-rt      (CSLRuntimeToPy.cpp, 933 lines)
  ▼
$OUT/layout.csl     ← LayoutEmitter walks csl.spatial_placement
$OUT/vecadd_pe.csl  ← KernelEmitter walks csl.kernel body
$OUT/run.py         ← HostEmitter walks csl_rt.* ops in host func
```

This is the most important diagram in this document. Notice:

- **`csl-to-csl-rt` does not consume `csl.*` ops.** It adds `csl_rt.*` ops alongside them. The output module has *both* dialects, mixed in the same functions.
- **The translate step has three independent emitters in one file.** They each scan the same module looking for the ops they care about. There is no single rooted walk.
- **Direction information flows the wrong way.** Direction is on the *host-level* `csl.export_name`, but the *kernel* needs to know it (to decide if a buffer is `var` or `const`). `KernelEmitter` builds a `hostExportDir` map by scanning host code, then back-patches the kernel emit. (See `CSLRuntimeToPy.cpp:613-620`.)

### 6.5 The honest pain points (to be fair, with severity)

Critical (block multi-PE / multi-arg):

1. **Hardcoded 1×1 in AIRToCSLDialect.cpp**: `width=1, height=1, x=0, y=0` baked in at `AIRToCSLDialect.cpp:310/320`. Validates that herds are `1×1` only at line 72–82.
2. **Hardcoded arg-naming for exactly 3 args** at `AIRToCSLDialect.cpp:223-230`: `const char *names3[] = {"a_buf","b_buf","c_buf"}`. Anything else falls into a generic `buf_<n>` fallback that doesn't match what the host emitter expects.
3. **Hardcoded test data + validator in `HostEmitter`** at `CSLRuntimeToPy.cpp:796-880`: only works for vec_add (`a = arange; b = arange*2; expected = a+b`). Not a generalizable host emitter — it's an example printer.
4. **`csl.code_region` body documented but never used** (`CSLOps.td:156-158`). Routes/colors are operands but never lowered — there's no pass that materializes them into anything.

Severe (block reuse / clean composition):

5. **Direction split between kernel and host.** `csl.export_symbol` (kernel) and `csl.export_name` (host) carry pieces of the same fact in different places. The emitter reconciles them by scanning. A kernel cannot be reused with different host directions.
6. **`csl.var` is symbol-named SSA.** The op produces an SSA value but the name lives in a symbol attribute, and the printer uses `getAsmResultNames` to align them (`CSLOps.cpp:26-29`). The kernel emitter walks SSA→string by hand. Naming should be either symbol or SSA, not both.
7. **`csl.kernel { source_file = "vecadd_pe.csl" }`** binds the IR to one emission strategy (one kernel → one named `.csl` file). That's a codegen artifact bleeding into IR semantics.
8. **CSLToCSLRuntime is additive, not transformative.** It adds `csl_rt.*` ops but leaves all `csl.*` ops in place. The result is a single mixed module that two emitters then walk separately. There is no clean handoff.

Moderate (work but make things hard later):

9. **Op allowlist in `KernelEmitter`** at `CSLRuntimeToPy.cpp:219-229`: 9 op classes hard-coded. Adding `arith.mulf` or `csl.task` requires editing the cpp.
10. **`csl.export_symbol` cannot express "do not export"** — every internal var is exported if there's an export op present.
11. **No verifier checks** for symbol consistency between `csl.export_symbol` aliases and `csl.var` names, or for placement coordinates being inside the rectangle.
12. **Data movement ops (DSDs) are deferred indefinitely.** `csl.get_mem_dsd`, `csl.get_fab_dsd`, `csl.mov` are commented out. There's no plan for how `air.channel` would lower to wavelets/colors when we go multi-PE.

### 6.6 What's actually good (so we don't throw it away)

- **Two-dialect split is the right idea.** Device-side and host-side are genuinely different concerns. AIE does the same with `aie` + `aiex`. We just need to enforce the split.
- **Kernel body holds no coordinates.** This is a real win and should be preserved.
- **`csl.place` separates kernel binding from region.** Kernels can in principle be placed anywhere — the abstraction is right, even if it's only used for (0,0) today.
- **Colors and routes are at region level**, not in the kernel. Right idea; needs to be wired up.
- **`source_file` aside, `csl.kernel` is a clean symbol-table region.** `csl.func` / `csl.task` / `csl.var` / `csl.import_module` / `csl.export_symbol` inside it form a coherent PE-program sub-IR.
- **`CSLToCSLRuntime` exists as a pass at all.** It's a stub today, but the *name* is right — there should be a conversion from device IR to host IR.

---

## 7. Side-by-side: where CSL diverges from AIE's separation of concerns

| Concern | AIE | Today's CSL | Verdict |
|---|---|---|---|
| Device wrapper | `aie.device(<arch>)` (1 per chip) | `csl.spatial_placement` (1 per program) | Same shape — good |
| Tile/PE handle | `%t = aie.tile(c, r)` (SSA) | None — `csl.place` carries x/y as attrs | We have no first-class PE handle. This makes "this buffer lives on PE (1,2)" awkward to express. |
| Tile→resource binding | SSA use-def: `aie.buffer(%t)`, `aie.lock(%t)`, `aie.core(%t)` | All resources are *inside* `csl.kernel`. There is no per-PE structural binding. | We made the kernel the unit, AIE made the tile the unit. Different model — see §9 sketches. |
| Kernel body | `aie.core { plain MLIR + few wrappers }` | `csl.kernel { csl.func { plain MLIR } }` | Same shape — good |
| Routing declaration | `aie.flow` / `aie.objectfifo` at device scope | `csl.color`/`csl.route` as operands of `csl.code_region` | Same shape — good in principle, never elaborated |
| Routing materialization | `AIEObjectFifoStatefulTransformPass` and `AIECreatePathFindFlows` elaborate into low-level ops | No equivalent | Missing — needed for multi-PE |
| Host runtime sequence | `aie.runtime_sequence` region inside `aie.device`, with `aiex.npu.*` body | `csl_rt.*` ops *inside the same host `func.func`* as the device IR | Mixed dialect inside one func; the boundary is convention, not structure |
| Host ↔ device exports | `aie.shim_dma_allocation` symbols referenced by `aiex.npu.dma_memcpy_nd { metadata = @sym }` | `csl.export_symbol` inside kernel *and* `csl.export_name` in host func, name-matched by string | AIE uses symbols; we use string-matched attribute pairs. Symbols are stronger. |
| Direction (in/out) | Implied by the runtime op (`dma_memcpy_nd` direction is determined by the shim_dma_allocation's `MM2S`/`S2MM` flag) | Attribute on `csl.export_name`, *re-derived* by the kernel emitter | We have to scan across IR fragments to learn one fact |
| Address/lock-id/BD-id assignment | Pass: `AIEAssignBuffers`, `AIEAssignLockIDs`, `AIEAssignBufferDescriptorIDs` | None | Missing — needed for multi-PE / DSDs |
| Compile unit per device-side artifact | One per `aie.core` (LLVM IR → ELF), structural config in one `aie_inc.cpp`, runtime in one binary stream | All three textual artifacts (`.csl` × 1, `run.py` × 1) emitted from one mixed module by three walkers in one file | We have no clean "one module → one artifact" mapping |

**The pattern:** AIE consistently uses symbols and SSA to make every relationship explicit, then has dedicated *single-purpose passes* to turn declarative high-level ops into mechanically-emittable low-level ops. We do half of that — we have declarative high-level ops — but we don't have the elaboration passes, and our final emit reads the high-level form directly. That's why the emitter is so smart, the IR is so thin, and the conversion has to hardcode so much.

---

## 8. The redesign question, framed

> "I started with `csl.spatial_placement` because I wanted to separate place, partition, and route from the kernel. Maybe the design decision was right but the implementation was wrong."

I think that's exactly correct, and the analysis in §7 supports it. The design instinct is sound; the execution leaks because:

1. The split is **vague**: what *exactly* is in the kernel vs. what's in the placement layer is decided by the emitter at print time, not by which dialect/op the data lives in.
2. The **direction of information flow** is wrong: the kernel needs to know things (like buffer direction) that live in the host-level layer. We should fix this so that information always flows *outward*: the kernel declares what it has, the layout binds kernels to PEs, and the host references the layout's exports.
3. There's **no elaboration step** between "user-visible declarative IR" and "emit-ready low-level IR." So either (a) the user has to write low-level IR and we lose the abstraction, or (b) the emitter has to be smart enough to elaborate at print time, and we end up with a 933-line emitter file.
4. The **two dialects today aren't really two layers** — they're two namespaces on the same module. A real layered design would have *separate compilation units* and *consume* its predecessor.

These four observations drive the three sketches.

---

## 9. Three redesign sketches

I'll give each one a name, a one-paragraph philosophy, the new op surface, the new pipeline, and what it gets right/wrong. Then a recommendation in §10.

### Sketch A — "Minimal cleanup": stay close to today, fix the leaks

**Philosophy.** The current shape is fine; just stop leaking. Keep one CSL dialect + one CSLRuntime dialect. Fix direction-duplication, symbol-vs-SSA confusion, hardcoded constants, and make `CSLToCSLRuntime` actually transform instead of additively decorate.

**What changes:**
- `csl.kernel { source_file = ... }` loses `source_file`. The kernel is now identified by its symbol name; the emitter decides the filename.
- `csl.var` becomes pure SSA (no symbol attribute). Names fall out of the printer; the emitter uses an MLIR `Namer` for the output `.csl`.
- `csl.export_symbol` keeps its alias but **drops direction**. Direction lives only on `csl.export_name` at the host layer.
- The kernel emitter no longer reads `hostExportDir`. Instead, a new pass `csl-elaborate-exports` runs *after* `CSLToCSLRuntime`: it inspects `csl_rt.memcpy_h2d`/`d2h` and writes a per-export `direction` attribute *back onto the corresponding `csl.export_symbol`* inside the kernel. The emitter reads only the kernel's own attributes.
- `csl.code_region` gets a real body: per-PE config ops (`csl.set_param`, `csl.bind_color`) live there.
- `csl.place` becomes `csl.place(%region, @kernel_sym, x, y, params)` — comptime params are an attribute dict on the place op, not on the kernel. (A kernel can be placed multiple times with different params.)
- A new pass `csl-elaborate-grid` *unrolls* `csl.place` over a (W, H) range when given attributes. This is what makes 1×1 → NxN actually work.
- `AIRToCSLDialect` stops hardcoding `1×1` and `(0,0)`. Instead it reads herd attrs `x_loc`/`y_loc` and pass options `-col-offset`/`-row-offset`, exactly like AIRToAIE.
- `CSLToCSLRuntime` becomes a *consuming* conversion: it removes the `csl.export_name` host ops and replaces them with `csl_rt.*` constructs in a runtime sequence.
- The 933-line `CSLRuntimeToPy.cpp` is split into three files (`LayoutEmitter.cpp`, `KernelEmitter.cpp`, `HostEmitter.cpp`) with a thin dispatcher.

**Pros.**
- Smallest delta from today; the existing tests stay working.
- All the wins are achievable without renaming dialects or moving ops between them.
- Easiest migration plan.

**Cons.**
- Doesn't change the fundamental shape. We still have two dialects living in one module, with the boundary defined by convention.
- The kernel body is still inside `csl.spatial_placement`, which is structurally weird (the placement layer "contains" the kernel rather than referencing it).
- The "elaborate exports backward" pass is itself a hack: information still flows the wrong direction, we just paper over it with a pass.
- `csl.code_region`'s body is still vague — it has to hold both per-PE config and routing, which are different concerns.

### Sketch B — "AIE-style structural mirror": the kernel is per-tile

**Philosophy.** Copy AIE's exact factoring. There is no `csl.kernel`. There are PEs (`csl.pe(col, row)`), each with a body region. Buffers/funcs/tasks live *inside* a PE. SPMD-ness is recovered later by a CSE/dedup pass that finds identical PE bodies and emits one `.csl` file shared by many `@set_tile_code` calls.

**Op surface.**

| Op | Purpose |
|---|---|
| `csl.program(@arch)` | Top-level symbol-table (analog of `aie.device`) |
| `csl.pe(col, row) { ... csl.end }` | A single PE with a body region. SSA result is a `!csl.pe` handle. |
| `csl.var(%pe) : memref<...> { name }` | A PE-local memory, tied to a PE by SSA |
| `csl.func(%pe) @sym { ... }` / `csl.task(%pe) @sym { color = @c }` | Symbols bound to a PE |
| `csl.import(%pe) "<memcpy/memcpy>"` | Per-PE imports |
| `csl.export(%pe) @sym { alias, direction }` | Export from a PE (direction stays here, declared once) |
| `csl.color(%program) @c` | Color symbol at program scope |
| `csl.route(%c) { ... } / csl.flow(%color, %src_pe, %dst_pe)` | Connectivity at program scope |
| `csl.host_sequence(%args...) { csl_host.* }` | Host runtime sibling region (mirrors `aie.runtime_sequence`) |
| `csl_host.*` | Renamed `csl_rt.*`, only used inside `csl.host_sequence` |

**Pipeline.**

```
AIR module
  │ -air-to-csl-program       (analog of -air-to-aie)
  ▼
csl.program {
  csl.pe(0,0) { ...body... }       ← one PE per herd-tile
  csl.color @bcast_x, ...           ← from air.channel
  csl.flow @bcast_x %src_pe %dst_pe ← from air.channel.put/get
  csl.host_sequence(...) { ... }    ← from air.launch + memcpy infrastructure
}
  │ -csl-elaborate-routes     (analog of objectfifo stateful transform)
  ▼
csl.program with concrete per-PE buffers, locks, DSD setups
  │ -csl-dedup-pe-bodies      (SPMD recovery)
  ▼
csl.program with each PE body referencing one of K shared `csl.kernel_template @t`
  │ csl-translate --emit-csl
  ▼
layout.csl, kernel_template_<i>.csl × K, run.py
```

**Pros.**
- The model is a *direct* translation of AIE. Anyone who knows AIE can read this in 5 minutes.
- Tile is the unit. Buffers and funcs are pinned to a PE by SSA, mirroring AIE's `aie.buffer(%tile)`.
- Direction lives in exactly one place (`csl.export(%pe)`), no host/kernel duplication.
- The dedup pass is an honest "SPMD recovery" step that maps the per-PE structural form to the per-source-file textual form CSL expects. SPMD becomes a *codegen detail*, not a model assumption.
- Multi-PE is the *default*. 1×1 is just `csl.pe(0,0)` once.

**Cons.**
- This is a fundamentally larger redesign. Almost every op renamed. Existing AIRToCSL has to be rewritten.
- It doesn't match how Cerebras programs are *written*. Real CSL programs are SPMD-by-template — you write *one* `pe_program.csl` and bind it to a rectangle. Sketch B inverts that ("write per-PE bodies, dedup to recover the template"), which feels backward when the human thinks in templates first.
- The "per-PE comptime param" model (where each PE gets different `x`, `width`, `M`, `N`) is harder to express. With per-PE bodies you'd inline the param values, then dedup pulls them back out — feels indirect.
- Tasks bound to colors are awkward when the color is at program scope but the task is inside a PE — colors are first-class symbols and tasks reference them by name, so it mostly works, but the SSA-operand discipline doesn't extend cleanly.

### Sketch C — "Three-layer split": layout / program / host as separate dialects

**Philosophy.** Take the three CSL files at face value and make each its own dialect. Each layer is its own MLIR module after its lowering pass; the transition between layers consumes the prior. This is the model that best matches how CSL programs are actually structured, and it makes the responsibility of each pass and each emitter unambiguous.

**Three dialects:**

1. **`csl_program` (PE-local kernel program).**
   - Models *one* PE program — what's in a `pe_program.csl` file.
   - Top op: `csl_program.module @pe_kernel { ... params, vars, funcs, tasks, exports ... }` — a symbol-table region with comptime params declared as op attributes.
   - Inside: `csl_program.param @width : i16`, `csl_program.var @y : memref<...>`, `csl_program.func @gemv() { ... plain memref/scf/arith/vector ... }`, `csl_program.task @t { color_sym = @c }`, `csl_program.import "<memcpy/memcpy>"`, `csl_program.export @y_ptr { alias = "y", direction = out }`, `csl_program.export @gemv { kind = func }`.
   - **Has no PE coordinates. No rectangle. No host.** Just one PE program.
   - This is the kernel writer's IR. It's the only IR that can contain compute.

2. **`csl_layout` (rectangle + placements + routing).**
   - Models a `layout.csl` file — the `@set_rectangle` + `@set_tile_code` + `@export_name` block.
   - Top op: `csl_layout.layout (W, H) @program_name { ... places, colors, routes, exports ... }`.
   - Inside: `csl_layout.color @c (id?)`, `csl_layout.route @r { src, dst, colors }`, `csl_layout.place @kernel_sym at (x, y) { params = #attrs }`, `csl_layout.export @sym from @kernel_export { type = !memref<...>, mutable }`.
   - For NxN, `csl_layout.place_grid @kernel from (x0,y0) to (x1,y1) { params = #per-pe-attrs }` is a single op that elaborates into individual `csl_layout.place` ops by a `csl-layout-elaborate` pass.
   - **References programs by symbol.** A `csl_layout.layout` can reference one or many `csl_program.module`s.
   - **Has no compute.** Just rectangle, placement, routing, exports.

3. **`csl_host` (host runner: SdkRuntime orchestration).**
   - Models a `run.py` file — but in MLIR, so we can analyze it.
   - Top op: `csl_host.runner @main(%args : memref<...>...) { ... } { layout = @layout_sym }`.
   - Inside: `csl_host.load`, `csl_host.run`, `csl_host.memcpy_h2d %arg to @export_sym { rectangle = (...) }`, `csl_host.launch @export_sym`, `csl_host.memcpy_d2h @export_sym to %arg { rectangle = (...) }`, `csl_host.stop`.
   - **References the layout and its exports by symbol.** It does not look inside the kernel.
   - This is the renamed/refined `csl_rt`.

**The pipeline becomes three independent compile units linked by symbols:**

```
AIR module
  │
  │  -air-to-csl-program     (extracts the herd body into a csl_program.module)
  │  -air-to-csl-layout      (extracts the herd shape into a csl_layout.layout, plus place/route)
  │  -air-to-csl-host        (extracts launch/memcpy into a csl_host.runner)
  ▼
Three submodules in one outer module:
  module {
    csl_program.module @vecadd_pe { ... }
    csl_layout.layout (1, 1) @main_layout {
      csl_layout.place @vecadd_pe at (0, 0)
      csl_layout.export @y from @vecadd_pe::@y_ptr { type = memref<256xf32>, mutable }
      csl_layout.export @compute from @vecadd_pe::@compute { kind = func }
    }
    csl_host.runner @main(%a, %b, %c) { layout = @main_layout } {
      csl_host.load
      csl_host.memcpy_h2d %a to @main_layout::@a
      csl_host.memcpy_h2d %b to @main_layout::@b
      csl_host.launch @main_layout::@compute
      csl_host.memcpy_d2h @main_layout::@c to %c
      csl_host.stop
    }
  }
  │
  │  -csl-elaborate-grid     (only csl_layout: turns place_grid into N place ops)
  │  -csl-allocate-colors    (only csl_layout: assigns IDs to symbolic colors)
  │  -csl-elaborate-dsds     (only csl_program: turns memref accesses into DSD ops where appropriate, future)
  ▼
Same shape, ready for emit
  │
  │  air-translate --emit-csl-program  → vecadd_pe.csl  (one per csl_program.module)
  │  air-translate --emit-csl-layout   → layout.csl
  │  air-translate --emit-csl-host     → run.py
  ▼
$OUT/vecadd_pe.csl  $OUT/layout.csl  $OUT/run.py
```

**Each emitter is single-purpose and reads only its own dialect.** The kernel emitter never sees host info; the layout emitter never sees compute; the host emitter never sees PE-local vars.

**Symbol references thread the layers.** A `csl_host.memcpy_h2d ... to @main_layout::@a` op lets us *verify* statically that `@main_layout::@a` exists and has the right element type. Today's design has no such verification because everything is name-matched at emit time.

**For multi-PE / NxN:**

```mlir
csl_program.module @gemv_pe {
  csl_program.param @M : i16
  csl_program.param @N : i16
  csl_program.param @col_index : i16
  csl_program.var @A : memref<24xf32>      // M*N elements
  ...
  csl_program.func @compute() { ... uses @col_index ... }
  csl_program.export @y_ptr  { alias = "y", direction = out }
  csl_program.export @compute { kind = func }
}

csl_layout.layout (4, 1) @gemv_layout {
  csl_layout.place_grid @gemv_pe
    from (0, 0) to (3, 0)
    params = #csl_layout.per_pe<{M=4, N=6, col_index=$col}>
    // $col is a placeholder for the per-PE column id, expanded by elaborate-grid
  csl_layout.export @y from @gemv_pe::@y_ptr { type = memref<4x4xf32>, mutable, rectangle = (0,0,4,1) }
  csl_layout.export @compute from @gemv_pe::@compute { kind = func }
}

csl_host.runner @main(%y_out : memref<4x4xf32>) { layout = @gemv_layout } {
  csl_host.load
  csl_host.launch @gemv_layout::@compute
  csl_host.memcpy_d2h @gemv_layout::@y to %y_out
  csl_host.stop
}
```

**Pros.**
- The IR mirrors the *actual* shape of CSL programs. There is exactly one `csl_program.module` per emittable `pe_program.csl`. There is exactly one `csl_layout.layout` per emittable `layout.csl`. There is exactly one `csl_host.runner` per emittable `run.py`.
- Each emitter walks one dialect. Splitting `CSLRuntimeToPy.cpp` into three is mechanical, not a redesign.
- Information flow is one-way: kernel declares → layout binds → host references. The kernel never needs to "know" about the host.
- Verification works across layers via symbols. The verifier can check that every `csl_host.memcpy_h2d ... to @export` references a real export.
- 1×1 vs NxN is a difference between `csl_layout.place` and `csl_layout.place_grid` — same dialect, same emitter, no other code paths.
- Per-PE comptime params are a *first-class* attribute on the place op. They naturally support both "all PEs get the same params" and "PE (col, row) gets these params" via the per-PE attribute dict.
- Tasks/colors are at the layer where they belong: tasks are inside `csl_program` (per-PE), colors and routes are inside `csl_layout` (program-wide), and a task references a color by symbol-from-layout — verified at link time.
- Future DSD ops are clearly inside `csl_program`, future routing-elaboration is clearly inside `csl_layout`, future host-side perf knobs are clearly inside `csl_host`. Every concern has a home.
- We can write tests for each layer in isolation. A `csl_program.mlir` test never needs to mention layouts or hosts.

**Cons.**
- The biggest delta from today. Nearly every op moves to a new dialect.
- Three dialects to maintain instead of two.
- The migration story for the existing vecadd test is more involved (though still mechanical).
- We need a clear "outer module" convention for holding the three submodules together. (This is solvable — MLIR supports nested module ops with their own symbol tables — but it's one more thing to design.)
- Symbol references like `@main_layout::@compute` require a small extension to `SymbolRefAttr` (nested symbols) but that is supported by upstream MLIR and used by other dialects.

### A couple of dimensions all three sketches share

Regardless of which sketch we pick:

1. **Direction lives in one place.** Whether that's `csl.export(%pe)` (Sketch B) or `csl_program.export` (Sketch C) or "elaborate-exports adds it to `csl.export_symbol`" (Sketch A), there is only ever one source of truth.
2. **`source_file` is gone from the IR.** The emitter decides the filename. The IR just carries the kernel's symbol name.
3. **Symbolic colors with a separate `csl-allocate-colors` pass.** Don't bake color IDs into the IR; let users name colors and have a pass assign 0–23.
4. **Kernel body remains plain MLIR (`memref` / `scf` / `arith` / `vector`)** plus a small CSL prelude (`csl_program.var`, exports, imports, optional task bindings). Same as `aie.core`.
5. **No hardcoding of test data, validators, or arg counts.** The host runner takes its inputs as block arguments and the test harness (Python `pytest`) decides what to pass in. `run.py` is a *trivial* shell — load, run, memcpy, launch, stop — with no app-specific logic.
6. **AIRToCSL becomes thin.** It mirrors AIRToAIE: extract herd body, allocate buffers, materialize channels symbolically, leave routing to a downstream pass. Today it does too much (validates, names, hardcodes). It should do almost nothing except re-shape.

---

## 10. Recommendation

**Go with Sketch C (three-layer split).** Reasons in order:

1. **It matches CSL's actual structure.** The three-file structure (`layout.csl` / `pe_program.csl` / `run.py`) is not an accident of tooling — it is how CSL works. An IR that mirrors it has the lowest impedance mismatch with the target.
2. **It enforces the separation we wanted.** §8 framed our problem as "the implementation leaks because the split is a convention, not a structure." Sketch C makes it a structure: different dialects, different submodules, different emitters. No convention to forget.
3. **NxN is free.** `csl_layout.place_grid` is a one-op grid placement; the elaborate-grid pass turns it into N place ops. No second code path.
4. **Verification is real.** Cross-layer references via symbols mean we can verify, at compile time, that every host memcpy targets a real layout export, that every layout export references a real program export, that every place op references a real program. Today these are name-matched at print time — silent failure mode.
5. **Each emitter becomes tiny.** Splitting `CSLRuntimeToPy.cpp` into three is roughly 200 lines per emitter, and each emitter is a near-mechanical walker over one dialect.
6. **It has a future.** When we add tasks, DSDs, and routing, each lives in the dialect that owns its concern. We never re-litigate where a new feature belongs.

**The honest cost of Sketch C** is that the migration is bigger than Sketch A and the dialect surface is bigger than Sketch B. It's the right *technical* answer, and given that we've already built a vecadd pipeline once, building it again on a cleaner foundation is *cheaper than* trying to retrofit the current shape — because we know exactly what we want this time.

**If you don't have appetite for a full Sketch-C migration tomorrow,** Sketch A is a perfectly fine intermediate state. Most of the leaks (#5–#11 in §6.5) get fixed by Sketch A. The unfixed parts are the *structural* ones (#1–#4, multi-PE, route elaboration), which Sketch A doesn't address. So the ladder is:

- **Today** → fragile single 1×1 vec_add path
- **Sketch A** → clean 1×1 path with `csl.code_region` actually doing something, no direction duplication, no hardcoded constants. Still 2 dialects.
- **Sketch C** → clean NxN path, three dialects, three emitters, no leaks.

If we have time only for one, Sketch C. If we want to ship correctness fast and refactor later, A first.

**I would not pick Sketch B.** It's the "feels like AIE" answer, but CSL's actual shape is template-first (one `.csl` file → many PEs), and Sketch B inverts that, then has to undo it with a dedup pass. The PE-as-unit model is right for AIE because each AIE core has its own ELF; it's wrong for CSL because each PE source file is shared.

---

## 11. Optimizations to revisit later (from AIE)

Skipping these for now per scope. Listed so we know what to come back to and where the prior art lives.

| AIE pass | File | What it does | Reusable for CSL? |
|---|---|---|---|
| `AIEObjectFifoStatefulTransform` | `lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp` (~103 KB) | Turns objectfifos into buffers + locks + DMA chains, with depth analysis from acquire/release counts | The **double-buffering and depth analysis** parts directly. Locks are different (we use colors). |
| `AIECreatePathFindFlows` / `AIEPathFinder` | `lib/Dialect/AIE/Transforms/AIE{CreatePathFindFlows,PathFinder}.cpp` | Graph routing of flows through switchboxes with congestion awareness, deadlock detection | Conceptually applicable to color/route assignment for CSL fabric |
| `AIEAssignBuffers` | `lib/Dialect/AIE/Transforms/AIEAssignBuffers.cpp` | Per-tile buffer placement with bank-awareness | Applicable for PE-local SRAM allocation when we have lots of vars |
| `AIEPlacer` | `lib/Dialect/AIE/Transforms/AIEPlacer.cpp` | Heuristic placement (greedy + simulated annealing) | Applicable for "given a herd shape, where on the fabric does it go" |
| `AIEAssignLockIDs` | `lib/Dialect/AIE/Transforms/AIEAssignLockIDs.cpp` | Symbolic lock → physical lock ID | We need the analog for **colors**: `csl-allocate-colors` |
| `AIEAssignBufferDescriptorIDs` | `lib/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.cpp` | Symbolic BD → physical BD index | Applicable for DSD ID assignment (when we add DSDs) |
| `AIEHoistVectorTransferPointers` | (~22 KB) | Loop-invariant hoisting of pointer math out of vector transfer loops | Generic enough to reuse |
| `AIEMaterializeRuntimeSequences` | `lib/Dialect/AIEX/Transforms/AIEMaterializeRuntimeSequences.cpp` | Inlines / unrolls runtime sequences | Direct analog needed for `csl_host` (handle nonblock, multi-launch) |

We will revisit these once vec_add and a 2×2 example are working on the new IR. The goal of the redesign is not to add optimizations; it is to make the *structure* clean enough that adding optimizations is straightforward.

---

## 12. Open questions for tomorrow

In rough priority order. Pick the ones you have an opinion on; the rest we can talk through.

1. **Sketch A or Sketch C?** Or "A first, then C as a follow-up"? My recommendation is C, but A is reasonable if you want to keep velocity and ship a multi-PE-capable cleanup before a bigger restructure.
2. **Is the three-submodule shape acceptable?** Sketch C assumes one outer `module {}` with three nested ops (`csl_program.module`, `csl_layout.layout`, `csl_host.runner`). MLIR supports this fine, but it's slightly unusual. Alternative: three top-level modules in three files, linked at a build step.
3. **Do we want `csl_program.task` from day one,** or only `csl_program.func`? vec_add doesn't need tasks. But if we don't reserve op-space now, the second example will force a redesign.
4. **DSDs as types or as ops?** Sketch C leaves this open. Two viable answers: (a) `!csl.dsd` is a type with `csl_program.get_mem_dsd` constructors and DSD-consuming builtins, mirroring CSL closely; or (b) keep memrefs and have a `csl-elaborate-dsds` pass that turns memref accesses into DSD ops only at the bottom of the pipeline. (a) is closer to how CSL is written; (b) is closer to how MLIR usually thinks. I lean (b) because it lets the kernel writer use plain memref/affine and gets us composable with linalg/vector dialect work later — but (a) is more honest to the language.
5. **Comptime params: dict-attribute or block-args?** Sketch C uses an attribute dict (`csl_program.module @pe { params = {M=4, N=6} }`). Alternative: `csl_program.module @pe(%M : i16, %N : i16)` with comptime block arguments. Block args feel more SSA-y but they imply runtime values, which is wrong for comptime. I think attribute dict is right but want to confirm.
6. **`csl_layout.color` IDs: explicit or pass-assigned?** I'm assuming pass-assigned (`csl-allocate-colors`). Confirm? Some users might want to fix specific color IDs for interop.
7. **One `csl_program.module` per kernel, or per PE-source-file?** In Sketch C I assumed one per *kernel symbol*, which means one emitted `.csl` per kernel, which means SPMD is trivial. But if we want to support a hybrid where a single kernel has per-PE specialization (e.g., "edge PEs run a slightly different function"), we may need to allow multiple emitted files from one `csl_program.module` via attribute-driven specialization. Worth thinking through.
8. **What's the exact AIRToCSLLayout / AIRToCSLProgram / AIRToCSLHost split?** Today's `air-to-csl-dialect` is one pass. In Sketch C, do we make it three passes (one per output dialect) or one pass that emits all three? Three is more orthogonal but harder to get right ordering. One is simpler but less testable.
9. **Migration path:** if we go with Sketch C, do we delete the `csl.*` and `csl_rt.*` dialects in one PR or maintain them in parallel during the migration? I'd vote for keeping them parallel for one milestone with a feature flag, then deleting.
10. **Do we want `air-translate --emit-csl-rt` to *remain* as a single front door** that produces all three files, or do we expose three separate translate targets (`--emit-csl-program`, `--emit-csl-layout`, `--emit-csl-host`)? AIE has many targets; we could too.
11. **Out of scope for this doc, but flagging:** when we add multi-PE, we will need a placement-hint pass *before* AIRToCSL (the analog of how AIRToAIE's `x_loc`/`y_loc` herd attrs come from a prior pass). What does that pass look like for Cerebras? Does the user write it directly, or do we infer it from herd shape?

---

## 13. Appendix — file index for tomorrow's reading

Things you may want to skim while reviewing this doc:

**Today's CSL/CSLRT (the targets of redesign):**
- `mlir-air/mlir/include/air/Dialect/CSL/CSLOps.td` — current CSL ops
- `mlir-air/mlir/include/air/Dialect/CSL/CSLBase.td` — types/enums
- `mlir-air/mlir/include/air/Dialect/CSLRuntime/CSLRuntimeOps.td` — current CSLRT ops
- `mlir-air/mlir/lib/Conversion/AIRToCSLDialect/AIRToCSLDialect.cpp` — the conversion that hardcodes 1×1 (lines 72–82, 223–230, 310, 320)
- `mlir-air/mlir/lib/Conversion/CSLToCSLRuntime/CSLToCSLRuntime.cpp` — the additive (not transformative) CSL → CSLRT pass
- `mlir-air/mlir/lib/Targets/CSLRuntimeToPy.cpp` — the 933-line emitter (LayoutEmitter ~80–167, KernelEmitter ~176–677, HostEmitter ~687–884)
- `mlir-air/mlir/test/Conversion/AIRToCSLDialect/vecadd.mlir` — the reference test

**AIE prior art (the inspiration):**
- `mlir-aie/include/aie/Dialect/AIE/IR/AIEOps.td` — primary AIE op definitions
- `mlir-aie/include/aie/Dialect/AIEX/IR/AIEX.td` — AIEX ops including `runtime_sequence` and `npu.*`
- `mlir-aie/lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp` — the elaboration pattern we want to copy
- `mlir-aie/lib/Targets/AIETargets.cpp` (lines 169–413) — translate target registration; see how each target is one focused emitter
- `mlir-aie/lib/Targets/AIETargetXAIEV2.cpp` — the closest analog to our `KernelEmitter`/`LayoutEmitter`: emits C++ source that programs the array
- `mlir-aie/test/aiecc/cpp_aie2p_target.mlir` (lines 26–64) — the small passthrough example I use in §2.2

**AIR prior art:**
- `mlir-air/mlir/lib/Conversion/AIRToAIEPass.cpp` — `outlineAIECores` (182–440), `LowerAIRChannelsPattern` (1117–1280), `AllocL1BuffersPattern` (889–938), `L2MemrefToMemTileMap` (1035–1115). This is the model for AIRToCSL in any sketch.
- `mlir-air/mlir/test/Conversion/AIRToAIE/air_herd_to_aie.mlir` — the smallest before/after I quoted in §4.4

**CSL language references (from the SDK docs):**
- <https://sdk.cerebras.net/csl/language_index>
- <https://sdk.cerebras.net/csl/tutorials/>
- <https://sdk.cerebras.net/csl/tutorials/gemv-01-complete-program/> (the single-PE GEMV)
- <https://sdk.cerebras.net/csl/tutorials/gemv-05-multiple-pes/> (the multi-PE row)
- <https://sdk.cerebras.net/csl/language/dsds>
- <https://sdk.cerebras.net/csl/language/task-ids>

**Caveats from the research:**
- The Cerebras docs page on routes / fabric DSDs / `@set_color_config` was partially unreachable. Before we finalize the layout dialect's route op shape, we should fetch the live "Routes and Fabric DSDs" page and the `language/builtins` page.
- All claims in §2–§5 are sourced from the listed files/URLs. Nothing was inferred from outside sources.
- The vocabulary table in §5.5 has a few rows marked "(loose)" or "no exact analog" — those are the genuinely uncertain mappings, mostly around tasks and host RPC semantics.

---

*End of design eval doc. Next step: read this, mark up §10 (recommendation) and §12 (open questions), and we'll convert the chosen sketch into a real implementation spec via the writing-plans skill.*
