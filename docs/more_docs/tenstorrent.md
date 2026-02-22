The Tenstorrent Compiler Stack: A Deep Study
The Tenstorrent stack is organized into three largely independent but tightly interfaced projects. Understanding how they relate to each other is the prerequisite for understanding any individual piece.
ML Frameworks (PyTorch, JAX, ONNX, TensorFlow)
        │
        ▼
 Front Ends (tt-xla via PJRT / tt-forge-fe via TVM)
        │  emit StableHLO or TTIR
        ▼
 TT-MLIR compiler (ttmlir-opt + ttmlir-translate)
   Dialects: TTCore → TTIR → [TTNN | TTKernel | TTMetal | D2M]
        │
        ├─── TTNN path ──────────────────────────────────────────────────┐
        │    ttmlir-translate --ttnn-to-flatbuffer → .ttnn binary        │
        │                                                                │
        └─── TTMetal / TTKernel path ────────────────────────────────────┤
             ttmlir-translate --ttmetal-to-flatbuffer → .ttm binary     │
             OR  ttmlir-translate --mlir-to-cpp (EmitC) → C++           │
                                                                         ▼
 TT-Metal runtime (tt-metalium)
   TTNN library (high-level op library) → tt-metal dispatch layer
        │
        ▼
 LLK (Low-Level Kernels): pre-compiled RISC-V C++ template library
        │
        ▼
 UMD (User-Mode Driver) + KMD (Kernel-Mode Driver)
        │
        ▼
 Wormhole / Blackhole hardware (Tensix cores)

Part 1: The Hardware Model (Tensix)
Before understanding the compiler you have to understand what it targets. A Tensix core is the fundamental compute tile, and it is radically different from a GPU streaming multiprocessor.
Each Tensix core contains five small RISC-V processors (called "Baby RISCVs") that run C/C++ kernels and dispatch instructions to the compute and data movement engines, plus 1 MB of SRAM (L1), a matrix engine (FPU) for 32×32 tile multiply-accumulate, and a vector engine (SFPU). The five RISC-Vs are named BRISC, TRISC0, TRISC1, TRISC2, and NCRISC. Hatenablog There are two Networks-on-Chip (NoC 0 and NoC 1) connecting tiles to each other and to DRAM.
The programming model consequently has three distinct execution threads per core — a reader (BRISC or NCRISC on NoC), a compute (TRISC0/1/2 driving the matrix and SFPU engines), and a writer (the other NoC RISC-V). They communicate exclusively through circular buffers (CBs) in L1. This is not an abstract software concept: CBs are hardware-managed semaphore-guarded ring buffers in SRAM. The reader fills CBs from DRAM or another tile's L1 via NoC, the compute engine reads from CBs, produces into CBs, and the writer drains output CBs back to DRAM.
The memory hierarchy is two-level: ~1.5 MB L1 SRAM per core (low latency, local), and ~12 GB DRAM shared across all cores (high latency, accessed via NoC). The entire tt-mlir optimizer exists to maximize data residency in L1 and minimize DRAM round-trips.

Part 2: The tt-mlir Dialect Tower
tt-mlir defines six custom MLIR dialects, each at a progressively lower level of abstraction. This is the most important structural fact about the whole stack.
Table 1: Dialect Hierarchy
DialectAbstraction levelAnalogous toKey primitiveTTCoreTypes/attributes onlyN/A — infrastructurettcore.tile<32x32, bf16>, ttcore.grid, ttcore.metal_layout, memory spaces (l1, dram)TTIRHigh-level named tensor opsStableHLO / TOSAttir.matmul, ttir.conv2d, ttir.add, ttir.to_layout — shape-level, device-agnosticTTNNTTNN API mirrorTFLite / ONNX-RT opsttnn.matmul, ttnn.linear, ttnn.deallocate — with layout annotations (#ttnn_layout) encoding memory space, grid, and interleave/shard modeD2MGeneric sharded computelinalg.genericd2m.generic, d2m.cb (circular buffer type), d2m.wait, d2m.reserve, d2m.yield, d2m.to_layout — models explicit data movement into/out of CBsTTKernelPer-core RISC-V kernelRISC-V assembly at C levelttkernel.cb_reserve_back, ttkernel.cb_wait_front, ttkernel.cb_push_back, ttkernel.cb_pop_front, ttkernel.tile_regs_acquire, ttkernel.mm_block_init, ttkernel.pack_tile, ttkernel.sfpu_*TTMetalHost-side device dispatchAIE's aie.core + aie.dmattmetal.create_buffer, ttmetal.enqueue_write_buffer, ttmetal.enqueue_program, ttmetal.enqueue_read_buffer, ttmetal.finish, ttmetal.deallocate_buffer
TTCore provides common types such as ttcore.tile, ttcore.metal_layout, ttcore.grid, and enums for data formats, memory spaces, and iterator types. TTIR is a high-level dialect modeling the tensor compute graph, accepting both TOSA and linalg input. TTKernel ops include matmul, add, and multiply for tile computations in register space, plus ttkernel.sfpu_* ops using the SFPU coprocessor on destination register space. TTMetal operations dispatch work from host to device. Tenstorrent
The key concept unique to Tenstorrent in the TTNN dialect is the #ttnn_layout attribute. Unlike AIE where placement is handled by routing passes, every TTNN tensor carries its full memory specification directly as an attribute: memory space (L1 vs DRAM), interleaving strategy (interleaved vs height/width/block sharded), the grid dimensions it spans, and the tile format. This annotation is what all the optimization passes operate on.

Part 3: ttmlir-opt — The Pass Pipeline
ttmlir-opt is the analog of mlir-opt / aie-opt. It runs transformation passes. There are two main compilation pipelines depending on target backend.
Table 2: TTNN Backend Passes (--ttir-to-ttnn-backend-pipeline)
This is the primary production path. It lowers TTIR to TTNN dialect, which maps 1:1 to the TTNN C++ library shipped in tt-metal.
Pass (flag)From → ToWhat it does--ttcore-register-device / --ttir-load-system-descSetupLoads a .ttsys system descriptor file (device grid size, DRAM capacity, L1 per core) and attaches it as a ttcore.system_desc attribute on the module. This is required before any hardware-dependent pass runs.--ttir-implicit-deviceTTIR setupInfers and inserts device attributes that were not explicitly provided.--ttir-layoutTTIRInserts ttir.to_layout ops to make memory movement explicit between different tensor memory spaces and layouts before TTNN lowering begins.--convert-ttir-to-ttnnTTIR → TTNNCore conversion pass. Rewrites each ttir.* op to its corresponding ttnn.* op. Inserts ttnn.deallocate for intermediate tensors that are no longer live. Applies at ModuleOp level, matching all ops via populateTTIRToTTNNPatterns().--ttnn-layoutTTNNAssigns initial #ttnn_layout attributes to all tensors — defaults to DRAM interleaved if no layout hint is present. This is the starting point before the optimizer.--ttnn-optimizerTTNNThe main performance pass. Runs the full TTNNOptimizer pipeline (see Part 4 below). Maximizes L1 residency, selects sharding configurations, inserts ttnn.to_layout reshards, and annotates all ops with final layout and op-specific configs.--ttnn-decompose-layoutsTTNNDecomposes complex layout conversions into simpler device-executable steps after optimizer has made its decisions.--ttnn-workaroundsTTNNApplies hardware-specific workarounds and fixes for known device limitations on Wormhole / Blackhole.--ttnn-modify-signatures-for-dylibTTNNAdapts function signatures for dynamic library (.so) output when building shared-library dispatch mode.
The minimal low-level invocation for a single pass is:
bashttmlir-opt \
  --ttcore-register-device="system-desc-path=/path/to/system_desc.ttsys" \
  --ttnn-layout \
  --convert-ttir-to-ttnn \
  input.mlir -o output_ttnn.mlir
The full pipeline:
bashttmlir-opt \
  --ttir-to-ttnn-backend-pipeline="system-desc-path=/path/to/system_desc.ttsys" \
  input.mlir -o output_ttnn.mlir
```

#### Table 3: TTMetal/TTKernel Backend Passes (`--ttir-to-ttmetal-backend-pipeline`)

This is the lower-level path for custom kernel generation via PyKernel / D2M.

| Pass (flag) | From → To | What it does |
|---|---|---|
| `--convert-ttir-to-ttmetal` or via D2M | TTIR → TTMetal | Lowers TTIR ops to `ttmetal.*` host dispatch ops. Produces `ttmetal.enqueue_program` calls with attached kernel configs describing which RISC-V threads get which kernel function. |
| `--convert-d2m-to-ttkernel` | D2M → TTKernel | Core kernel lowering for the D2M path. Converts `d2m.generic` regions (with `d2m.wait`/`d2m.reserve`/`d2m.yield`) into `ttkernel.*` ops for reader, compute, and writer threads separately. |
| `--convert-ttkernel-to-emitc` | TTKernel → EmitC | **Translation to C++.** Lowers `ttkernel.*` ops into `emitc.*` ops — MLIR's standard "emit C/C++" dialect. Each `ttkernel.cb_reserve_back` becomes an `emitc.call` to `cb_reserve_back(...)`, etc. |
| `--ttmetal-to-flatbuffer` (via ttmlir-translate) | TTMetal → binary | Serializes the TTMetal host program and embedded kernel C++ source into a `.ttm` Flatbuffer file. |

The PyKernel compilation flow runs a series of transformations on the MLIR module and lowers to the emitc dialect to translate the module into C++ code. This C++ code is the artifact that is consumed by the runtime to execute on Tenstorrent hardware. 

---

### Part 4: The TTNNOptimizer Pass — How Layout Selection Actually Works

This is the most architecturally interesting part of the entire stack and has no direct equivalent in mlir-aie. It warrants a detailed breakdown.

The TTNNOptimizer pass determines optimal memory layouts and op configurations for TTNN operations to maximize performance on Tenstorrent hardware. The fundamental goal is to maximize data residency in L1 memory while maintaining correctness. 

The optimizer runs five sequential sub-analyses followed by a graph transformation stage:

**1. ScalarDataTypeAnalysis** — collects all element types used across the graph (bf16, f32, etc.) to determine which layouts are valid.

**2. LegalTensorLayoutAnalysis** — for each (tensor type × scalar type) pair, enumerates all possible `#ttnn_layout` combinations across: page layout (Tiled 32×32 vs RowMajor), memory (L1 vs DRAM), distribution strategy (Interleaved vs Height/Width/Block sharded), and grid dimensions (1×1, 1×8, 8×8, etc.). This can produce hundreds of candidates per tensor.

**3. LegalOpLayoutAnalysis** — per op, filters the candidate layouts from step 2 through OpModel validation (a backend query API that checks whether a given op can execute with specified input/output layouts). Capped by `maxLegalLayouts` to bound the search space.

**4. LegalOpConfigAnalysis** — for ops with additional configuration knobs (notably Conv2d, which has block size overrides, activation handling flags), generates the Cartesian product of valid layouts × op-specific parameter values.

**5. DFShardingPolicy + ShardSolver** — the core scheduler and constraint solver:

An L1 chain is a sequence of operations whose intermediate tensors can reside in L1 memory. The goal is to identify maximal chains where data flows through L1 without spilling to DRAM. The policy walks the graph in DFS (Depth-First Search) schedulable order, adding ops to a chain if shardable, continuing the chain if: (1) the next op uses the current as operand[0], and (2) the current op has a single use. Otherwise it finalizes the chain and starts a new one. 

ShardSolver then performs constraint satisfaction over each chain: it tracks which configurations remain valid for each op using bitsets, constructs a PathSet graph of compatible producer→consumer config pairs, propagates constraints bidirectionally until convergence, inserts reshards (`ttnn.to_layout` ops) where adjacent ops have incompatible layouts, and finally selects the config assignment that maximizes total core utilization across the chain.

**6. Graph Transformation** — applies the resolved layout attributes and op configs to the IR, inserts `ttnn.to_layout` reshards, spills chain outputs to DRAM where necessary, and reorders the op schedule for memory-efficient execution.

The docs also openly acknowledge the current approach's known limitations and describe a planned replacement — a simpler two-pass greedy + DP architecture — validated against 50+ models where empirical analysis showed 40–94% L1 headroom, with most spills driven by ops that structurally require DRAM inputs (reduce, permute, reshape), not by actual memory pressure.

---

### Part 5: ttmlir-translate — The Backend Emission Story

This is where your instinct about `mlir-translate` being the right fit for AI accelerator backends is directly confirmed by Tenstorrent's actual design.

`ttmlir-translate` converts from IR to external representation (and inverse). For example, IR in EmitC dialect can be converted into C++ code. 

There are three distinct translation paths:

#### Table 4: ttmlir-translate Translation Modes

| Flag | Input dialect | Output | Output format | Used for |
|---|---|---|---|---|
| `--ttnn-to-flatbuffer` | TTNN | `.ttnn` binary | FlatBuffer | **Primary production path.** Serializes the TTNN program (op sequence, tensor layouts, buffer addresses, memory configs) as a FlatBuffer binary. The tt-mlir runtime deserializes this and dispatches TTNN C++ API calls. |
| `--ttmetal-to-flatbuffer` | TTMetal | `.ttm` binary | FlatBuffer | Serializes TTMetal host programs. Kernel C++ source code (from EmitC) is embedded inside the FlatBuffer. The runtime extracts and compiles the kernels. |
| `--mlir-to-cpp` | EmitC | `.cpp` C++ source | Text | **Kernel codegen path.** Translates TTKernel ops (after `--convert-ttkernel-to-emitc`) to C++ source using `emitc.call` → function call mappings. Produces source that calls tt-metal's `cb_reserve_back`, `tile_regs_acquire`, `matmul_tiles`, `pack_tile`, etc. directly. |
| `--mlir-to-cpp` (EmitPy) | EmitPy | `.py` Python | Text | Python code generation path for testing and interactive workflows. |

The TTNN → FlatBuffer path is architecturally the central design decision. Rather than JIT-compiling to LLVM IR → machine code for the host (like traditional ML compilers do), or generating RISC-V ELFs for device cores (like mlir-aie does for AIE tiles), Tenstorrent encodes the *operation sequence* in a binary interchange format. The tt-metal C++ runtime then plays back that sequence by calling pre-compiled TTNN library functions. The RISC-V kernel binaries are pre-compiled into the TTNN library — the compiler chooses *which* pre-compiled kernel to dispatch, not the kernel binary itself.

The kernel source path (TTKernel → EmitC → C++) is used when the PyKernel or D2M infrastructure generates a *novel* kernel that isn't in the pre-compiled library — in that case the C++ source gets JIT-compiled at kernel load time by tt-metal's runtime compiler.

---

### Part 6: The tt-metal Runtime Layer

tt-metal (tt-metalium) is not a compiler — it is the SDK that the compiler's output runs on top of. It has two programming APIs:

**TTNN** is the high-level operator library. It provides PyTorch-parity Python/C++ APIs (`ttnn.matmul`, `ttnn.conv2d`, etc.) backed by pre-compiled kernels. The TTNN dialect in ttmlir maps 1:1 to this API. When the runtime reads a `.ttnn` FlatBuffer, it deserializes each op and calls the corresponding TTNN C++ function.

**tt-metal** (lower-level) is the bare-metal dispatch layer. You can write kernels directly in C++ using the circular buffer API (`cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front`), configure programs, and dispatch them to specific Tensix cores. This is what the TTMetal dialect maps to.

Below both sits **LLK (Low-Level Kernels)** — a C++ template library of hand-optimized RISC-V routines for tile operations (matrix multiply, elementwise, SFPU operations). Both TTNN precompiled kernels and user-written kernels call into LLK functions. LLK is the analog of Peano (the custom LLVM backend) in the mlir-aie world, except it is a C++ library rather than a compiler — the compilation of LLK kernels into actual RISC-V machine code is handled by the tt-metal build system using a standard RISC-V cross-compiler.

Below LLK is UMD (User-Mode Driver) and KMD (Kernel-Mode Driver), which handle PCIe communication, DMA setup, and hardware initialization.

---

### Part 7: The Full E2E Pipeline (tt-forge / tt-xla)

TT-XLA is the primary frontend for running PyTorch and JAX models. It leverages a PJRT interface to integrate JAX (and in the future other frameworks), TT-MLIR, and Tenstorrent hardware, providing StableHLO (SHLO) graphs to the TT-MLIR compiler. TT-Forge-FE is a TVM-based graph compiler designed to optimize and transform computational graphs, supporting ingestion of ONNX, TensorFlow, and PaddlePaddle via TT-TVM, and can also support PyTorch. TT-Forge-FE does not support multi-chip configurations. 

The end-to-end flow for a PyTorch model looks like this:
```
PyTorch model
    │ torch.compile() or JAX jit()
    ▼
TT-XLA (PJRT plugin)
    │ emits StableHLO graph
    ▼
ttmlir: StableHLO → TTIR
    │ standard HLO-to-TTIR lowering patterns
    ▼
ttmlir-opt --ttir-to-ttnn-backend-pipeline
    │
    ├─ ttcore-register-device (load .ttsys)
    ├─ ttir-layout             (explicit memory movement)
    ├─ convert-ttir-to-ttnn    (TTIR ops → TTNN ops)
    ├─ ttnn-layout             (default DRAM interleaved)
    ├─ ttnn-optimizer          (maximize L1, shard, insert reshards)
    ├─ ttnn-decompose-layouts  (complex layout → simple steps)
    └─ ttnn-workarounds        (hardware-specific fixes)
    │
    ▼ TTNN dialect MLIR
ttmlir-translate --ttnn-to-flatbuffer
    │
    ▼ .ttnn FlatBuffer binary
ttrt run out.ttnn
    │ (or tt-xla runtime directly)
    ▼
TT-MLIR runtime (C++)
    │ deserializes FlatBuffer, calls TTNN C++ API
    ▼
TTNN library (tt-metal)
    │ dispatches pre-compiled kernels to Tensix cores
    ▼
LLK → UMD/KMD → Wormhole/Blackhole silicon

Comparison: Tenstorrent vs AMD AIE
AspectAMD mlir-aieTenstorrent tt-mlirPrimary backend outputPer-core RISC-V .elf binaries (via Peano/LLVM) + host C++ using libXAIEFlatBuffer binary (.ttnn) dispatched by C++ runtimeKernel compilationFull LLVM pipeline per core at compile time (AOT)Kernels pre-compiled into TTNN library; custom kernels JIT-compiled by tt-metal at load timeHost configurationC++ calling libXAIE register writes, or XRT transaction binaryTTNN C++ API calls driven by FlatBuffer runtimeKey optimizer-air-ping-pong-transform, -air-dependency, routing passesTTNNOptimizer (ShardSolver + DFShardingPolicy)Layout representationmemref with memory space attributes; explicit aie.dma operations#ttnn_layout attribute on tensors encoding grid, shard strategy, memory typeRoutingExplicit Pathfinder routing pass; switchbox configurationNoC routing is implicit — TTNN library handles it, compiler only decides which tiles hold which tensor shardsKernel languageC++ compiled by Peano (custom LLVM AIE backend)C++ compiled by standard RISC-V GCC/LLVM calling LLK intrinsicstranslate roleaie-translate → libXAIE C++ or LLVM IRttmlir-translate → FlatBuffer binary or EmitC C++JIT vs AOTFully AOTMostly AOT (FlatBuffer), with runtime JIT for custom kernelsD2M / generic computelinalg.generic + air.channelD2M dialect (explicit CB-based linalg analog) + TTKernel

Your Key Intuition, Confirmed
Your framing — that mlir-translate is the right tool for AI accelerator backends — maps perfectly onto what Tenstorrent built. ttmlir-translate is the entire mechanism for leaving the MLIR world: everything above it is IR transformation, everything below it is the runtime. The split between ttmlir-opt (optimize) and ttmlir-translate (emit) is the exact pattern that makes the pipeline testable at every stage — you can run FileCheck on the MLIR and separately test the binary output — and it's the same reason mlir-aie uses aie-opt vs aie-translate for the same architectural reason.
The distinctive Tenstorrent choice is what the translator emits: not LLVM IR (since the compute is pre-compiled into TTNN) but a FlatBuffer, which is effectively a serialized dispatch plan. This makes deployment lightweight — the runtime doesn't need LLVM at all — and matches the fact that Tenstorrent's performance comes from layout and sharding decisions (optimizer's domain), not from per-op code generation (compiler backend's domain).