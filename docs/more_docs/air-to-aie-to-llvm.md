
---

## The Full Picture: MLIR-AIE Lowering Passes, How MLIR-AIR Interfaces, and What the Backend Actually Emits

---

### Architecture Overview

The two repos are separate but tightly coupled. MLIR-AIR is the *scheduling / tile-mapping* layer. MLIR-AIE is the *physical resource / hardware configuration* layer. The handoff between them is the `aie.*` dialect — MLIR-AIR's `-air-to-aie` pass produces an AIE dialect module, which then goes through the MLIR-AIE pass pipeline independently.

```
MLIR-AIR (air.* dialect)
    └─ -air-to-aie ──────────────────────┐
                                          ▼
                                   aie.* dialect (MLIR-AIE)
                                          │
                              ┌───────────┴───────────────┐
                              ▼                           ▼
                    Device code path              Host config path
                  (one per core / herd)         (partition setup)
                              │                           │
                    aie-opt passes               aie-translate
                              │                 --aie-generate-xaiev2
                              ▼                           │
                        LLVM IR (MLIR)                    ▼
                              │               C++ calling libXAIE / XRT
                    Peano (llvm-aie)                      │
                     or xchesscc                          ▼
                              │               compiled into host .a/.so
                              ▼
                      per-core .elf
```

---

### Table 1: MLIR-AIE Passes (aie-opt)

These are the passes in the AIE dialect that operate *after* MLIR-AIR has handed off an `aie.device` module. They handle physical resource allocation, routing, and lowering to hardware-ready representation.

| Pass name (flag) | C++ class | Category | What it does |
|---|---|---|---|
| `--aie-assign-buffer-addresses` | `AIEAssignBufferAddressesPass` | Buffer allocation | Assigns concrete memory addresses to every `aie.buffer` that was declared without one. Two schemes: `bank-aware` (default, respects SRAM banking) or `basic-sequential`. |
| `--aie-assign-buffer-descriptor-ids` | `AIEAssignBufferDescriptorIDsPass` | DMA allocation | Assigns hardware BD (Buffer Descriptor) slot IDs to each DMA operation. |
| `--aie-assign-lock-ids` | `AIEAssignLockIDsPass` | Lock allocation | Assigns integer lock IDs to `aie.lock` ops that have none. Each tile has a fixed pool (typically 16 locks). |
| `--aie-assign-tile-ctrl-ids` | `AIEAssignTileCtrlIDsPass` | Control routing | Assigns unique controller IDs per `aie.tile`. Option: `-column-wise-unique-ids` for column-scoped uniqueness vs. global. |
| `--aie-create-pathfinder-flows` | `AIEPathfinderPass` | Routing | Automatically routes `aie.flow` operations through the stream switch network using a Pathfinder algorithm. The key routing pass — takes logical flows and produces `aie.switchbox` + `aie.connect` operations with actual channel numbers. |
| `--aie-find-flows` | `AIEFindFlowsPass` | Routing analysis | Verifies that every configured switchbox contributes to an end-to-end circuit-switched or packet-switched flow. Used primarily for testing the Pathfinder. |
| `--aie-create-packet-flows` | `AIERoutePacketFlowsPass` | Routing | Routes `aie.packetflow` (packet-switched) connections through the switch network, assigning packet IDs and switch configurations. |
| `--aie-lower-cascade-flows` | `AIELowerCascadeFlowsPass` | Cascade | Replaces `aie.cascade_flow` with concrete `aie.configure_cascade` operations. |
| `--aie-localize-locks` | `AIELocalizeLocksPass` | Address space | Converts global lock references to tile-local lock indices. Each of 4 adjacent tiles sees a lock under a different address offset — this pass resolves that. |
| `--aie-normalize-address-spaces` | `AIENormalizeAddressSpacesPass` | Core codegen | Strips non-default address spaces from `memref` types in `aie.core` regions. After outlining, each core only sees its own local address space. |
| `--aie-objectfifo-stateful-transform` | `AIEObjectFifoStatefulTransformPass` | ObjectFifo lowering | **Key structural pass.** Lowers `aie.objectFifo.createObjectFifo` into: `aie.buffer` + `aie.lock` on the producer tile; `aie.flow` + `aie.dma` between non-adjacent tiles; and acquire/release lock protocols. |
| `--aie-objectfifo-unroll` | `AIEObjectFifoUnrollPass` | ObjectFifo lowering | Unrolls loops containing objectFifo access patterns based on the FIFO depth. Option: `-dynamic-objFifos` to use runtime access instead of static unrolling. |
| `--aie-register-objectfifo-accessor-patterns` | `AIERegisterObjectFifoAccessorPatternsPass` | ObjectFifo lowering | Generates acquire/release patterns inside `aie.core` regions for `aie.objectfifo.register_process` operations. |
| `--aie-generate-column-control-overlay` | `AIEGenerateColumnControlOverlayPass` | Control infrastructure | Spawns a control packet streaming network across tile columns for runtime configuration. Options: `-route-shim-to-tct`, `-route-shim-to-tile-ctrl`. |
| `--aie-insert-device` | `AIEInsertDevicePass` | Structural | Wraps designs that lack a top-level `aie.device` op, inserting one automatically. |
| `--aie-vectorize` | `AIEVectorizePass` | Vectorization | Transforms MLIR `vector.*` ops into `aievec.*` ops matching the AIE vector permute network, multiply-accumulate units, and data types. |

---

### Table 2: AIEX / AIEX Experimental Passes

The `aiex` dialect holds NPU-specific and experimental ops — primarily the host-side NPU instruction sequence.

| Pass name (flag) | Category | What it does |
|---|---|---|
| `--aiex-insert-trace-packet-flow` | Tracing | Injects trace packet routing for hardware performance counters. |
| `--aiex-insert-shim-dma-bd-chain-to-host` | DMA | Builds the BD chain for host-to-device / device-to-host shim DMA transfers. |
| `--convert-aiex-to-standard` | Conversion | Lowers experimental AIEX ops to standard MLIR dialects before further lowering. |
| `--aie-npu-serialize-control-packets` | NPU | Serializes control packets into the flat `aiex.npu.write32` / `aiex.npu.dma_memcpy_nd` instruction stream consumed by the NPU firmware. |
| `--aie-control-packet-to-transaction` | NPU | Converts control packet ops into transaction ops for XRT's transaction buffer. |

---

### Table 3: AIEVec Passes (vector dialect → AIEVec → LLVM)

| Pass name (flag) | What it does |
|---|---|
| `--affine-super-vectorize` | Standard MLIR upstream pass: extracts generic `vector.*` ops from affine loop nests (e.g., `-virtual-vector-size=8`). |
| `--aie-vectorize` | Lowers `vector.*` → `aievec.*` — AIE-specific MAC units, permute networks, cascade accumulation. |
| `--convert-aievec-to-llvm` | Lowers `aievec.*` → LLVM IR intrinsics that map directly to AIE ISA instructions. This is the final codegen step before Peano. |

---

### Table 4: Translation passes (`aie-translate`)

`aie-translate` is a separate binary from `aie-opt`. It handles *translation* — one-shot lowering to text or binary formats that are not MLIR modules.

| Translation flag | Output | Used for |
|---|---|---|
| `--aie-generate-xaiev2` | C++ source (libXAIE calls) | Host-side hardware configuration code — sets up switchboxes, locks, DMA buffer descriptors, flow routing using the XAIEv2 API. |
| `--aie-generate-txn` | XRT transaction binary | Direct binary encoding of configuration commands for the NPU, consumed by XRT's `xclbin` transaction buffer mechanism. |
| `--aie-mlir-to-llvm` | LLVM IR text | Core function lowering — produces LLVM IR for `aie.core` regions, ready for Peano or xchesscc. |
| `--aie-flows-to-json` | JSON | Routing visualization — dumps flow topology for the `visualize.py` tool. |
| `--aievec-to-cpp` | C++ with intrinsics | Debug output of vectorized core code as readable C++. |

---

### How MLIR-AIR Interfaces With MLIR-AIE

The interface is clean and one-directional: MLIR-AIR produces `aie.*` IR, then hands it to `aiecc.py` as a subprocess. There is no shared C++ API between the two at the MLIR level.

The `air-to-aie` pass generates an AIE dialect MLIR module for each AIR dialect partition, and adds runtime metadata to the AIR dialect program. The AIR dialect program is then lowered to control code by running `air-to-std` to generate AIRRt dialect, then `airrt-to-llvm` to lower AIRRt to LLVM dialect, then invoking `aiecc.py` on the AIE dialect module.

Here is the precise sequence inside `aircc.py`:

```
air.mlir
  │
  ├─ [MLIR-AIR passes]
  │     -air-dependency
  │     -air-dependency-canonicalize
  │     -air-dependency-schedule-opt
  │     -air-ping-pong-transform
  │     ...optimization passes...
  │     -air-to-aie        ←── produces aie.device module(s)
  │     -air-to-std        ←── produces airrt + llvm dialect (host control)
  │     -airrt-to-llvm     ←── lowers airrt to pure LLVM dialect
  │
  ├─── HOST PATH ──────────────────────────────────────────────────────────
  │     mlir-translate --mlir-to-llvmir   →  host.ll
  │     clang host.ll → host.o
  │     Link with AIR runtime (libair) → libmydesign.a / .so
  │
  └─── DEVICE PATH (one per core in each herd) ───────────────────────────
        aie-opt [AIE passes above: objectfifo, routing, buffer assign, ...]
        │
        aie-translate --aie-mlir-to-llvm  →  core_N.ll
        │
        Peano (llvm-aie clang) OR xchesscc
        │
        core_N.elf  (loaded onto AIE tile at runtime via XRT)
```

The handoff document is the `aie.device { ... }` MLIR module. MLIR-AIR writes it; MLIR-AIE's passes consume it.

---

### What MLIR-AIR Actually Emits as Backend Code

There are two completely separate outputs, not one:

**1. Device ELF files — one per AIE core**

The device path is: `aie.core` region → `aie-translate --aie-mlir-to-llvm` → LLVM IR → **Peano** (`llvm-aie`, a fork of LLVM with a custom AIE target) → ELF. The project supports both Peano (`--no-xbridge`) and the proprietary `xchesscc` (`--xbridge`, requires Vitis license) as compiler backends for the core ELFs. Peano is the open-source default for Ryzen AI (AIE2/AIE2P). These ELFs are not linked into any library — they are loaded separately into each tile's program memory at runtime by XRT/the NPU driver.

**2. Host configuration code — C++ calling libXAIE or XRT transaction buffers**

The `aiecc.py` pass generates C++ code to configure the partition at runtime using the `-aie-generate-xaiev2` option, which calls the XAIEv2 API (libXAIE). The generated C++ wrappers are compiled and linked with the control code generated by the MLIR passes into a single library — either a shared `.so` or a static `.a`. The AIE ELF files are not part of the generated library and must be available separately at runtime.

For NPU/Ryzen AI targets, the newer path uses `--aie-generate-txn` instead — this generates a binary XRT *transaction buffer* (a sequence of `write32` / DMA configure operations) that bypasses C++ entirely and is passed directly to XRT's `xclbin` loading mechanism as a serialized blob.

---

### The Key Architectural Insight

The reason this split exists — LLVM for device, libXAIE/XRT transactions for host configuration — is that these are genuinely different hardware components:

- The **AIE cores** are VLIW scalar+vector processors that execute ELF binaries. They need a real ISA compiler (Peano/LLVM).
- The **stream switches, DMA engines, and lock controllers** are configuration registers, not processors. "Programming" them means writing a sequence of MMIO writes. libXAIE is a C wrapper around those writes; XRT transaction buffers are the pre-serialized binary form of those same writes, suitable for fast replay without the host CPU executing them at runtime.

This is also why there is no JIT in this stack: both outputs are inherently static. The ELF is compiled for a fixed program; the configuration bitstream is computed once from the topology. Nothing is data-dependent at runtime in the way that would require JIT.