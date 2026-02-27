# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Before running any tools, load modules and activate the environment:

```bash
module load python/3.10.3 cmake/3.26.0 rocm/6.2.1 singularity
source sandbox/bin/activate
source utils/env_setup_gpu.sh install llvm/install
```

This puts `air-opt`, `aircc.py`, `mlir-opt`, and `mlir-runner` on `PATH` and sets `PYTHONPATH` for the `air` Python package.

## Build

### First-time build (GPU-only, no AIE dependency)
```bash
./utils/build-llvm-local.sh llvm        # build LLVM into llvm/install
./utils/build-mlir-air-gpu.sh llvm      # build AIR into build/ + install/
```

### Rebuild after source changes
```bash
cd build && ninja install
```

### CMake flags
| Flag | Default | Effect |
|------|---------|--------|
| `AIR_ENABLE_AIE` | ON | Requires mlir-aie; disable for GPU-only builds |
| `AIR_ENABLE_GPU` | ON | Enables ROCDL/HIP passes |

## Testing

### Run all MLIR unit tests (from build dir)
```bash
cd build && ninja check-airmlir
```

### Run a single MLIR test file
```bash
# lit must be on PATH (activated via sandbox)
lit mlir/test/Dialect/CSL/roundtrip.mlir
lit mlir/test/Conversion/AIRToCSL/basic.mlir
```

### Run all tests
```bash
cd build && ninja check-all
```

Tests use LLVM's `lit` + `FileCheck`. Each `.mlir` test has `// RUN: air-opt %s ...` directives at the top.

## Using the Tools

### air-opt — pass optimizer
```bash
# Apply a single pass
air-opt input.mlir -air-to-rocdl -o output.mlir

# Apply AIR-to-CSL lowering (emits CSL files into output dir)
air-opt input.mlir -air-to-csl="output-dir=./output"

# Print IR after every pass (debugging)
air-opt input.mlir -air-to-csl="output-dir=./output" --mlir-print-ir-after-all
```

### aircc.py — compiler driver (GPU pipeline)
```bash
# Compile for MI300X (gfx942)
aircc.py --target gpu --gpu-arch gfx942 -o output.mlir test/gpu/4k_4k_mul/air_sync.mlir

# With verbose output and preserved intermediates
aircc.py --target gpu --gpu-arch gfx942 -v --tmpdir output_intermediate -o output.mlir input.mlir
```

### mlir-runner — execute compiled MLIR
```bash
mlir-runner --entry-point-result=void \
    --shared-libs=$LLVM_INSTALL_DIR/lib/libmlir_rocm_runtime.so \
    output.mlir
```

## Architecture

### Project layout

```
mlir/
  include/air/
    Dialect/AIR/       # AIR dialect TableGen + headers
    Dialect/AIRRt/     # AIRRt runtime metadata dialect
    Dialect/CSL/       # CSL dialect (Cerebras WSE target)
    Conversion/        # Pass headers
    Transform/         # Transform op headers
  lib/
    Dialect/AIR/IR/    # AIR dialect C++ impl
    Dialect/CSL/IR/    # CSL dialect C++ impl
    Conversion/        # Lowering pass implementations
    Targets/           # Text-format emitters (JSON, CSL)
  test/
    Dialect/{AIR,AIRRt,CSL}/   # Dialect round-trip + verification tests
    Conversion/{AIRToCSL,...}/ # Lowering tests (FileCheck)
tools/
  air-opt/             # Registers all passes; entry point for air-opt
  air-translate/       # Translation entry point (--csl-emit, etc.)
  aircc/               # aircc.py driver
python/air/compiler/aircc/main.py  # GPU/AIE compilation pipeline logic
utils/                 # Build + env setup scripts
test/gpu/              # GPU integration tests (4k_4k_mul, simple_test)
docs/more_docs/        # Design docs (CSL dialect, GPU toolchain, etc.)
```

### AIR dialect — the core abstraction

AIR models a three-level spatial hierarchy:

| Scope | Op | Maps to (GPU) | Maps to (AIE) |
|-------|----|---------------|---------------|
| Host | `air.launch` | `gpu.launch` | Shim DMA |
| L2 buffer | `air.segment` | — | Memtile |
| Compute tile | `air.herd` | GPU thread block | AIE core |

Data movement must use `air.dma_memcpy_nd` or `air.channel.put/get` — direct cross-level loads/stores are not valid. Memory spaces: `memref<..>` (L3/DDR), `memref<.., 1>` (L2), `memref<.., 2>` (L1).

Async tokens (`!air.async.token`) propagate data dependences; `air.execute` wraps imperative ops into the async graph.

### Conversion pass map

```
AIR dialect
  ├─ -air-to-rocdl          → gpu.launch  (AIRToROCDLPass.cpp)
  ├─ -air-gpu-outlining      → gpu.module  (GPUKernelOutlingPass.cpp)
  ├─ -air-to-csl             → CSL text files (AIRToCSLPass.cpp)  [Phase 1 emitter]
  ├─ -air-to-aie             → AIE dialect  (requires AIR_ENABLE_AIE)
  └─ -air-to-async           → async dialect (AIRToAsyncPass.cpp)

AIRRt dialect
  ├─ -airrt-to-llvm          → LLVM dialect
  └─ -airrt-to-npu           → NPU dispatch ops
```

### CSL dialect (current development focus)

The CSL dialect (`mlir/include/air/Dialect/CSL/`) targets Cerebras WSE. Ops are split across six TableGen files within a single `csl` namespace:

- `CSLLayoutOps.td` — spatial grid (`csl.layout`)
- `CSLPlacementOps.td` — tile-to-code mapping
- `CSLRoutingOps.td` — color declaration and routing
- `CSLKernelOps.td` — PE-level functions/tasks/variables
- `CSLDataMovementOps.td` — DSD creation and bulk moves
- `CSLRuntimeOps.td` — module system, exports, comptime blocks

Phase 1 (`-air-to-csl` pass) is a direct text emitter that writes `layout.csl`, `pe_program.csl`, and `run.py`. Phase 2 is the structured CSL dialect above, lowered to text via `air-translate --csl-emit`.

### GPU compilation pipeline (aircc.py)

1. `air-opt -air-to-rocdl` — `air.launch/segment/herd` → `gpu.launch`
2. `air-opt -air-gpu-outlining` — outline into `gpu.module`
3. `mlir-opt` — lower affine/scf/cf + `gpu-kernel-outlining`
4. `mlir-opt` — `rocdl-attach-target`, `convert-gpu-to-rocdl`, `gpu-module-to-binary`, `gpu-to-llvm`

Intermediate files land in `--tmpdir` (default: auto temp dir). Use `-v` to see each command.

### C++ namespace

All AIR C++ code lives in `namespace xilinx::air`. CSL code lives in `namespace xilinx::csl`. Pass definitions use `#define GEN_PASS_DEF_<PASSNAME>` + `#include "air/Conversion/Passes.h.inc"` (TableGen-generated).
