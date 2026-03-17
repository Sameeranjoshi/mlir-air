# AIRcc Compilation Pipelines

## GPU Compilation Pipeline

```
Input: AIR MLIR
  │
  ├─ Step 1: air-opt -air-to-rocdl ──────────────┐
  │                                               │
  │  Converts AIR dialect to ROCDL dialect       │
  │                                               │
  ├─────────────────────────────────────────────┤
  │                                               │
  ├─ Step 2: air-opt -air-gpu-outlining ────────┐
  │                                               │
  │  Extracts GPU kernels                        │
  │                                               │
  ├─────────────────────────────────────────────┤
  │                                               │
  ├─ Step 3: mlir-opt passes ─────────────────┐
  │                                              │
  │  Pass Pipeline:                             │
  │  ├─ lower-affine                           │
  │  ├─ convert-linalg-to-loops               │
  │  ├─ convert-scf-to-cf                     │
  │  └─ gpu-kernel-outlining                  │
  │                                              │
  ├────────────────────────────────────────────┤
  │                                              │
  ├─ Step 4: mlir-opt GPU Pipeline ─────────┐
  │                                            │
  │  Pass Pipeline:                           │
  │  ├─ rocdl-attach-target                  │
  │  ├─ gpu.module(                          │
  │  │  ├─ convert-gpu-to-rocdl             │
  │  │  └─ reconcile-unrealized-casts       │
  │  │)                                       │
  │  ├─ gpu-module-to-binary                 │
  │  ├─ func.func(gpu-async-region)          │
  │  ├─ gpu-to-llvm                          │
  │  ├─ convert-to-llvm                      │
  │  └─ reconcile-unrealized-casts           │
  │                                            │
  └────────────────────────────────────────────┘
              │
              ▼
Output: LLVM IR (final_output.mlir or stdout)
```

### GPU Pipeline Parameters
- **gpu_arch**: GPU architecture (e.g., gfx90a, gfx942)
- **gpu_runtime**: GPU runtime (e.g., rocm)
- **tmpdir**: Temporary directory for intermediate files
- **Intermediate files created**:
  - `{base}_step1_rocdl.mlir`
  - `{base}_step2_outlined.mlir`
  - `{base}_step3_gpu.mlir`
  - Final output (or stdout if no output file)

---

## AIE Compilation Pipeline

```
Input: AIR MLIR
  │
  ├─ Phase 1: Placement & Optimization ──────────────────────┐
  │                                                            │
  │  Core Passes:                                             │
  │  ├─ air-insert-launch-around-herd{insert-segment=true}  │
  │  ├─ func.func(air-lower-herd-parallel)                  │
  │  ├─ scf-forall-to-parallel                              │
  │  │                                                        │
  │  └─ [If NPU device] Optimization Passes:                │
  │     ├─ air-dependency                                   │
  │     ├─ air-hoist-dma-in-accum-pattern                   │
  │     ├─ air-broadcast-detection                          │
  │     ├─ air-specialize-dma-broadcast                     │
  │     ├─ air-dma-to-channel                               │
  │     ├─ canonicalize, cse                                │
  │     ├─ air-dependency-canonicalize                      │
  │     ├─ air-isolate-async-dma-loop-nests                 │
  │     ├─ air-fuse-channels                                │
  │     ├─ [Conditional] L2 splitting passes                │
  │     ├─ [Conditional] air-loop-fusion or                 │
  │     │  (air-fuse-alloc-dealloc +                        │
  │     │   air-shrink-memref-sizes-by-access)              │
  │     ├─ air-label-scf-for-to-ping-pong                   │
  │     ├─ air-ping-pong-transform                          │
  │     ├─ [Optional] air-linalg-to-func                    │
  │     ├─ [Optional] convert-linalg-to-loops               │
  │     └─ func.func(air-opt-memtile-dma-bds)               │
  │                                                          │
  │  Placement Passes:                                       │
  │  ├─ air-collapse-herd{max-col-size=4}                   │
  │  ├─ air-place-herds{params}                             │
  │  └─ func.func(air-renumber-dma)                         │
  │                                                          │
  │  Output: placed.*.mlir                                   │
  │                                                          │
  ├─────────────────────────────────────────────────────────┤
  │                                                          │
  ├─ Phase 2: Convert to AIE ─────────────────────────────┐
  │                                                        │
  │  Pass: air-to-aie{options}                            │
  │  Output: aie.*.mlir                                   │
  │                                                        │
  ├────────────────────────────────────────────────────────┤
  │                                                        │
  ├─ Phase 3: Device-Specific Processing ────────────────┐
  │                                                       │
  │  [If NPU device]:                                    │
  │  │                                                   │
  │  ├─ func.func(air-opt-shim-dma-bds{device})         │
  │  ├─ air-to-std                                       │
  │  ├─ symbol-dce, affine-expand-index-ops             │
  │  ├─ airrt-to-npu{trace-size, trace-offset}          │
  │  │                                                   │
  │  │  Output: npu.*.mlir                              │
  │  │                                                   │
  │  │  Then: Call aiecc.run() for AIE compilation      │
  │  │  ├─ Generates xclbin or txn format               │
  │  │  └─ Outputs: .xclbin, .insts.bin                 │
  │  │                                                   │
  │  [Else (non-NPU)]:                                   │
  │  │                                                   │
  │  └─ lower_airrt_to_airhost():                        │
  │     ├─ air-split-devices                             │
  │     ├─ convert-vector-to-llvm                        │
  │     ├─ convert-math-to-llvm                          │
  │     ├─ func.func(air-label-broadcast-channel)       │
  │     ├─ lower-affine                                 │
  │     ├─ func.func(air-opt-shim-dma-bds)              │
  │     ├─ air-to-std                                   │
  │     ├─ air-lower-linalg-tensors                     │
  │     ├─ airrt-to-llvm                                │
  │     ├─ LLVM lowering chain                          │
  │     │  ├─ expand-strided-metadata                   │
  │     │  ├─ lower-affine                              │
  │     │  ├─ convert-scf-to-cf                         │
  │     │  ├─ finalize-memref-to-llvm                   │
  │     │  ├─ convert-func-to-llvm                      │
  │     │  ├─ convert-arith-to-llvm                     │
  │     │  └─ convert-cf-to-llvm                        │
  │     ├─ aie-translate --mlir-to-llvmir                │
  │     ├─ opt -O3 (LLVM optimization)                  │
  │     ├─ llc (compilation to object files)            │
  │     ├─ For each segment:                            │
  │     │  ├─ aiecc.py (AIE compilation)                │
  │     │  ├─ Generate .inc (AIE C++ code)              │
  │     │  └─ Compile wrapper C++ to object file         │
  │     └─ Link all objects into .so/.a library         │
  │                                                     │
  └──────────────────────────────────────────────────────┘
              │
              ▼
Output: .xclbin/.insts.bin (NPU) or .so/.a library (non-NPU)
```


================================================================================
                  AIRCC GPU PIPELINE - SNAKE FLOW
================================================================================

                                    Input MLIR
                                        │
                                        ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    ▼                                                                  │
┌──────────────────────┐     ┌──────────────────────┐                 │
│ air-opt              │────▶│ air-opt              │                 │
│                      │     │                      │                 │
│ air-to-rocdl         │     │ air-gpu-outlining    │                 │
└──────────────────────┘     └──────────────────────┘                 │
                                    │                                 │
                                    ▼                                 │
                            ┌──────────────────────┐                 │
                            │ mlir-opt             │                 │
                            │                      │                 │
                            │ lower-affine         │                 │
                            │ convert-linalg-to-   │                 │
                            │ loops                │                 │
                            │ convert-scf-to-cf    │                 │
                            │ gpu-kernel-outlining │                 │
                            └──────────────────────┘                 │
                                    │                                 │
                                    ▼                                 │
                            ┌──────────────────────┐                 │
                            │ mlir-opt             │                 │
                            │                      │                 │
                            │ rocdl-attach-target  │                 │
                            │ gpu.module(          │                 │
                            │  convert-gpu-to-     │                 │
                            │  rocdl,              │                 │
                            │  reconcile-          │                 │
                            │  unrealized-casts)   │                 │
                            │ gpu-module-to-binary │                 │
                            │ func.func(           │                 │
                            │  gpu-async-region)   │                 │
                            │ gpu-to-llvm          │                 │
                            │ convert-to-llvm      │                 │
                            │ reconcile-           │                 │
                            │ unrealized-casts     │                 │
                            └──────────────────────┘                 │
                                    │                                 │
                                    └──────────────────────────────────┘
                                            │
                                            ▼
                                    Output: LLVM IR


================================================================================
                   AIRCC AIE PIPELINE - SNAKE FLOW
================================================================================

                                    Input MLIR
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    ▼                                                                         │
┌──────────────────────────────┐  ┌──────────────────────┐                   │
│ air-insert-launch-around-    │─▶│ func.func(air-lower- │                   │
│ herd{insert-segment=true}    │  │ herd-parallel)       │                   │
└──────────────────────────────┘  └──────────────────────┘                   │
                                           │                                 │
                                           ▼                                 │
                                  ┌──────────────────────┐                   │
                                  │ scf-forall-to-       │                   │
                                  │ parallel             │                   │
                                  └──────────────────────┘                   │
                                           │                                 │
                                           ▼                                 │
                                  ┌──────────────────────┐                   │
                                  │ air-dependency       │                   │
                                  │ [NPU only]           │                   │
                                  └──────────────────────┘                   │
                                           │                                 │
                                           ▼                                 │
                                  ┌──────────────────────┐                   │
                                  │ air-hoist-dma-in-    │                   │
                                  │ accum-pattern        │                   │
                                  └──────────────────────┘                   │
                                           │                                 │
                                           ▼                                 │
                                  ┌──────────────────────┐                   │
                                  │ air-broadcast-       │                   │
                                  │ detection            │                   │
                                  └──────────────────────┘                   │
                                           │                                 │
                                           ▼                                 │
                                  ┌──────────────────────┐                   │
                                  │ air-specialize-dma-  │                   │
                                  │ broadcast            │                   │
                                  └──────────────────────┘                   │
                                           │                                 │
                                           ▼                                 │
                                  ┌──────────────────────┐                   │
                                  │ air-dma-to-channel   │                   │
                                  └──────────────────────┘                   │
                                           │                                 │
                                           ▼                                 │
                                  ┌──────────────────────┐                   │
                                  │ canonicalize / cse   │                   │
                                  └──────────────────────┘                   │
                                           │                                 │
                                           ▼                                 │
                                  ┌──────────────────────┐                   │
                                  │ air-dependency-      │                   │
                                  │ canonicalize         │                   │
                                  └──────────────────────┘                   │
                                           │                                 │
                                           ▼                                 │
                                  ┌──────────────────────┐                   │
                                  │ canonicalize / cse   │                   │
                                  └──────────────────────┘                   │
                                           │                                 │
                                           ▼                                 │
                                  ┌──────────────────────┐                   │
                                  │ air-isolate-async-   │                   │
                                  │ dma-loop-nests       │                   │
                                  └──────────────────────┘                   │
                                           │                                 │
                                           ▼                                 │
                                  ┌──────────────────────┐                   │
                                  │ canonicalize / cse   │                   │
                                  └──────────────────────┘                   │
                                           │                                 │
                                           ▼                                 │
                                  ┌──────────────────────┐                   │
                                  │ air-fuse-channels    │                   │
                                  │ [±aggressive-mode]   │                   │
                                  └──────────────────────┘                   │
                                           │                                 │
                                           ▼                                 │
                                  ┌──────────────────────┐                   │
                                  │ canonicalize / cse   │                   │
                                  └──────────────────────┘                   │
                                           │                                 │
              ┌────────────────────────────┼────────────────────────────┐   │
              │                            │                            │   │
           [if NOT npu_1col]          [else or always]           [if air_loop_fusion]
              │                            │                            │   │
              ▼                            ▼                            ▼   │
         ┌─────────┐        ┌──────────────────────┐    ┌──────────────────────┐
         │ L2-split│        │ air-label-scf-for-to-│    │ func.func(air-loop-  │
         │ passes  │        │ ping-pong            │    │ fusion)              │
         └─────────┘        └──────────────────────┘    └──────────────────────┘
              │                            │                            │   │
              └────────────────┬───────────┴──────────────┬─────────────┘   │
                               ▼                          ▼               │
                    ┌──────────────────────┐    ┌──────────────────────┐  │
                    │ air-ping-pong-       │    │ func.func(           │  │
                    │ transform            │    │ air-fuse-alloc-      │  │
                    │ [if omit_pingpong]   │    │ dealloc,             │  │
                    └──────────────────────┘    │ air-shrink-memref-   │  │
                               │                │ sizes-by-access)     │  │
                               │                └──────────────────────┘  │
                               │                           │              │
                               └──────────────┬────────────┘              │
                                              ▼                          │
                                  ┌──────────────────────┐                │
                                  │ canonicalize / cse   │                │
                                  └──────────────────────┘                │
                                           │                              │
                                           ▼                              │
                                  ┌──────────────────────┐                │
                                  │ air-linalg-to-func   │                │
                                  │ OR convert-linalg-   │                │
                                  │ to-loops             │                │
                                  └──────────────────────┘                │
                                           │                              │
                                           ▼                              │
                                  ┌──────────────────────┐                │
                                  │ func.func(air-opt-   │                │
                                  │ memtile-dma-bds)     │                │
                                  └──────────────────────┘                │
                                           │                              │
                                           ▼                              │
                                  ┌──────────────────────┐                │
                                  │ canonicalize / cse   │                │
                                  └──────────────────────┘                │
                                           │                              │
                                           ▼                              │
                                  ┌──────────────────────┐                │
                                  │ air-collapse-herd    │                │
                                  │ {max-col-size=4}     │                │
                                  └──────────────────────┘                │
                                           │                              │
                                           ▼                              │
                                  ┌──────────────────────┐                │
                                  │ canonicalize / cse   │                │
                                  └──────────────────────┘                │
                                           │                              │
                                           ▼                              │
                                  ┌──────────────────────┐                │
                                  │ air-place-herds      │                │
                                  │ {placement-params}   │                │
                                  └──────────────────────┘                │
                                           │                              │
                                           ▼                              │
                                  ┌──────────────────────┐                │
                                  │ canonicalize / cse   │                │
                                  └──────────────────────┘                │
                                           │                              │
                                           ▼                              │
                                  ┌──────────────────────┐                │
                                  │ func.func(air-       │                │
                                  │ renumber-dma)        │                │
                                  └──────────────────────┘                │
                                           │                              │
                                           └──────────────────────────────┼──┐
                                                                          │  │
                                                            Output: placed.mlir
                                                                          │  │
                                                                          ▼  │
                                                            ┌──────────────────────────────┐
                                                            │ air-to-aie                   │
                                                            │ {emit-while-loop,            │
                                                            │  row-offset, col-offset,     │
                                                            │  device, insert-trace,       │
                                                            │  use-lock-race-condition-fix}│
                                                            └──────────────────────────────┘
                                                                          │
                                                                          └──────────────┬──┐
                                                                                        │  │
                                                                        Output: aie.mlir
                                                                                        │  │
                                                              ┌─────────────────────────┴──┘
                                                              │
                                                              ▼
                                                        Is NPU Device?
                                                              │
                                                    ┌─────────┴─────────┐
                                                    │                   │
                                                   YES                 NO
                                                    │                   │
                                    ┌───────────────┴──────┐     ┌──────┴───────────────┐
                                    │                      │     │                      │
                                    ▼                      │     ▼                      │
                    [NPU PATH]                             │  [NON-NPU PATH]           │
┌──────────────────────────────┐                           │  ┌──────────────────────┐ │
│ func.func(air-opt-shim-dma-  │                           │  │ air-split-devices    │ │
│ bds{device,shim-dma-tile})   │                           │  │ {output-prefix}      │ │
└──────────────────────────────┘                           │  └──────────────────────┘ │
           │                                               │             │              │
           ▼                                               │             ▼              │
┌──────────────────────────────┐                           │  ┌──────────────────────┐ │
│ canonicalize / cse           │                           │  │ convert-vector-to-   │ │
└──────────────────────────────┘                           │  │ llvm                 │ │
           │                                               │  │ convert-math-to-llvm │ │
           ▼                                               │  │ func.func(air-label- │ │
┌──────────────────────────────┐                           │  │ broadcast-channel)   │ │
│ air-to-std                   │                           │  └──────────────────────┘ │
│ symbol-dce                   │                           │             │              │
│ affine-expand-index-ops      │                           │             ▼              │
└──────────────────────────────┘                           │  ┌──────────────────────┐ │
           │                                               │  │ lower-affine         │ │
           ▼                                               │  │ func.func(air-opt-   │ │
┌──────────────────────────────┐                           │  │ shim-dma-bds)        │ │
│ canonicalize / cse           │                           │  └──────────────────────┘ │
└──────────────────────────────┘                           │             │              │
           │                                               │             ▼              │
           ▼                                               │  ┌──────────────────────┐ │
┌──────────────────────────────┐                           │  │ air-to-std           │ │
│ airrt-to-npu{trace-size,     │                           │  │ air-lower-linalg-    │ │
│ trace-offset}                │                           │  │ tensors              │ │
└──────────────────────────────┘                           │  │ canonicalize / cse   │ │
           │                                               │  └──────────────────────┘ │
           ▼                                               │             │              │
┌──────────────────────────────┐                           │             ▼              │
│ canonicalize / cse           │                           │  ┌──────────────────────┐ │
└──────────────────────────────┘                           │  │ airrt-to-llvm        │ │
           │                                               │  │ one-shot-bufferize   │ │
      ┌────┴────────────────────┐                          │  └──────────────────────┘ │
      │                         │                          │             │              │
      ▼                         │                          │             ▼              │
Output: npu.*.mlir              │                          │  ┌──────────────────────┐ │
      │                         │                          │  │ expand-strided-      │ │
      ▼                         │                          │  │ metadata             │ │
┌──────────────────────────────┐│                          │  │ lower-affine         │ │
│ aiecc.run()                  ││                          │  │ convert-scf-to-cf    │ │
│ {aiecc options}              ││                          │  │ finalize-memref-to-  │ │
│ Generate xclbin/txn          ││                          │  │ llvm                 │ │
└──────────────────────────────┘│                          │  │ convert-func-to-llvm │ │
           │                    │                          │  │ convert-arith-to-    │ │
           ▼                    │                          │  │ llvm                 │ │
Output: .xclbin/.insts.bin      │                          │  │ convert-cf-to-llvm   │ │
                                │                          │  └──────────────────────┘ │
                                │                          │             │              │
                                │                          │             ▼              │
                                │                          │  ┌──────────────────────┐ │
                                │                          │  │ aie-translate        │ │
                                │                          │  │ --mlir-to-llvmir     │ │
                                │                          │  └──────────────────────┘ │
                                │                          │             │              │
                                │                          │             ▼              │
                                │                          │  ┌──────────────────────┐ │
                                │                          │  │ opt -O3              │ │
                                │                          │  │ (LLVM optimize)      │ │
                                │                          │  └──────────────────────┘ │
                                │                          │             │              │
                                │                          │             ▼              │
                                │                          │  ┌──────────────────────┐ │
                                │                          │  │ llvm-dis             │ │
                                │                          │  │ (bitcode→IR)         │ │
                                │                          │  └──────────────────────┘ │
                                │                          │             │              │
                                │                          │             ▼              │
                                │                          │  ┌──────────────────────┐ │
                                │                          │  │ llc -O3              │ │
                                │                          │  │ (compile to .o)      │ │
                                │                          │  └──────────────────────┘ │
                                │                          │             │              │
                                │                          │             ▼              │
                                │                          │  ┌──────────────────────┐ │
                                │                          │  │ For each segment:    │ │
                                │                          │  │  ├─ air-translate    │ │
                                │                          │  │  ├─ aiecc.py         │ │
                                │                          │  │  ├─ generate .inc    │ │
                                │                          │  │  └─ cc (wrapper)     │ │
                                │                          │  └──────────────────────┘ │
                                │                          │             │              │
                                │                          │             ▼              │
                                │                          │  ┌──────────────────────┐ │
                                │                          │  │ Link all objects     │ │
                                │                          │  │  ├─ clang -shared    │ │
                                │                          │  │  │ OR                │ │
                                │                          │  │  └─ llvm-ar rc       │ │
                                │                          │  └──────────────────────┘ │
                                │                          │             │              │
                                └──────────────────────────┼─────────────┘              │
                                                           │                           │
                                                    Output: Library (.so/.a)          │
                                                           │                           │
                                                           └───────────────────────────┘


================================================================================
