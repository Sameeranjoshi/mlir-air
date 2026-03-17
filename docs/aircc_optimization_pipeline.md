# AIR Compiler (aircc) Non-GPU Optimization Pipeline

Reference list of optimization passes and flags for the **AIE** (non-GPU) compilation path in `python/air/compiler/aircc/main.py`, in execution order. GPU path is separate (`run_gpu_compilation`).

**Copy-paste pipelines:** For **all passes from input AIR to AIE (and NPU)** in one place, see [Full pipeline strings: input → placed AIR → AIE → NPU](#full-pipeline-strings-input--placed-air--aie--npu).

---

## Execution order overview

1. **Phase 1 — Setup & herd lowering** (always, for NPU and non-NPU)
2. **Phase 2 — AIR optimizations** (only when `"npu" in device`)
3. **Phase 3 — Placement & numbering** (always)
4. **Phase 4 — AIR → AIE** (always)
5. **Phase 5a — NPU: AIR → std → NPU** (when `"npu" in device`)
6. **Phase 5b — Non-NPU: split devices, AIRRt, LLVM** (when device is not NPU)

---

## Phase 1 — Setup & herd lowering

*Always run; first part of the single “placement” pipeline.*


| Order | Pass                                                 | Category           |
| ----- | ---------------------------------------------------- | ------------------ |
| 1     | `air-insert-launch-around-herd{insert-segment=true}` | Structure / launch |
| 2     | `func.func(air-lower-herd-parallel)`                 | Herd lowering      |
| 3     | `scf-forall-to-parallel`                             | SCF normalization  |


---

## Phase 2 — AIR optimizations

*Run only when `"npu" in opts.device`; built by `get_air_optimization_pass()`.*

### 2.1 Dependency & async


| Order | Pass                             | Controlling flag |
| ----- | -------------------------------- | ---------------- |
| 4     | `air-dependency`                 | —                |
| 5     | `air-hoist-dma-in-accum-pattern` | —                |


### 2.2 Broadcast (optional)


| Order | Pass                           | Controlling flag                   |
| ----- | ------------------------------ | ---------------------------------- |
| 6     | `air-broadcast-detection`      | Omitted if `--omit-auto-broadcast` |
| 7     | `air-specialize-dma-broadcast` | Omitted if `--omit-auto-broadcast` |


### 2.3 DMA to channel & dependency cleanup


| Order | Pass                               | Controlling flag |
| ----- | ---------------------------------- | ---------------- |
| 8     | `air-dma-to-channel`               | —                |
| 9     | `canonicalize`                     | —                |
| 10    | `cse`                              | —                |
| 11    | `air-dependency-canonicalize`      | —                |
| 12    | `canonicalize`                     | —                |
| 13    | `cse`                              | —                |
| 14    | `air-isolate-async-dma-loop-nests` | —                |
| 15    | `canonicalize`                     | —                |
| 16    | `cse`                              | —                |


### 2.4 Channel fusion


| Order | Pass                                                            | Controlling flag                                |
| ----- | --------------------------------------------------------------- | ----------------------------------------------- |
| 17    | `air-fuse-channels` or `air-fuse-channels{aggressive-mode=...}` | `--channel-multiplexing` for aggressive variant |
| 18    | `canonicalize`                                                  | —                                               |
| 19    | `cse`                                                           | —                                               |


### 2.5 L2 splitting (optional)

*Skipped when `device == "npu_1col"`.*


| Order | Pass                               | Controlling flag       |
| ----- | ---------------------------------- | ---------------------- |
| 20    | `func.func(air-split-l2-memref)`   | Skipped for `npu_1col` |
| 21    | `canonicalize`                     | —                      |
| 22    | `cse`                              | —                      |
| 23    | `air-isolate-async-dma-loop-nests` | —                      |
| 24    | `canonicalize`                     | —                      |
| 25    | `cse`                              | —                      |


### 2.6 Loop fusion vs alloc/shrink

*One branch or the other.*


| Order | Pass                                           | Controlling flag                     |
| ----- | ---------------------------------------------- | ------------------------------------ |
| 26a   | `func.func(air-loop-fusion)`                   | `--air-loop-fusion`                  |
| 26b   | `func.func(air-fuse-alloc-dealloc)`            | Default when not `--air-loop-fusion` |
| 26b   | `func.func(air-shrink-memref-sizes-by-access)` | Default when not `--air-loop-fusion` |


### 2.7 Ping-pong / double buffering (optional)

*Run when `omit_pingpong in ["", "L1", "L2"]`; can restrict to memory space via options.*


| Order | Pass                                                                                          | Controlling flag                     |
| ----- | --------------------------------------------------------------------------------------------- | ------------------------------------ |
| 27    | `air-label-scf-for-to-ping-pong` or `air-label-scf-for-to-ping-pong{omit-memory-space=L1|L2}` | `--omit-pingpong` ("" / "L1" / "L2") |
| 28    | `air-ping-pong-transform` or `air-ping-pong-transform{omit-memory-space=L1|L2}`               | Same                                 |
| 29    | `canonicalize`                                                                                | —                                    |
| 30    | `cse`                                                                                         | —                                    |


### 2.8 Linalg lowering

*One or the other.*


| Order | Pass                                   | Controlling flag                         |
| ----- | -------------------------------------- | ---------------------------------------- |
| 31a   | `air-linalg-to-func{link-with=<file>}` | `--lower-linalg-to-func <file>`          |
| 31b   | `func.func(convert-linalg-to-loops)`   | Default when no `--lower-linalg-to-func` |


### 2.9 Memtile DMA BD optimization


| Order | Pass                                                  | Controlling flag |
| ----- | ----------------------------------------------------- | ---------------- |
| 32    | `func.func(air-opt-memtile-dma-bds{device=<device>})` | —                |
| 33    | `canonicalize`                                        | —                |
| 34    | `cse`                                                 | —                |


---

## Phase 3 — Placement & numbering

*Always run; end of the placement pipeline.*


| Order | Pass                                                                          | Category      |
| ----- | ----------------------------------------------------------------------------- | ------------- |
| 35    | `func.func(air-collapse-herd{max-col-size=4})`                                | Herd shape    |
| 36    | `canonicalize`                                                                | —             |
| 37    | `cse`                                                                         | —             |
| 38    | `air-place-herds{num-rows=..., num-cols=..., row-anchor=..., col-anchor=...}` | Placement     |
| 39    | `canonicalize`                                                                | —             |
| 40    | `cse`                                                                         | —             |
| 41    | `func.func(air-renumber-dma)`                                                 | DMA numbering |


---

## Phase 4 — AIR → AIE

*Single pass; separate pipeline.*


| Order | Pass                                                                                                                                         | Options (from CLI)                                                              |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 42    | `air-to-aie{emit-while-loop=..., row-offset=..., col-offset=..., device=..., insert-trace-packet-flow=..., use-lock-race-condition-fix=...}` | `--omit-while-true-loop`, `--trace-size`, `--use-lock-race-condition-fix`, etc. |


---

## Phase 5a — NPU: control path to std and NPU

*Run only when `"npu" in opts.device`.*


| Order | Pass                                             | Category    |
| ----- | ------------------------------------------------ | ----------- |
| 43    | `func.func(air-opt-shim-dma-bds{device=...})`    | Shim DMA BD |
| 44    | `canonicalize`                                   | —           |
| 45    | `cse`                                            | —           |
| 46    | `air-to-std`                                     | AIR → std   |
| 47    | `symbol-dce`                                     | DCE         |
| 48    | `affine-expand-index-ops`                        | Affine      |
| 49    | `canonicalize`                                   | —           |
| 50    | `cse`                                            | —           |
| 51    | `airrt-to-npu{trace-size=..., trace-offset=...}` | AIRRt → NPU |


---

## Phase 5b — Non-NPU: device split and host lowering

*Run when device is not NPU (e.g. AIE only).*

### Device split


| Order | Pass                                   | Category        |
| ----- | -------------------------------------- | --------------- |
| 1     | `air-split-devices{output-prefix=...}` | Split by device |


### AIRRt pipeline (to airrt.*.mlir)


| Order | Pass                                               | Category           |
| ----- | -------------------------------------------------- | ------------------ |
| 2     | `convert-vector-to-llvm`                           | Vector lowering    |
| 3     | `convert-math-to-llvm`                             | Math lowering      |
| 4     | `func.func(air-label-broadcast-channel-with-tile)` | Broadcast labeling |
| 5     | `lower-affine`                                     | Affine lowering    |
| 6     | `func.func(air-opt-shim-dma-bds{device=...})`      | Shim DMA BD        |
| 7     | `air-to-std`                                       | AIR → std          |
| 8     | `air-lower-linalg-tensors`                         | Linalg tensors     |
| 9     | `canonicalize`                                     | —                  |
| 10    | `cse`                                              | —                  |


### AIRRt → LLVM (to aie_ctrl.*.mlir)


| Order | Pass                 | Category      |
| ----- | -------------------- | ------------- |
| 11    | `airrt-to-llvm`      | AIRRt → LLVM  |
| 12    | `one-shot-bufferize` | Bufferization |


### Refback pipeline (placed module → refback.*.mlir)


| Order | Pass                                               | Category |
| ----- | -------------------------------------------------- | -------- |
| 1     | `convert-vector-to-llvm`                           | —        |
| 2     | `convert-math-to-llvm`                             | —        |
| 3     | `func.func(air-label-broadcast-channel-with-tile)` | —        |
| 4     | `lower-affine`                                     | —        |
| 5     | `func.func(air-opt-shim-dma-bds{device=...})`      | —        |
| 6     | `air-to-std`                                       | —        |
| 7     | `air-lower-linalg-tensors`                         | —        |
| 8     | `canonicalize`                                     | —        |
| 9     | `cse`                                              | —        |
| 10    | `airrt-to-llvm`                                    | —        |
| 11    | `canonicalize`                                     | —        |
| 12    | `cse`                                              | —        |


### LLVM dialect pipeline (to llvm.*.mlir)


| Order | Pass                      | Category |
| ----- | ------------------------- | -------- |
| 1     | `expand-strided-metadata` | Memref   |
| 2     | `lower-affine`            | —        |
| 3     | `convert-scf-to-cf`       | —        |
| 4     | `finalize-memref-to-llvm` | —        |
| 5     | `convert-func-to-llvm`    | —        |
| 6     | `convert-arith-to-llvm`   | —        |
| 7     | `convert-cf-to-llvm`      | —        |
| 8     | `canonicalize`            | —        |
| 9     | `cse`                     | —        |


### Per-segment air-opt (before aiecc.py)


| Order | Pass                       | Category |
| ----- | -------------------------- | -------- |
| 1     | `air-lower-linalg-tensors` | —        |
| 2     | `lower-affine`             | —        |
| 3     | `canonicalize`             | —        |
| 4     | `cse`                      | —        |


---

## Summary: CLI flags that affect optimization


| Flag                            | Effect                                                                                          |
| ------------------------------- | ----------------------------------------------------------------------------------------------- |
| `--omit-auto-broadcast`         | Skip `air-broadcast-detection` and `air-specialize-dma-broadcast`                               |
| `--omit-pingpong`               | Skip ping-pong passes; use `"L1"` or `"L2"` to omit only that memory space                      |
| `--air-loop-fusion`             | Use `air-loop-fusion` instead of `air-fuse-alloc-dealloc` + `air-shrink-memref-sizes-by-access` |
| `--lower-linalg-to-func <file>` | Use `air-linalg-to-func{link-with=<file>}` instead of `convert-linalg-to-loops`                 |
| `--channel-multiplexing`        | Use aggressive `air-fuse-channels{aggressive-mode=...}`                                         |
| Device name                     | `npu` → run Phase 2; `npu_1col` → skip L2 splitting block                                       |


---

## Full pipeline strings: input → placed AIR → AIE → NPU
### GPU pipeline

```bash
air-opt mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  --pass-pipeline="builtin.module(air-dependency,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-specialize-dma-broadcast,air-dma-to-channel,canonicalize,cse,func.func(air-split-l2-memref),canonicalize,cse,func.func(air-collapse-herd{max-col-size=4}),canonicalize,cse,air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},canonicalize,cse,func.func(air-renumber-dma))" \
  --mlir-print-ir-after-all
```

Below are **single pass-pipeline strings** that run **all** passes from the doc in order, for the **NPU path** with default options (broadcast on, L2 splitting on, default loop/alloc path, ping-pong on, `convert-linalg-to-loops`). Use these with `air-opt --pass-pipeline="..."` or with `--mlir-print-ir-after-all` to see IR after every pass.

**Build requirement:** The full pipeline (sections 1–5 below) uses **AIE-only** passes. You must build mlir-air with **AIE enabled** (`AIR_ENABLE_AIE=ON`). If you see:

```text
'air-isolate-async-dma-loop-nests' does not refer to a registered pass or pass pipeline
```

(or similar for `air-fuse-channels`, `air-hoist-dma-in-accum-pattern`, `air-opt-memtile-dma-bds`, `air-to-aie`, etc.), your `air-opt` was built **without** AIE support (e.g. GPU-only). Rebuild with AIE enabled so those passes are registered. See the project’s build docs for how to set `AIR_ENABLE_AIE`.

**Input:** Use **AIR dialect** input for AIE (e.g. `mlir/test/Transform/AIRDependency/matmul_nd.mlir`). Files like `test/gpu/4k_4k_mul/air_sync.mlir` are intended for the **GPU** path (`air-opt -air-to-rocdl`), not for this AIE pipeline.

**Configuration used:** `device=npu1_4col`, placement `num-rows=4 num-cols=4 row-anchor=2 col-anchor=0`. For other devices or flags, adjust the options inside the pass names (e.g. `device=...`, or omit L2/ping-pong/broadcast per the table above).

### 1. Phases 1 + 2 + 3 only (input AIR → placed AIR)

Gets you from input AIR to placed, renumbered AIR (before `air-to-aie`).

```bash
--pass-pipeline="builtin.module(
  air-insert-launch-around-herd{insert-segment=true},
  func.func(air-lower-herd-parallel),
  scf-forall-to-parallel,
  air-dependency,
  air-hoist-dma-in-accum-pattern,
  air-broadcast-detection,
  air-specialize-dma-broadcast,
  air-dma-to-channel,
  canonicalize,cse,
  air-dependency-canonicalize,
  canonicalize,cse,
  air-isolate-async-dma-loop-nests,
  canonicalize,cse,
  air-fuse-channels,
  canonicalize,cse,
  func.func(air-split-l2-memref),
  canonicalize,cse,
  air-isolate-async-dma-loop-nests,
  canonicalize,cse,
  func.func(air-fuse-alloc-dealloc),
  func.func(air-shrink-memref-sizes-by-access),
  air-label-scf-for-to-ping-pong,
  air-ping-pong-transform,
  canonicalize,cse,
  func.func(convert-linalg-to-loops),
  func.func(air-opt-memtile-dma-bds{device=npu1_4col}),
  canonicalize,cse,
  func.func(air-collapse-herd{max-col-size=4}),
  canonicalize,cse,
  air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},
  canonicalize,cse,
  func.func(air-renumber-dma)
)"
```

**Single line (copy-paste):**

```
builtin.module(air-insert-launch-around-herd{insert-segment=true},func.func(air-lower-herd-parallel),scf-forall-to-parallel,air-dependency,air-hoist-dma-in-accum-pattern,air-broadcast-detection,air-specialize-dma-broadcast,air-dma-to-channel,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-isolate-async-dma-loop-nests,canonicalize,cse,air-fuse-channels,canonicalize,cse,func.func(air-split-l2-memref),canonicalize,cse,air-isolate-async-dma-loop-nests,canonicalize,cse,func.func(air-fuse-alloc-dealloc),func.func(air-shrink-memref-sizes-by-access),air-label-scf-for-to-ping-pong,air-ping-pong-transform,canonicalize,cse,func.func(convert-linalg-to-loops),func.func(air-opt-memtile-dma-bds{device=npu1_4col}),canonicalize,cse,func.func(air-collapse-herd{max-col-size=4}),canonicalize,cse,air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},canonicalize,cse,func.func(air-renumber-dma))
```

### 2. Phase 4 only (placed AIR → AIE dialect)

Run on the **output of step 1** (placed AIR). Converts AIR to AIE dialect.

```bash
--pass-pipeline="builtin.module(air-to-aie{emit-while-loop=true row-offset=2 col-offset=0 device=npu1_4col use-lock-race-condition-fix=false})"
```

### 3. Phases 1 + 2 + 3 + 4 in one run (input AIR → AIE dialect)

Same as step 1 with Phase 4 appended. One `air-opt` run from input AIR to AIE dialect.

```bash
--pass-pipeline="builtin.module(
  air-insert-launch-around-herd{insert-segment=true},
  func.func(air-lower-herd-parallel),
  scf-forall-to-parallel,
  air-dependency,
  air-hoist-dma-in-accum-pattern,
  air-broadcast-detection,
  air-specialize-dma-broadcast,
  air-dma-to-channel,
  canonicalize,cse,
  air-dependency-canonicalize,
  canonicalize,cse,
  air-isolate-async-dma-loop-nests,
  canonicalize,cse,
  air-fuse-channels,
  canonicalize,cse,
  func.func(air-split-l2-memref),
  canonicalize,cse,
  air-isolate-async-dma-loop-nests,
  canonicalize,cse,
  func.func(air-fuse-alloc-dealloc),
  func.func(air-shrink-memref-sizes-by-access),
  air-label-scf-for-to-ping-pong,
  air-ping-pong-transform,
  canonicalize,cse,
  func.func(convert-linalg-to-loops),
  func.func(air-opt-memtile-dma-bds{device=npu1_4col}),
  canonicalize,cse,
  func.func(air-collapse-herd{max-col-size=4}),
  canonicalize,cse,
  air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},
  canonicalize,cse,
  func.func(air-renumber-dma),
  air-to-aie{emit-while-loop=true row-offset=2 col-offset=0 device=npu1_4col use-lock-race-condition-fix=false}
)"
```

### 4. Phase 5a (AIE module → NPU control code)

Run on the **output of step 2 or 3** (module in AIE dialect). Lowers to std and then to NPU (airrt-to-npu). Default trace: `trace-size=0 trace-offset=0`.

```bash
--pass-pipeline="builtin.module(
  func.func(air-opt-shim-dma-bds{device=npu1_4col}),
  canonicalize,cse,
  air-to-std,
  symbol-dce,
  affine-expand-index-ops,
  canonicalize,cse,
  airrt-to-npu{trace-size=0 trace-offset=0},
  canonicalize,cse
)"
```

### 5. Run full pipeline (1→4) with IR after every pass

Use the **single-line** pipeline from section 3 inside the quotes, and add the print flag:

```bash
air-opt your_input.mlir \
  --pass-pipeline="builtin.module(air-insert-launch-around-herd{insert-segment=true},func.func(air-lower-herd-parallel),scf-forall-to-parallel,air-dependency,air-hoist-dma-in-accum-pattern,air-broadcast-detection,air-specialize-dma-broadcast,air-dma-to-channel,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-isolate-async-dma-loop-nests,canonicalize,cse,air-fuse-channels,canonicalize,cse,func.func(air-split-l2-memref),canonicalize,cse,air-isolate-async-dma-loop-nests,canonicalize,cse,func.func(air-fuse-alloc-dealloc),func.func(air-shrink-memref-sizes-by-access),air-label-scf-for-to-ping-pong,air-ping-pong-transform,canonicalize,cse,func.func(convert-linalg-to-loops),func.func(air-opt-memtile-dma-bds{device=npu1_4col}),canonicalize,cse,func.func(air-collapse-herd{max-col-size=4}),canonicalize,cse,air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},canonicalize,cse,func.func(air-renumber-dma),air-to-aie{emit-while-loop=true row-offset=2 col-offset=0 device=npu1_4col use-lock-race-condition-fix=false})" \
  --mlir-print-ir-after-all
```

Pipe to a file or `less` (output is large).

### Notes

- **Input:** Step 1 (and 3) expect **AIR dialect** input (e.g. already has `air.launch`, `air.herd`, `air.dma_memcpy_nd`). Tests under `mlir/test/Transform/AIRDependency/` are in that form. If you start from linalg/scf, run the front-end (e.g. air-linalg-codegen, air-par-to-herd, air-par-to-launch, air-copy-to-dma) first; that sequence is not included in the strings above.
- **Phase 5b (non-NPU):** After `air-to-aie`, the non-NPU path uses `air-split-devices`, then separate pipelines for AIRRt, refback, and LLVM, plus `aiecc.py` per segment. There is no single “all-in-one” pass string for that path; see Phase 5b in the doc for the list of passes.
- **Device / placement:** To use another device (e.g. `npu2_4col`) or placement, change `device=npu1_4col` and the `air-place-herds{...}` options in the strings above. For `npu_1col`, remove the L2 block (the two `func.func(air-split-l2-memref),...,air-isolate-async-dma-loop-nests,...` chunks).

### GPU-only build: which passes work

If your build is **GPU-only** (no `AIR_ENABLE_AIE`), only **core** and **GPU** passes are registered. Use the following.

**1. GPU path (AIR → ROCDL → GPU outline)** — for files like `test/gpu/4k_4k_mul/air_sync.mlir`:

```bash
# Step 1: AIR → ROCDL (gpu.launch, etc.)
air-opt test/gpu/4k_4k_mul/air_sync.mlir -air-to-rocdl -o step1.mlir

# Step 2: GPU kernel outlining (optional, for full GPU pipeline)
air-opt step1.mlir -air-gpu-outlining -o step2.mlir
```

Or as a single pipeline:

```bash
air-opt test/gpu/4k_4k_mul/air_sync.mlir \
  --pass-pipeline="builtin.module(air-to-rocdl,air-gpu-outlining)"
```

**2. Core-only pipeline (no AIE passes)** — for AIR dependency tests like `mlir/test/Transform/AIRDependency/matmul_nd.mlir`. Uses only passes that are always registered (no `air-isolate-async-dma-loop-nests`, `air-fuse-channels`, `air-to-aie`, etc.):

```bash
air-opt mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  --pass-pipeline="builtin.module(
    air-dependency,
    canonicalize,cse,
    air-dependency-canonicalize,
    canonicalize,cse,
    air-specialize-dma-broadcast,
    air-dma-to-channel,
    canonicalize,cse,
    func.func(air-split-l2-memref),
    canonicalize,cse,
    func.func(air-collapse-herd{max-col-size=4}),
    canonicalize,cse,
    air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},
    canonicalize,cse,
    func.func(air-renumber-dma)
  )"
```

Single line:

```bash
air-opt mlir/test/Transform/AIRDependency/matmul_nd.mlir --pass-pipeline="builtin.module(air-dependency,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-specialize-dma-broadcast,air-dma-to-channel,canonicalize,cse,func.func(air-split-l2-memref),canonicalize,cse,func.func(air-collapse-herd{max-col-size=4}),canonicalize,cse,air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},canonicalize,cse,func.func(air-renumber-dma))"
```

With IR after every pass:

```bash
air-opt mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  --pass-pipeline="builtin.module(air-dependency,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-specialize-dma-broadcast,air-dma-to-channel,canonicalize,cse,func.func(air-split-l2-memref),canonicalize,cse,func.func(air-collapse-herd{max-col-size=4}),canonicalize,cse,air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},canonicalize,cse,func.func(air-renumber-dma))" \
  --mlir-print-ir-after-all
```

If your build does not include the Conversion library (e.g. no `air-insert-launch-around-herd`), you can still run the core Transform passes above. If `air-place-herds` or `air-split-l2-memref` is not registered in your build, drop that pass from the pipeline and run the remaining ones.


---

## Categories (by purpose)

- **Structure / launch:** insert launch, lower herd parallel, forall→parallel.
- **Dependency / async:** dependency, hoist DMA, dependency-canonicalize, isolate async DMA loops.
- **Broadcast:** broadcast detection, specialize DMA broadcast.
- **Channels:** DMA to channel, fuse channels.
- **L2 / memtile:** split L2 memref, opt memtile DMA BDs.
- **Loop / memory:** loop fusion, fuse alloc/dealloc, shrink memref sizes, ping-pong.
- **Linalg:** linalg-to-func or convert-linalg-to-loops.
- **Placement:** collapse herd, place herds, renumber DMA.
- **Lowering:** air-to-aie, air-to-std, airrt-to-llvm, lower-affine, convert-*-to-llvm, etc.
- **Shim / NPU:** opt shim DMA BDs, airrt-to-npu.

---

## Running passes step-by-step (see IR after each pass)

Use **`air-opt`** with MLIR’s print flags to see the IR after every pass (or only after specific passes).

### Print IR after every pass

```bash
# From repo root; use your install path if different
AIR_OPT=./install/bin/air-opt
TEST_FILE=mlir/test/Transform/AIRDependency/matmul_nd.mlir

# Print IR after each pass (long output; pipe to less or a file)
$AIR_OPT $TEST_FILE \
  --pass-pipeline="builtin.module(air-dependency,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse)" \
  --mlir-print-ir-after-all
```

Useful variants:

- **`--mlir-print-ir-after-all`** — print full IR after **every** pass (verbose).
- **`--mlir-print-ir-before-all`** — print before every pass.
- **`--mlir-print-ir-after=<pass-name>`** — print only after a specific pass (e.g. `--mlir-print-ir-after=air-dependency`). Can be repeated for multiple passes.
- **`--mlir-print-ir-after-change`** — with the “after” flags, only print when the IR actually changed.
- **`--mlir-print-ir-module-scope`** — when printing, always show the top-level module (full module scope).

### Example: only after `air-dependency`

```bash
$AIR_OPT $TEST_FILE \
  --pass-pipeline="builtin.module(air-dependency)" \
  --mlir-print-ir-after=air-dependency
```

### Example: save dumps to a file

```bash
$AIR_OPT $TEST_FILE \
  --pass-pipeline="builtin.module(air-dependency,canonicalize,cse,air-dependency-canonicalize)" \
  --mlir-print-ir-after-all 2>&1 | tee /tmp/air_passes_dump.mlir
```

### Example: minimal “dependency + canonicalize” pipeline

A short pipeline that matches the ACDG docs and is good for inspecting `air-dependency` and cleanup:

```bash
$AIR_OPT mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  --pass-pipeline="builtin.module(air-dependency,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse)" \
  --mlir-print-ir-after-all
```

Each pass name will appear in a comment in the dump (e.g. `// IR Dump after pass: air-dependency`), so you can see the IR after that pass.

