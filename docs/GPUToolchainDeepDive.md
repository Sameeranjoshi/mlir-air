# Teaching Moment: MLIR-AIR GPU Toolchain Deep Dive

This document explains the internal transformation steps that occur when you run `aircc.py --target gpu` on an AIR MLIR file. We use the `test/gpu/4k_4k_mul/air_sync.mlir` matrix multiplication example as our reference.

## 1. Input: Higher-Level AIR Dialect

In the starting MLIR, we describe computations using **AIR (AI Runtime)** operations. The key concepts are:
- `air.launch`: Defines the compute domain (e.g., a grid of work).
- `air.segment`: A unit of work that typically maps to a GPU kernel or an AIE segment.
- `air.herd`: A parallel group of "tiles" or threads.

```mlir
// Original input snippet
func.func @forward(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) {
  %c32 = arith.constant 32 : index
  air.launch (%arg3, %arg4) in (%arg5=%c32, %arg6=%c32) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) {
    air.segment @forward_0 args(...) {
      air.herd @herd_0 tile (%arg31, %arg32) in (%arg33=%c256, %arg34=%c1) {
         // parallel work here...
      }
    }
  }
}
```

## 2. Step 1: Lowering AIR to GPU Dialect
**File:** `air_sync_step1_rocdl.mlir`

The compiler first maps AIR's hierarchy onto the standard **MLIR GPU Dialect**. 
- `air.launch` and `air.segment` are flattened.
- `air.herd` becomes a `gpu.launch` operation.
- The `workgroup` (shared memory) and `private` (registers) memrefs are explicitly allocated.

```mlir
// Lowered to GPU Dialect
gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c32, %arg10 = %c32, %arg11 = %c1) 
           threads(%arg6, %arg7, %arg8) in (%arg12 = %c256, %arg13 = %c1_1, %arg14 = %c1_0) 
           workgroup(%arg15 : memref<128x8xf32, 3>, ...) {
    // Computations now use gpu.thread_id and standard arith/memref ops
    %thread_id_x = gpu.thread_id x
    ...
}
```

## 3. Step 2: GPU Kernel Outlining
**File:** `air_sync_step2_outlined.mlir`

Next, the compiler "outlines" the code inside the `gpu.launch` block. 
- It creates a separate `gpu.module` containing a `gpu.func`.
- The original `gpu.launch` is replaced by a `gpu.launch_func` which calls this new kernel.
- This is necessary because the kernel will eventually be compiled into a separate binary (GCN/HSA).

```mlir
// Host side: Calls the kernel
gpu.launch_func @forward_module::@forward_module blocks in (%c32, %c32, %c1) ...

// Kernel side: The actual code shifted to a module
gpu.module @forward_module {
  gpu.func @forward_module(...) kernel {
    // Kernel body...
  }
}
```

## 4. Step 3: Lowering to ROCDL & Control Flow
**File:** `air_sync_step3_gpu.mlir`

Now the kernel code itself is lowered to its final IR form before binary generation.
- High-level loops (`scf.for`) are converted to low-level branches (`cf.br`, `cf.cond_br`).
- Memory operations map to their respective address spaces (AS 3 for shared, AS 5 for private).
- Hardware-specific intrinsics (like `rocdl.barrier`) are inserted.

```mlir
// Low-level form (CFG)
^bb1(%4: index):
  %5 = arith.cmpi slt, %4, %c64 : index
  cf.cond_br %5, ^bb2, ^bb3
^bb2:
  memref.store %cst, %arg7[%4] : memref<64xf32, 5>
  %6 = arith.addi %4, %c1 : index
  cf.br ^bb1(%6 : index)
```

## 5. Final Stage: Host-to-LLVM & Binary (The "Missing Link")
**Final output:** `output.mlir` (if linking succeeded)

The last step (which requires a GPU/ROCm environment) would be:
1. **Binary Generation**: The `gpu.module` is compiled into a GCN binary using `lld`.
2. **Embedding**: The binary is serialized and embedded as a constant string in the MLIR module.
3. **Execution Engine**: `gpu.launch_func` is lowered to calls into the ROCm/HIP runtime libraries (`libmlir_rocm_runtime.so`) to load and run that binary.

```mlir
// Final Low-level Host Code Calling the Runtime
%runtime_ptr = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
llvm.call @mgpuLaunchKernel(%kernel_ptr, %gridX, ...) : (...) -> ()
```

---
**Summary**: You've just seen a "source-to-source-to-binary" transformation where high-level parallel intent is gradually refined into hardware-specific machine code! 🚀
