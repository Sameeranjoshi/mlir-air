//===- datamovement_ops.mlir - CSL data movement ops tests ----*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Round-trip tests for CSL data movement ops.
//
// v5 scope: csl.get_mem_dsd reads shape + strided layout from its memref
// operand's type. Mem1d and mem4d DSDs share the same op. Strided views are
// built via upstream `memref.subview` rather than a custom view op.
//
// Deferred: fabric DSDs + @mov (require colors/routes/tasks).
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --verify-roundtrip %s | FileCheck %s

// ---- Contiguous mem1d over f32 ----

// CHECK-LABEL: csl.wafer @dsd_f32
// CHECK: csl.get_mem_dsd
// CHECK-SAME: memref<128xf32>
// CHECK-SAME: !csl.dsd
module {
  csl.wafer @dsd_f32 {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<128xf32>
      csl.func @compute {
        %d = csl.get_mem_dsd %a : memref<128xf32> -> !csl.dsd
        csl.return
      }
    }
  }
}

// ---- Contiguous mem1d over i32 ----

// CHECK-LABEL: csl.wafer @dsd_i32
// CHECK: csl.get_mem_dsd
// CHECK-SAME: memref<64xi32>
// CHECK-SAME: !csl.dsd
module {
  csl.wafer @dsd_i32 {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<64xi32>
      csl.func @compute {
        %d = csl.get_mem_dsd %a : memref<64xi32> -> !csl.dsd
        csl.return
      }
    }
  }
}

// ---- Strided mem1d via memref.subview (stride 2) ----

// CHECK-LABEL: csl.wafer @dsd_stride2
// CHECK: memref.subview
// CHECK: csl.get_mem_dsd
// CHECK-SAME: strided<[2]>
// CHECK-SAME: !csl.dsd
module {
  csl.wafer @dsd_stride2 {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<128xf32>
      csl.func @compute {
        %v = memref.subview %a[0] [64] [2]
             : memref<128xf32> to memref<64xf32, strided<[2]>>
        %d = csl.get_mem_dsd %v : memref<64xf32, strided<[2]>> -> !csl.dsd
        csl.return
      }
    }
  }
}

// ---- mem4d DSD: 2-D row-major view via memref.reinterpret_cast ----
//
// subview keeps rank; for a flat 1-D buffer we use reinterpret_cast to
// express the 2-D access pattern.

// CHECK-LABEL: csl.wafer @dsd_mem4d
// CHECK: memref.reinterpret_cast
// CHECK: csl.get_mem_dsd
// CHECK-SAME: strided<[16, 1]>
// CHECK-SAME: !csl.dsd
module {
  csl.wafer @dsd_mem4d {arch = "wse3"} {
    csl.program @pe {
      %M = csl.var @M : memref<128xf32>
      csl.func @compute {
        %v = memref.reinterpret_cast %M to offset: [0],
                                     sizes:   [8, 16],
                                     strides: [16, 1]
             : memref<128xf32> to memref<8x16xf32, strided<[16, 1]>>
        %d = csl.get_mem_dsd %v
             : memref<8x16xf32, strided<[16, 1]>> -> !csl.dsd
        csl.return
      }
    }
  }
}
