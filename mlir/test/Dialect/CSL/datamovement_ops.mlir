//===- datamovement_ops.mlir - CSL data movement ops tests ----*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Round-trip tests for CSL data movement ops.
//
// v5 scope: csl.get_mem_dsd (mem1d only).
// Deferred: csl.get_fab_dsd, csl.mov (fabric DSDs require colors/routes which
// are intentionally excluded from SIMD-only execution).
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --verify-roundtrip %s | FileCheck %s

// ---- csl.get_mem_dsd — mem1d over f32 memref ----

// CHECK-LABEL: csl.wafer @dsd_f32
// CHECK:   csl.program @pe
// CHECK:     csl.var @a
// CHECK:     csl.func @compute
// CHECK:       csl.get_mem_dsd
// CHECK-SAME:  memref<128xf32>
// CHECK-SAME:  index
// CHECK-SAME:  !csl.dsd
module {
  csl.wafer @dsd_f32 {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<128xf32>
      csl.func @compute {
        %len = arith.constant 128 : index
        %d = csl.get_mem_dsd %a, %len : memref<128xf32>, index -> !csl.dsd
        csl.return
      }
    }
  }
}

// ---- csl.get_mem_dsd — mem1d over i32 memref ----

// CHECK-LABEL: csl.wafer @dsd_i32
// CHECK:       csl.get_mem_dsd
// CHECK-SAME:  memref<64xi32>
// CHECK-SAME:  !csl.dsd
module {
  csl.wafer @dsd_i32 {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<64xi32>
      csl.func @compute {
        %len = arith.constant 64 : index
        %d = csl.get_mem_dsd %a, %len : memref<64xi32>, index -> !csl.dsd
        csl.return
      }
    }
  }
}

// ---- csl.get_mem_dsd — dynamic length (SSA index) ----

// CHECK-LABEL: csl.wafer @dsd_dynamic_len
// CHECK:       csl.get_mem_dsd
// CHECK-SAME:  memref<?xf32>
// CHECK-SAME:  index
// CHECK-SAME:  !csl.dsd
module {
  csl.wafer @dsd_dynamic_len {arch = "wse3"} {
    csl.program @pe(%n: !csl.comptime<index>) {
      %a = csl.var @a : memref<?xf32>
      csl.func @compute {
        %len = arith.constant 32 : index
        %d = csl.get_mem_dsd %a, %len : memref<?xf32>, index -> !csl.dsd
        csl.return
      }
    }
  }
}
