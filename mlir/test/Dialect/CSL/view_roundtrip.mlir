//===- view_roundtrip.mlir - !csl.view + strided DSD round-trip --*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Round-trip tests for csl.view.strided and the strided form of
// csl.get_mem_dsd (DSD with optional !csl.view operand).
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --verify-roundtrip %s | FileCheck %s

// ---- Reusable view: one %v applied to two buffers ----

// CHECK-LABEL: csl.wafer @view_reuse
// CHECK: %[[V:.*]] = csl.view.strided %{{.*}}, %{{.*}}, %{{.*}} : !csl.view
// CHECK: csl.get_mem_dsd %{{.*}}, %{{.*}} view %[[V]]
// CHECK: csl.get_mem_dsd %{{.*}}, %{{.*}} view %[[V]]
module {
  csl.wafer @view_reuse {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      csl.func @compute {
        %n   = arith.constant 128 : index
        %ext = arith.constant  64 : index
        %str = arith.constant   2 : index
        %off = arith.constant   0 : index
        %v   = csl.view.strided %ext, %str, %off : !csl.view
        %xd  = csl.get_mem_dsd %x, %n view %v
                 : memref<128xf32>, index, !csl.view -> !csl.dsd
        %yd  = csl.get_mem_dsd %y, %n view %v
                 : memref<128xf32>, index, !csl.view -> !csl.dsd
        csl.return
      }
    }
  }
}

// ---- Contiguous form still parses (no view operand) ----

// CHECK-LABEL: csl.wafer @view_none
// CHECK: csl.get_mem_dsd
// CHECK-NOT: view
module {
  csl.wafer @view_none {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<32xf32>
      csl.func @compute {
        %n = arith.constant 32 : index
        %d = csl.get_mem_dsd %a, %n : memref<32xf32>, index -> !csl.dsd
        csl.return
      }
    }
  }
}

// ---- Non-zero offset ----

// CHECK-LABEL: csl.wafer @view_offset
// CHECK: csl.view.strided
// CHECK: csl.get_mem_dsd
// CHECK-SAME: view
module {
  csl.wafer @view_offset {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<256xf32>
      csl.func @compute {
        %n   = arith.constant 256 : index
        %ext = arith.constant 128 : index
        %str = arith.constant   1 : index
        %off = arith.constant  16 : index
        %v   = csl.view.strided %ext, %str, %off : !csl.view
        %d   = csl.get_mem_dsd %a, %n view %v
                 : memref<256xf32>, index, !csl.view -> !csl.dsd
        csl.return
      }
    }
  }
}
