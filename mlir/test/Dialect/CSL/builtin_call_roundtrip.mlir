//===- builtin_call_roundtrip.mlir - csl.builtin_call round-trip --*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Round-trip tests for csl.builtin_call — the single generic op that covers
// every CSL language builtin (@fmacs, @fadds, @fmovs, @fabs, @iadds, ...) and
// every imported-module member call (math.sqrt, layout.get_x_coord, ...).
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --verify-roundtrip %s | FileCheck %s

// ---- Bare builtin: @fmacs with DSD + DSD + DSD + scalar operand ----

// CHECK-LABEL: csl.wafer @bc_fmacs
// CHECK: csl.builtin_call "fmacs"
// CHECK-SAME: (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
module {
  csl.wafer @bc_fmacs {arch = "wse3"} {
    csl.program @pe {
      %A = csl.var @A : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      csl.func @compute {
        %n    = arith.constant 128 : index
        %scal = arith.constant 2.0 : f32
        %Ad   = csl.get_mem_dsd %A : memref<128xf32> -> !csl.dsd
        %yd   = csl.get_mem_dsd %y : memref<128xf32> -> !csl.dsd
        csl.builtin_call "fmacs"(%yd, %yd, %Ad, %scal)
            : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
        csl.return
      }
    }
  }
}

// ---- Bare builtin: @fadds (no scalar) ----

// CHECK-LABEL: csl.wafer @bc_fadds
// CHECK: csl.builtin_call "fadds"
// CHECK-SAME: (!csl.dsd, !csl.dsd, !csl.dsd) -> ()
module {
  csl.wafer @bc_fadds {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<64xf32>
      %y = csl.var @y : memref<64xf32>
      %z = csl.var @z : memref<64xf32>
      csl.func @compute {
        %n  = arith.constant 64 : index
        %xd = csl.get_mem_dsd %x : memref<64xf32> -> !csl.dsd
        %yd = csl.get_mem_dsd %y : memref<64xf32> -> !csl.dsd
        %zd = csl.get_mem_dsd %z : memref<64xf32> -> !csl.dsd
        csl.builtin_call "fadds"(%zd, %xd, %yd)
            : (!csl.dsd, !csl.dsd, !csl.dsd) -> ()
        csl.return
      }
    }
  }
}

// ---- Module-member form: math.sqrt ----

// CHECK-LABEL: csl.wafer @bc_math
// CHECK: csl.import_module "<math>"
// CHECK: csl.builtin_call "sqrt" in
// CHECK-SAME: (f32) -> f32
module {
  csl.wafer @bc_math {arch = "wse3"} {
    csl.program @pe {
      %math = csl.import_module "<math>" : !csl.imported_module
      csl.func @compute {
        %x = arith.constant 4.0 : f32
        %r = csl.builtin_call "sqrt" in %math (%x) : (f32) -> f32
        csl.return
      }
    }
  }
}

// ---- Builtin returning a scalar (no module) ----

// CHECK-LABEL: csl.wafer @bc_ret
// CHECK: csl.builtin_call "fabs"
// CHECK-SAME: (f32) -> f32
module {
  csl.wafer @bc_ret {arch = "wse3"} {
    csl.program @pe {
      csl.func @compute {
        %x = arith.constant -1.5 : f32
        %a = csl.builtin_call "fabs"(%x) : (f32) -> f32
        csl.return
      }
    }
  }
}
