// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
// CHECK: scf.for
// CHECK-NOT: csl.get_mem_dsd
//
// The loop carries a reduction via iter_args — analyzeForLoop rejects it
// ("loop has iter_args (reduction?)").  The accumulated result is stored to a
// memref so the function body stays well-typed without a return value.
module {
  csl.wafer @reduce {arch = "wse3"} {
    csl.program @pe {
      %a   = csl.var @a   : memref<64xf32>
      %out = csl.var @out : memref<1xf32>
      csl.func @compute {
        %c0   = arith.constant 0 : index
        %n    = arith.constant 64 : index
        %c1   = arith.constant 1 : index
        %zero = arith.constant 0.0 : f32
        %r = scf.for %i = %c0 to %n step %c1 iter_args(%s = %zero) -> f32 {
          %v  = memref.load %a[%i] : memref<64xf32>
          %s2 = arith.addf %s, %v : f32
          scf.yield %s2 : f32
        }
        memref.store %r, %out[%c0] : memref<1xf32>
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {}
  }
}
