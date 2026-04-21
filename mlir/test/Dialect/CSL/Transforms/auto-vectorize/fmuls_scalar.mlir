// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// c[i] = alpha * a[i]  (alpha is loop-invariant scalar)  ->  @fmuls(dc, da, alpha)

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// CHECK: csl.builtin_call "fmuls"(%{{.*}}, %{{.*}}, %{{.*}}) : (!csl.dsd, !csl.dsd, f32) -> ()

module {
  csl.wafer @scale {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<128xf32>
      %c = csl.var @c : memref<128xf32>
      csl.func @compute {
        %alpha = arith.constant 2.0 : f32
        %c0 = arith.constant 0 : index
        %n  = arith.constant 128 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<128xf32>
          %v  = arith.mulf %va, %alpha : f32
          memref.store %v, %c[%i] : memref<128xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
    }
  }
}
