// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// y[i] = alpha * a[i] + y[i]  (saxpy, alpha loop-invariant scalar)
// ->  @fmacs(dy, dy, da, alpha) : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// CHECK: csl.builtin_call "fmacs"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()

module {
  csl.wafer @saxpy {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<256xf32>
      %y = csl.var @y : memref<256xf32>
      csl.func @compute {
        %alpha = arith.constant 2.0 : f32
        %c0 = arith.constant 0 : index
        %n  = arith.constant 256 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<256xf32>
          %vy = memref.load %y[%i] : memref<256xf32>
          %m  = arith.mulf %va, %alpha : f32
          %s  = arith.addf %m, %vy : f32
          memref.store %s, %y[%i] : memref<256xf32>
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
