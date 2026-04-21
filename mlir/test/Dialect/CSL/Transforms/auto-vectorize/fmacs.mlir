// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// c[i] = a[i] * b[i] + c[i]  ->  @fmacs(dc, dc, da, db)
// (The expected SDK form; b is placed after da per 4-operand @fmacs signature.)

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// CHECK: csl.builtin_call "fmacs"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!csl.dsd, !csl.dsd, !csl.dsd, !csl.dsd) -> ()

module {
  csl.wafer @vecmac {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<1024xf32>
      %b = csl.var @b : memref<1024xf32>
      %c = csl.var @c : memref<1024xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<1024xf32>
          %vb = memref.load %b[%i] : memref<1024xf32>
          %vc = memref.load %c[%i] : memref<1024xf32>
          %m  = arith.mulf %va, %vb : f32
          %s  = arith.addf %m, %vc : f32
          memref.store %s, %c[%i] : memref<1024xf32>
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
