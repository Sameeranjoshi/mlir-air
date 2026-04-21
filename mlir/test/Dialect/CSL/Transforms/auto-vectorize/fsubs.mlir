// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// c[i] = a[i] - b[i]  →  @fsubs(dc, da, db)

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// CHECK: csl.builtin_call "fsubs"(%{{.*}}, %{{.*}}, %{{.*}}) : (!csl.dsd, !csl.dsd, !csl.dsd) -> ()

module {
  csl.wafer @vecsub {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<512xf32>
      %b = csl.var @b : memref<512xf32>
      %c = csl.var @c : memref<512xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 512 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<512xf32>
          %vb = memref.load %b[%i] : memref<512xf32>
          %vc = arith.subf %va, %vb : f32
          memref.store %vc, %c[%i] : memref<512xf32>
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
