// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// c[i] = a[i]  ->  @fmovs(dc, da)
// (pure buffer copy, no arith op in body)

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// CHECK: csl.builtin_call "fmovs"(%{{.*}}, %{{.*}}) : (!csl.dsd, !csl.dsd) -> ()

module {
  csl.wafer @veccopy {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<256xf32>
      %c = csl.var @c : memref<256xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 256 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %v = memref.load %a[%i] : memref<256xf32>
          memref.store %v, %c[%i] : memref<256xf32>
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
