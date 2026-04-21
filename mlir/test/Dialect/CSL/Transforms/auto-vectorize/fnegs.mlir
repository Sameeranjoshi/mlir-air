// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// c[i] = -a[i]  ->  @fnegs(dc, da)

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// CHECK: csl.builtin_call "fnegs"(%{{.*}}, %{{.*}}) : (!csl.dsd, !csl.dsd) -> ()

module {
  csl.wafer @vecneg {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<256xf32>
      %c = csl.var @c : memref<256xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 256 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %v = memref.load %a[%i] : memref<256xf32>
          %n0 = arith.negf %v : f32
          memref.store %n0, %c[%i] : memref<256xf32>
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
