// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
// CHECK: scf.for
// CHECK-NOT: csl.get_mem_dsd
module {
  csl.wafer @oob {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<64xf32>
      %c = csl.var @c : memref<64xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %im1 = arith.subi %i, %c1 : index
          %v = memref.load %a[%im1] : memref<64xf32>
          memref.store %v, %c[%i] : memref<64xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {}
  }
}
