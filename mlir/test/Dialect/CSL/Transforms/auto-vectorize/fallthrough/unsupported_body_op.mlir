// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
// CHECK: scf.for
// CHECK-NOT: csl.get_mem_dsd
module {
  csl.wafer @sqrt {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<64xf32>
      %c = csl.var @c : memref<64xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %v = memref.load %a[%i] : memref<64xf32>
          %r = math.sqrt %v : f32
          memref.store %r, %c[%i] : memref<64xf32>
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
