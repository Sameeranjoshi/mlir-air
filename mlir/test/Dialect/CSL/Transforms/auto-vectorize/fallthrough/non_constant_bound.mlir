// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
// CHECK: scf.for
// CHECK-NOT: csl.get_mem_dsd
//
// The upper bound is loaded from a memref at runtime — not a compile-time
// constant — so analyzeForLoop rejects it ("non-constant bounds or step").
module {
  csl.wafer @nonconst {arch = "wse3"} {
    csl.program @pe {
      %a   = csl.var @a   : memref<64xf32>
      %c   = csl.var @c   : memref<64xf32>
      %ub_buf = csl.var @ub_buf : memref<1xindex>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        // Load the upper bound at runtime — this is NOT a constant.
        %dyn_ub = memref.load %ub_buf[%c0] : memref<1xindex>
        scf.for %i = %c0 to %dyn_ub step %c1 {
          %v = memref.load %a[%i] : memref<64xf32>
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
