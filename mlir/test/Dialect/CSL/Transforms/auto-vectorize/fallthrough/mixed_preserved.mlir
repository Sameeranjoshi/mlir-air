// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// Input mixes: (a) a hand-written csl.get_mem_dsd + csl.builtin_call "fmacs"
// that must be preserved verbatim; (b) an unmatched scalar scf.for (extent
// too large: 100_000 > kMaxDsdExtent) that must also survive.  Both pieces
// share the csl.func — pattern must not tangle them.

// CHECK-LABEL: csl.func @compute
// CHECK: csl.get_mem_dsd
// CHECK: csl.builtin_call "fmacs"
// CHECK: scf.for
// CHECK-NOT: csl.get_mem_dsd %a :

module {
  csl.wafer @mixed {arch = "wse3"} {
    csl.program @pe {
      %A = csl.var @A : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      %a = csl.var @a : memref<100000xf32>
      %c = csl.var @c : memref<100000xf32>
      csl.func @compute {
        %scal = arith.constant 2.0 : f32
        %Ad = csl.get_mem_dsd %A : memref<128xf32> -> !csl.dsd
        %yd = csl.get_mem_dsd %y : memref<128xf32> -> !csl.dsd
        csl.builtin_call "fmacs"(%yd, %yd, %Ad, %scal)
            : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()

        %c0 = arith.constant 0 : index
        %n  = arith.constant 100000 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %v = memref.load %a[%i] : memref<100000xf32>
          memref.store %v, %c[%i] : memref<100000xf32>
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
