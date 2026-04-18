// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/norm_sq/pe.csl
//
// v5 Task 7 — sum of squares: out[0] = sum_i x[i] * x[i].
// (Full norm = sqrt(sum_sq); sqrt arrives with the math-lib follow-up patch.
// The generic csl.builtin_call op already supports the module-member form
// needed for math.sqrt once tests are added.)

// CHECK-LABEL: fn compute() void
// CHECK: while (
// CHECK: var {{.*}} = x[
// CHECK: = {{.*}} * {{.*}};
// CHECK: = {{.*}} + {{.*}};
// CHECK: out[

module {
  csl.wafer @norm_sq {arch = "wse3"} {
    csl.program @pe {
      %x   = csl.var @x   : memref<128xf32>
      %out = csl.var @out : memref<1xf32>
      csl.func @compute {
        %c0 = arith.constant 0   : index
        %c1 = arith.constant 1   : index
        %n  = arith.constant 128 : index
        scf.for %i = %c0 to %n step %c1 {
          %v  = memref.load %x[%i] : memref<128xf32>
          %sq = arith.mulf %v, %v  : f32
          %s  = memref.load %out[%c0] : memref<1xf32>
          %s2 = arith.addf %s, %sq : f32
          memref.store %s2, %out[%c0] : memref<1xf32>
        }
        csl.return
      }
      csl.export @x   {alias = "x"}
      csl.export @out {alias = "out"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%x_in: memref<128xf32>, %o_out: memref<1xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %x_in to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@out to %o_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1xf32>
    }
  }
}
