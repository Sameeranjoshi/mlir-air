// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/dot/pe.csl
//
// v5 Task 7 — dot product via scalar scf.for.
// out[0] = sum_i x[i] * y[i].
// Uses scalar accumulation (no DSD reduction) — simpler surface for v5.
// Host pulls out[0] via memcpy_d2h.

// CHECK-LABEL: fn compute() void
// CHECK: while (
// CHECK: var {{.*}} = x[
// CHECK: var {{.*}} = y[
// CHECK: = {{.*}} * {{.*}};
// CHECK: = {{.*}} + {{.*}};
// CHECK: out[

module {
  csl.wafer @dot {arch = "wse3"} {
    csl.program @pe {
      %x   = csl.var @x   : memref<128xf32>
      %y   = csl.var @y   : memref<128xf32>
      %out = csl.var @out : memref<1xf32>
      csl.func @compute {
        %c0 = arith.constant 0   : index
        %c1 = arith.constant 1   : index
        %n  = arith.constant 128 : index
        scf.for %i = %c0 to %n step %c1 {
          %vx = memref.load %x[%i]   : memref<128xf32>
          %vy = memref.load %y[%i]   : memref<128xf32>
          %p  = arith.mulf %vx, %vy  : f32
          %s  = memref.load %out[%c0] : memref<1xf32>
          %s2 = arith.addf %s, %p    : f32
          memref.store %s2, %out[%c0] : memref<1xf32>
        }
        csl.return
      }
      csl.export @x   {alias = "x"}
      csl.export @y   {alias = "y"}
      csl.export @out {alias = "out"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%x_in: memref<128xf32>, %y_in: memref<128xf32>,
                   %o_out: memref<1xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %x_in to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.memcpy_h2d %y_in to @layout::@y
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@out to %o_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1xf32>
    }
  }
}
