// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-auto-vectorize -csl-infer-exports | \
// RUN:   air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/fmuls_scalar_e2e/pe.csl
//
// c[i] = alpha * a[i]  (alpha = 2.0, loop-invariant scalar)
// ->  @fmuls(dc, da, alpha)  with alpha passed as f32 scalar

// CHECK-LABEL: fn compute() void
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &a, .extent = 128 });
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &c, .extent = 128 });
// CHECK: @fmuls(

module {
  csl.wafer @fmuls_scalar_e2e {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<128xf32>
      %c = csl.var @c : memref<128xf32>
      csl.func @compute {
        %alpha = arith.constant 2.0 : f32
        %c0 = arith.constant 0 : index
        %n  = arith.constant 128 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<128xf32>
          %v  = arith.mulf %va, %alpha : f32
          memref.store %v, %c[%i] : memref<128xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a_in: memref<128xf32>, %c_out: memref<128xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
    }
  }
}
