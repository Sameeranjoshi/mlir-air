// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-auto-vectorize -csl-infer-exports | \
// RUN:   air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/fmacs_e2e/pe.csl
//
// c[i] = a[i]*b[i] + c[i]  ->  @fmacs(dc, dc, da, db)
// c is both an input (accumulator) and an output, so it gets h2d and d2h.

// CHECK-LABEL: fn compute() void
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &a, .extent = 1024 });
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &b, .extent = 1024 });
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &c, .extent = 1024 });
// CHECK: @fmacs(

module {
  csl.wafer @fmacs_e2e {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<1024xf32>
      %b = csl.var @b : memref<1024xf32>
      %c = csl.var @c : memref<1024xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<1024xf32>
          %vb = memref.load %b[%i] : memref<1024xf32>
          %vc = memref.load %c[%i] : memref<1024xf32>
          %m  = arith.mulf %va, %vb : f32
          %s  = arith.addf %m, %vc : f32
          memref.store %s, %c[%i] : memref<1024xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a_in: memref<1024xf32>, %b_in: memref<1024xf32>,
                   %c_io: memref<1024xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1024xf32>
      csl_host.memcpy_h2d %b_in to @layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1024xf32>
      csl_host.memcpy_h2d %c_io to @layout::@c
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1024xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@c to %c_io
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1024xf32>
    }
  }
}
