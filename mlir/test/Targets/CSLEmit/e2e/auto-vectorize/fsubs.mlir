// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-auto-vectorize -csl-infer-exports | \
// RUN:   air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/fsubs_e2e/pe.csl
//
// c[i] = a[i] - b[i]  ->  @fsubs(dc, da, db)

// CHECK-LABEL: fn compute() void
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &a, .extent = 512 });
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &b, .extent = 512 });
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &c, .extent = 512 });
// CHECK: @fsubs(

module {
  csl.wafer @fsubs_e2e {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<512xf32>
      %b = csl.var @b : memref<512xf32>
      %c = csl.var @c : memref<512xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 512 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<512xf32>
          %vb = memref.load %b[%i] : memref<512xf32>
          %vc = arith.subf %va, %vb : f32
          memref.store %vc, %c[%i] : memref<512xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a_in: memref<512xf32>, %b_in: memref<512xf32>,
                   %c_out: memref<512xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<512xf32>
      csl_host.memcpy_h2d %b_in to @layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<512xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<512xf32>
    }
  }
}
