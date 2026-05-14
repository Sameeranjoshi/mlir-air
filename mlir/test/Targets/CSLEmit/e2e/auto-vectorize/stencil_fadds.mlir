// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-auto-vectorize -csl-infer-exports | \
// RUN:   air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/stencil_fadds_e2e/pe.csl
//
// 2-point stencil: c[i] = a[i-1] + a[i]  for i in [1, N-1)
// Pass emits two subviews of `a` (offsets 0 and 1) via @increment_dsd_offset,
// one subview of `c` (offset 1), then calls @fadds.

// CHECK-LABEL: fn compute() void
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &a, .extent = 128 });
// CHECK: @increment_dsd_offset(
// CHECK: @fadds(

module {
  csl.wafer @stencil_fadds_e2e {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<128xf32>
      %c = csl.var @c : memref<128xf32>
      csl.func @compute {
        %c1  = arith.constant 1 : index
        %Nm1 = arith.constant 127 : index
        scf.for %i = %c1 to %Nm1 step %c1 {
          %im1 = arith.subi %i, %c1 : index
          %vl  = memref.load %a[%im1] : memref<128xf32>
          %vc  = memref.load %a[%i]   : memref<128xf32>
          %s   = arith.addf %vl, %vc  : f32
          memref.store %s, %c[%i]     : memref<128xf32>
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
