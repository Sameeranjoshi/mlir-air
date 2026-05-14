// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-auto-vectorize -csl-infer-exports | \
// RUN:   air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/fnegs_e2e/pe.csl
//
// c[i] = -a[i]  ->  @fnegs(dc, da)

// CHECK-LABEL: fn compute() void
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &a, .extent = 256 });
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &c, .extent = 256 });
// CHECK: @fnegs(

module {
  csl.wafer @fnegs_e2e {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<256xf32>
      %c = csl.var @c : memref<256xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 256 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %v  = memref.load %a[%i] : memref<256xf32>
          %n0 = arith.negf %v : f32
          memref.store %n0, %c[%i] : memref<256xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a_in: memref<256xf32>, %c_out: memref<256xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
    }
  }
}
