// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/saxpy/pe.csl
//
// v5 Task 7 — saxpy via DSDs: y = a*x + y
// Alpha is taken from a 1-element memref (host-visible) so the same program
// runs for any scalar without recompiling.

// CHECK-LABEL: fn compute() void
// CHECK: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &x, .extent = 128 });
// CHECK: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &y, .extent = 128 });
// CHECK: @fmacs(

module {
  csl.wafer @saxpy {arch = "wse3"} {
    csl.program @pe {
      %x     = csl.var @x     : memref<128xf32>
      %y     = csl.var @y     : memref<128xf32>
      %alpha = csl.var @alpha : memref<1xf32>
      csl.func @compute {
        %c0 = arith.constant 0   : index
        %n  = arith.constant 128 : index
        %a  = memref.load %alpha[%c0] : memref<1xf32>
        %xd = csl.get_mem_dsd %x, %n : memref<128xf32>, index -> !csl.dsd
        %yd = csl.get_mem_dsd %y, %n : memref<128xf32>, index -> !csl.dsd
        csl.builtin_call "fmacs"(%yd, %yd, %xd, %a)
            : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
        csl.return
      }
      csl.export @x {alias = "x"}
      csl.export @y {alias = "y"}
      csl.export @alpha {alias = "alpha"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%x_in: memref<128xf32>, %y_io: memref<128xf32>,
                   %a_in: memref<1xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %x_in to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.memcpy_h2d %y_io to @layout::@y
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.memcpy_h2d %a_in to @layout::@alpha
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y_io
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
    }
  }
}
