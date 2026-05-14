// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/bcast/pe.csl
//
// Stride-0 broadcast via memref.subview — the SDK spec's canonical
// "replicate a single element" pattern. y[i] += alpha * x[0] for every i,
// expressed as a stride-0 DSD over x.

// CHECK-LABEL: fn compute() void
// CHECK: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &y, .extent = 128 });
// CHECK: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &x, .extent = 128, .stride = 0 });
// CHECK: @fmacs(

module {
  csl.wafer @bcast {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      csl.func @compute {
        %a  = arith.constant 2.0 : f32
        %yd = csl.get_mem_dsd %y : memref<128xf32> -> !csl.dsd
        // Stride-0, 128 copies of x[0].
        %xv = memref.subview %x[0] [128] [0]
              : memref<128xf32> to memref<128xf32, strided<[0]>>
        %xd = csl.get_mem_dsd %xv : memref<128xf32, strided<[0]>> -> !csl.dsd
        // y := y + a * x[0]  (x[0] broadcast across all 128 lanes)
        csl.builtin_call "fmacs"(%yd, %yd, %xd, %a)
            : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
        csl.return
      }
      csl.export @x {alias = "x"}
      csl.export @y {alias = "y"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%x_in: memref<128xf32>, %y_io: memref<128xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %x_in to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.memcpy_h2d %y_io to @layout::@y
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y_io
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
    }
  }
}
