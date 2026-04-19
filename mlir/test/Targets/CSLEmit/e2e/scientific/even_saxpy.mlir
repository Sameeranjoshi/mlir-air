// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/even_saxpy/pe.csl
//
// Strided DSD via upstream memref.subview: y[2i] = a*x[2i] + y[2i] for
// i in [0, 64). One subview defines the stride-2 layout; it's used to build
// the two DSDs (x and y) directly — the emitter reads stride/extent from
// each result memref's type.

// CHECK-LABEL: fn compute() void
// CHECK: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &x, .extent = 64, .stride = 2 });
// CHECK: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &y, .extent = 64, .stride = 2 });
// CHECK: @fmacs(

module {
  csl.wafer @even_saxpy {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      csl.func @compute {
        %a  = arith.constant 2.0 : f32
        %xv = memref.subview %x[0] [64] [2]
              : memref<128xf32> to memref<64xf32, strided<[2]>>
        %yv = memref.subview %y[0] [64] [2]
              : memref<128xf32> to memref<64xf32, strided<[2]>>
        %xd = csl.get_mem_dsd %xv : memref<64xf32, strided<[2]>> -> !csl.dsd
        %yd = csl.get_mem_dsd %yv : memref<64xf32, strided<[2]>> -> !csl.dsd
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
