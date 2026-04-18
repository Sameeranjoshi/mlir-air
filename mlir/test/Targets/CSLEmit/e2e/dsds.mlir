// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/saxpy_dsd/pe.csl
//
// v5 Tasks 3 + 4 — emit csl.get_mem_dsd and csl.builtin_call.
//
// Saxpy via DSDs: y = a*x + y   (expressed as `@fmacs(y, y, A, alpha);`
// where alpha is a scalar broadcast operand).

// CHECK-LABEL: fn compute() void
// CHECK: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &A, .extent = 128 });
// CHECK: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &y, .extent = 128 });
// CHECK: @fmacs(
// CHECK: @fadds(

module {
  csl.wafer @saxpy_dsd {arch = "wse3"} {
    csl.program @pe {
      %A = csl.var @A : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      csl.func @compute {
        %n    = arith.constant 128 : index
        %scal = arith.constant 2.0 : f32
        %Ad   = csl.get_mem_dsd %A, %n : memref<128xf32>, index -> !csl.dsd
        %yd   = csl.get_mem_dsd %y, %n : memref<128xf32>, index -> !csl.dsd
        // y = a*A + y
        csl.builtin_call "fmacs"(%yd, %yd, %Ad, %scal)
            : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
        // y = y + A
        csl.builtin_call "fadds"(%yd, %yd, %Ad)
            : (!csl.dsd, !csl.dsd, !csl.dsd) -> ()
        csl.return
      }
      csl.export @A {alias = "A"}
      csl.export @y {alias = "y"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%A_in: memref<128xf32>, %y_io: memref<128xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %A_in to @layout::@A
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
