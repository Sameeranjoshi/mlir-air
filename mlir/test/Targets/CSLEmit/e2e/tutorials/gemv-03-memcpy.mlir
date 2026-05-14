// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/gemv03/pe.csl
//
// Tutorial 03 — GEMV with host-side memcpy for all arrays.
// Reproduces SDK tutorial gemv-03-memcpy:
//   y[i] = sum_j A[:,j] * x[j] + b    (M=4, N=6)
//
// Key CSL concepts:
//   - All four arrays (A, x, b, y) provided by the host via memcpy_h2d
//   - All array pointers exported as writable symbols so the host can write
//   - No PE-side initialization — the host sets up all data
//   - DSD-based column-major GEMV (same as tutorial 02)
//
// Corresponding SDK tutorial: gemv-03-memcpy/pe_program.csl

// CHECK-LABEL: fn compute() void
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &y, .extent = 4 });
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &b, .extent = 4 });
// CHECK: while (
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &A, .extent = 4, .stride = 6 });
// CHECK: @increment_dsd_offset(
// CHECK: @fmacs(
// CHECK: @fadds(

module {
  csl.wafer @gemv03 {arch = "wse3"} {
    csl.program @pe {
      %A = csl.var @A : memref<24xf32>  // [M*N], row-major; host-provided
      %x = csl.var @x : memref<6xf32>  // host-provided
      %b = csl.var @b : memref<4xf32>  // host-provided
      %y = csl.var @y : memref<4xf32>  // zeroed by host before launch

      csl.func @compute {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %m  = arith.constant 4 : index  // M = 4 rows
        %n  = arith.constant 6 : index  // N = 6 cols

        // y_dsd and b_dsd span all M rows (contiguous)
        %y_dsd = csl.get_mem_dsd %y : memref<4xf32> -> !csl.dsd
        %b_dsd = csl.get_mem_dsd %b : memref<4xf32> -> !csl.dsd

        // For each column j: y += A[:,j] * x[j]
        scf.for %j = %c0 to %n step %c1 {
          %col_view = memref.subview %A[%j] [4] [6]
                      : memref<24xf32> to memref<4xf32, strided<[6], offset: ?>>
          %A_col_dsd = csl.get_mem_dsd %col_view
                       : memref<4xf32, strided<[6], offset: ?>> -> !csl.dsd
          %xj = memref.load %x[%j] : memref<6xf32>
          csl.builtin_call "fmacs"(%y_dsd, %y_dsd, %A_col_dsd, %xj)
              : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
        }
        // y += b
        csl.builtin_call "fadds"(%y_dsd, %y_dsd, %b_dsd)
            : (!csl.dsd, !csl.dsd, !csl.dsd) -> ()
        csl.return
      }
      // Export all four arrays writable so the host can push data via memcpy_h2d
      csl.export @A {alias = "A"}
      csl.export @x {alias = "x"}
      csl.export @b {alias = "b"}
      csl.export @y {alias = "y"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%A_in: memref<24xf32>, %x_in: memref<6xf32>,
                   %b_in: memref<4xf32>,  %y_io: memref<4xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %A_in to @layout::@A
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<24xf32>
      csl_host.memcpy_h2d %x_in to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<6xf32>
      csl_host.memcpy_h2d %b_in to @layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<4xf32>
      csl_host.memcpy_h2d %y_io to @layout::@y
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<4xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y_io
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<4xf32>
    }
  }
}
