// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/gemv02/pe.csl
//
// Tutorial 02 — GEMV with memory DSDs.
// Reproduces SDK tutorial gemv-02-memory-dsds:
//   y[i] = sum_j A[:,j] * x[j] + b    (M=4, N=6, column-major A access)
//
// Key CSL concepts:
//   - mem1d_dsd for contiguous y/b vectors
//   - Strided DSD for column-major A access (stride = N = 6)
//   - @fmacs: y_dsd += A_col_dsd * x[j]  (SIMD multiply-accumulate)
//   - @fadds: y_dsd += b_dsd              (SIMD add)
//   - @increment_dsd_offset to advance to next A column each iteration
//
// Note: The SDK tutorial uses module-level DSD vars initialized at load time
// (`var y_dsd = @get_dsd(...)`). MLIR creates DSDs at function scope, which
// produces equivalent CSL output.
//
// Corresponding SDK tutorial: gemv-02-memory-dsds/pe_program.csl

// CHECK-LABEL: fn compute() void
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &y, .extent = 4 });
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &b, .extent = 4 });
// CHECK: while (
// CHECK: @get_dsd(mem1d_dsd, .{ .base_address = &A, .extent = 4, .stride = 6 });
// CHECK: @increment_dsd_offset(
// CHECK: @fmacs(
// CHECK: @fadds(

module {
  csl.wafer @gemv02 {arch = "wse3"} {
    csl.program @pe {
      %A = csl.var @A : memref<24xf32>  // [M*N] = [4*6], row-major storage
      %x = csl.var @x : memref<6xf32>
      %b = csl.var @b : memref<4xf32>
      %y = csl.var @y : memref<4xf32>

      csl.func @compute {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %m  = arith.constant 4 : index   // M = 4 rows
        %n  = arith.constant 6 : index   // N = 6 cols
        %mn = arith.constant 24 : index  // M*N

        // Initialize A[idx] = 1.0 (SDK uses idx as float; arith.sitofp is not
        // yet supported by the CSL emitter, so a constant is used instead)
        %one  = arith.constant 1.0 : f32
        %two  = arith.constant 2.0 : f32
        %zero = arith.constant 0.0 : f32
        scf.for %idx = %c0 to %mn step %c1 {
          memref.store %one, %A[%idx] : memref<24xf32>
        }
        // Initialize x[j] = 1.0, b[i] = 2.0, y[i] = 0.0
        scf.for %j = %c0 to %n step %c1 {
          memref.store %one, %x[%j] : memref<6xf32>
        }
        scf.for %i = %c0 to %m step %c1 {
          memref.store %two,  %b[%i] : memref<4xf32>
          memref.store %zero, %y[%i] : memref<4xf32>
        }

        // DSD-based GEMV: accumulate column-wise contributions
        // y_dsd = DSD for y[0..M-1] (contiguous)
        %y_dsd = csl.get_mem_dsd %y : memref<4xf32> -> !csl.dsd
        // b_dsd = DSD for b[0..M-1] (contiguous)
        %b_dsd = csl.get_mem_dsd %b : memref<4xf32> -> !csl.dsd

        // For each column j of A: y += A[:,j] * x[j]
        // A[:,j] has stride=N=6 starting at element j
        scf.for %j = %c0 to %n step %c1 {
          // A column j: elements A[j], A[j+6], A[j+12], A[j+18] (stride 6)
          %col_view = memref.subview %A[%j] [4] [6]
                      : memref<24xf32> to memref<4xf32, strided<[6], offset: ?>>
          %A_col_dsd = csl.get_mem_dsd %col_view
                       : memref<4xf32, strided<[6], offset: ?>> -> !csl.dsd
          %xj = memref.load %x[%j] : memref<6xf32>
          // y_dsd += A_col_dsd * x[j]
          csl.builtin_call "fmacs"(%y_dsd, %y_dsd, %A_col_dsd, %xj)
              : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
        }
        // y_dsd += b_dsd
        csl.builtin_call "fadds"(%y_dsd, %y_dsd, %b_dsd)
            : (!csl.dsd, !csl.dsd, !csl.dsd) -> ()
        csl.return
      }
      csl.export @y {alias = "y"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%y_out: memref<4xf32>) {layout = @layout} {
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<4xf32>
    }
  }
}
