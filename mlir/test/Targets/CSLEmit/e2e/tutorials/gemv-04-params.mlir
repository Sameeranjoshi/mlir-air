// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/gemv04/pe.csl
//
// Tutorial 04 — GEMV with parameterized dimensions.
// Reproduces SDK tutorial gemv-04-params:
//   y[i] = sum_j A[:,j] * x[j] + b    (M=4, N=6, but M/N are 'param' in CSL)
//
// Key CSL concepts:
//   - `param M: i16; param N: i16;` — compile-time parameters
//   - Same compute as tutorial 03, but array sizes are symbolic at write time
//   - Allows recompiling for different matrix dimensions without code changes
//
// MLIR limitation: CSL dialect has no `param` mechanism. Compile-time params
// are represented as arith.constant values (M=4, N=6). A future dialect
// extension could add csl.comptime_param to close this gap.
//
// Corresponding SDK tutorial: gemv-04-params/pe_program.csl

// CHECK-LABEL: fn compute() void
// CHECK: @get_dsd(mem1d_dsd
// CHECK: while (
// CHECK: @fmacs(
// CHECK: @fadds(

module {
  csl.wafer @gemv04 {arch = "wse3"} {
    csl.program @pe {
      // Array sizes determined by M=4, N=6 (would be `param` in CSL source).
      // In a real parameterized program these would be runtime-sized memrefs.
      %A = csl.var @A : memref<24xf32>  // M*N = 4*6
      %x = csl.var @x : memref<6xf32>  // N = 6
      %b = csl.var @b : memref<4xf32>  // M = 4
      %y = csl.var @y : memref<4xf32>  // M = 4

      csl.func @compute {
        // These constants correspond to `param M: i16 = 4; param N: i16 = 6;`
        // in the CSL source.  A param-aware dialect would hoist these to the
        // csl.program block arguments.
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %m  = arith.constant 4 : index  // param M
        %n  = arith.constant 6 : index  // param N

        %y_dsd = csl.get_mem_dsd %y : memref<4xf32> -> !csl.dsd
        %b_dsd = csl.get_mem_dsd %b : memref<4xf32> -> !csl.dsd

        scf.for %j = %c0 to %n step %c1 {
          %col_view = memref.subview %A[%j] [4] [6]
                      : memref<24xf32> to memref<4xf32, strided<[6], offset: ?>>
          %A_col_dsd = csl.get_mem_dsd %col_view
                       : memref<4xf32, strided<[6], offset: ?>> -> !csl.dsd
          %xj = memref.load %x[%j] : memref<6xf32>
          csl.builtin_call "fmacs"(%y_dsd, %y_dsd, %A_col_dsd, %xj)
              : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
        }
        csl.builtin_call "fadds"(%y_dsd, %y_dsd, %b_dsd)
            : (!csl.dsd, !csl.dsd, !csl.dsd) -> ()
        csl.return
      }
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
