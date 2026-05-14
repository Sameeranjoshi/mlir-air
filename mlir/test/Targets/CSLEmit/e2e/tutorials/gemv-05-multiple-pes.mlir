// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s --check-prefix=PE     < %t/gemv05/pe.csl
// RUN: FileCheck %s --check-prefix=LAYOUT < %t/gemv05/layout.csl
//
// Tutorial 05 — GEMV on a 1×2 PE grid (column partitioning).
// Reproduces SDK tutorial gemv-05-multiple-pes:
//   PE i computes:  y += A_shard[:,j] * x_shard[j]   (N_per_PE = 3 cols each)
//
// Key CSL concepts:
//   - Multiple PEs running the same program (uniform SPMD)
//   - Column partitioning: each PE gets N_per_PE=3 columns of A and x
//   - csl_layout.place @pe over [0:2, 0] places the program on 2 PEs
//   - Host broadcasts A/x shards to corresponding PEs via the 2-wide h2d op
//   - y results collected per-PE (each PE produces M=4 output elements)
//
// Note: In the SDK tutorial the grid size and N_per_PE are `param`. Here we
// use a 2-PE concrete example (N=6 split evenly: N_per_PE=3).
//
// Corresponding SDK tutorial: gemv-05-multiple-pes/pe_program.csl

// PE-LABEL: fn compute() void
// PE: @get_dsd(mem1d_dsd, .{ .base_address = &y, .extent = 4 });
// PE: @get_dsd(mem1d_dsd, .{ .base_address = &b, .extent = 4 });
// PE: @fmacs(
// PE: @fadds(

// LAYOUT: @set_rectangle(2, 1);
// LAYOUT: while (

module {
  csl.wafer @gemv05 {arch = "wse3"} {
    // One program placed on every PE in the 1×2 grid.
    // Each PE receives its shard: A[M*N_per_PE], x[N_per_PE], b[M], y[M].
    csl.program @pe {
      %A = csl.var @A : memref<12xf32>  // M*N_per_PE = 4*3, column-major shard
      %x = csl.var @x : memref<3xf32>  // N_per_PE = 3 elements of x
      %b = csl.var @b : memref<4xf32>  // bias vector (full M)
      %y = csl.var @y : memref<4xf32>  // output vector (full M), zeroed by host

      csl.func @compute {
        %c0  = arith.constant 0 : index
        %c1  = arith.constant 1 : index
        %m   = arith.constant 4 : index  // M = 4 output rows
        %npp = arith.constant 3 : index  // N_per_PE = 3 columns per PE

        // Contiguous DSDs for y and b
        %y_dsd = csl.get_mem_dsd %y : memref<4xf32> -> !csl.dsd
        %b_dsd = csl.get_mem_dsd %b : memref<4xf32> -> !csl.dsd

        // Accumulate each column of A into y using SIMD fmacs
        // A stored column-major in this shard: A[:,j] = A[j*M .. (j+1)*M-1]
        scf.for %j = %c0 to %npp step %c1 {
          %col_view = memref.subview %A[%j] [4] [3]
                      : memref<12xf32> to memref<4xf32, strided<[3], offset: ?>>
          %A_col_dsd = csl.get_mem_dsd %col_view
                       : memref<4xf32, strided<[3], offset: ?>> -> !csl.dsd
          %xj = memref.load %x[%j] : memref<3xf32>
          csl.builtin_call "fmacs"(%y_dsd, %y_dsd, %A_col_dsd, %xj)
              : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
        }
        // y += b  (only the last PE adds bias; but for simplicity all PEs do
        // it here — a production design would use pe_id param to gate this)
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
    // 1×2 grid: two PEs side by side, both running @pe
    csl.layout {width = 2 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe over [0:2, 0]
    }
    // Host provides 2-wide shards: memref<2xN_per_PE> and memref<2xM>
    csl.host @main(
        %A_in: memref<2x12xf32>,  // [PE, M*N_per_PE]
        %x_in: memref<2x3xf32>,   // [PE, N_per_PE]
        %b_in: memref<2x4xf32>,   // [PE, M]
        %y_io: memref<2x4xf32>)   // [PE, M]
        {layout = @layout} {
      csl_host.memcpy_h2d %A_in to @layout::@A
          {px = 0 : i64, py = 0 : i64, width = 2 : i64, height = 1 : i64}
          : memref<2x12xf32>
      csl_host.memcpy_h2d %x_in to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 2 : i64, height = 1 : i64}
          : memref<2x3xf32>
      csl_host.memcpy_h2d %b_in to @layout::@b
          {px = 0 : i64, py = 0 : i64, width = 2 : i64, height = 1 : i64}
          : memref<2x4xf32>
      csl_host.memcpy_h2d %y_io to @layout::@y
          {px = 0 : i64, py = 0 : i64, width = 2 : i64, height = 1 : i64}
          : memref<2x4xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y_io
          {px = 0 : i64, py = 0 : i64, width = 2 : i64, height = 1 : i64}
          : memref<2x4xf32>
    }
  }
}
