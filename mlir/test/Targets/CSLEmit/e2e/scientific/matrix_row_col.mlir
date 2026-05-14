// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s --check-prefix=ROW < %t/pick_row/pe.csl
// RUN: FileCheck %s --check-prefix=COL < %t/pick_col/pe.csl
// RUN: FileCheck %s --check-prefix=MIX < %t/row_and_col/pe.csl
//
// 2-D access on flattened storage using memref.subview:
//   - row i (length W):  offset = i*W, size = W, stride = 1
//   - col j (length H):  offset = j,   size = H, stride = W

// ---- Row 3 of an 8x16 row-major matrix: stride = 1, offset = 48 ----

// ROW-LABEL: fn compute() void
// ROW: @get_dsd(mem1d_dsd, .{ .base_address = &M, .extent = 16 });
// ROW: @increment_dsd_offset({{.*}}, 48, f32);

// ---- Column 5: stride = 16, offset = 5 ----

// COL-LABEL: fn compute() void
// COL: @get_dsd(mem1d_dsd, .{ .base_address = &M, .extent = 8, .stride = 16 });
// COL: @increment_dsd_offset({{.*}}, 5, f32);

// ---- Mixed: 4-elem strided slices of row 2 + col 7 reduced into a    ----
// ----         contiguous out[0..4]. Two strided reads + one contiguous  ----
// ----         write — all extent 4 so the DSD engine stays aligned.    ----

// MIX-LABEL: fn compute() void
// MIX-DAG: @increment_dsd_offset({{.*}}, 32, f32);
// MIX-DAG: @increment_dsd_offset({{.*}}, 7, f32);
// MIX-DAG: @get_dsd(mem1d_dsd, .{ .base_address = &out, .extent = 4 });

module {
  // Row picker: output = row[3] of an 8x16 row-major matrix.
  csl.wafer @pick_row {arch = "wse3"} {
    csl.program @pe {
      %M   = csl.var @M   : memref<128xf32>   // 8 x 16 flat
      %row = csl.var @row : memref<16xf32>
      csl.func @compute {
        %rview = memref.subview %M[48] [16] [1]
                 : memref<128xf32> to memref<16xf32, strided<[1], offset: 48>>
        %Mr = csl.get_mem_dsd %rview
              : memref<16xf32, strided<[1], offset: 48>> -> !csl.dsd
        %rd = csl.get_mem_dsd %row : memref<16xf32> -> !csl.dsd
        csl.builtin_call "fmovs"(%rd, %Mr) : (!csl.dsd, !csl.dsd) -> ()
        csl.return
      }
      csl.export @M   {alias = "M"}
      csl.export @row {alias = "row"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%M_in: memref<128xf32>, %r_out: memref<16xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %M_in to @layout::@M
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@row to %r_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<16xf32>
    }
  }

  // Column picker: output = col[5] of an 8x16 row-major matrix.
  csl.wafer @pick_col {arch = "wse3"} {
    csl.program @pe {
      %M   = csl.var @M   : memref<128xf32>
      %col = csl.var @col : memref<8xf32>
      csl.func @compute {
        %cview = memref.subview %M[5] [8] [16]
                 : memref<128xf32> to memref<8xf32, strided<[16], offset: 5>>
        %Mc = csl.get_mem_dsd %cview
              : memref<8xf32, strided<[16], offset: 5>> -> !csl.dsd
        %cd = csl.get_mem_dsd %col : memref<8xf32> -> !csl.dsd
        csl.builtin_call "fmovs"(%cd, %Mc) : (!csl.dsd, !csl.dsd) -> ()
        csl.return
      }
      csl.export @M   {alias = "M"}
      csl.export @col {alias = "col"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%M_in: memref<128xf32>, %c_out: memref<8xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %M_in to @layout::@M
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@col to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<8xf32>
    }
  }

  // Two strided reads (row + column) reduced into a contiguous write —
  // exercises row and column views coexisting on the same buffer with the
  // output DSD being a plain contiguous view of out[0..4]. All extents are
  // 4, so the DSD engine stays aligned.
  csl.wafer @row_and_col {arch = "wse3"} {
    csl.program @pe {
      %M   = csl.var @M   : memref<128xf32>
      %out = csl.var @out : memref<8xf32>
      csl.func @compute {
        %row_v = memref.subview %M[32] [4] [1]
                 : memref<128xf32> to memref<4xf32, strided<[1], offset: 32>>
        %col_v = memref.subview %M[7]  [4] [16]
                 : memref<128xf32> to memref<4xf32, strided<[16], offset: 7>>
        %out_v = memref.subview %out[0] [4] [1]
                 : memref<8xf32>   to memref<4xf32, strided<[1]>>

        %Mr = csl.get_mem_dsd %row_v
              : memref<4xf32, strided<[1], offset: 32>> -> !csl.dsd
        %Mc = csl.get_mem_dsd %col_v
              : memref<4xf32, strided<[16], offset: 7>> -> !csl.dsd
        %od = csl.get_mem_dsd %out_v
              : memref<4xf32, strided<[1]>>             -> !csl.dsd

        csl.builtin_call "fmovs"(%od, %Mr) : (!csl.dsd, !csl.dsd) -> ()
        csl.builtin_call "fadds"(%od, %od, %Mc)
            : (!csl.dsd, !csl.dsd, !csl.dsd) -> ()
        csl.return
      }
      csl.export @M   {alias = "M"}
      csl.export @out {alias = "out"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%M_in: memref<128xf32>, %o_out: memref<8xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %M_in to @layout::@M
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@out to %o_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<8xf32>
    }
  }
}
