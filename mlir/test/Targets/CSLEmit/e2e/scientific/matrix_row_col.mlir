// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s --check-prefix=ROW < %t/pick_row/pe.csl
// RUN: FileCheck %s --check-prefix=COL < %t/pick_col/pe.csl
// RUN: FileCheck %s --check-prefix=MIX < %t/row_and_col/pe.csl
//
// Full 2-D access story on flat storage via 1-D strided views.
// Row-major M = [H x W] laid flat in a memref<(H*W)xT>:
//   - row i (length W): stride = 1, offset = i*W
//   - col j (length H): stride = W, offset = j
//
// Three wafers:
//   pick_row     — grab a specific row i=3 of a 8x16 matrix.
//   pick_col     — grab a specific col j=5 of the same 8x16 shape.
//   row_and_col  — both views co-exist in one kernel, plus a stride-2
//                   view of the output to demonstrate view reuse +
//                   independence across multiple DSDs in one func.

// ---- Row access: stride=1, offset=i*W  (here i=3, W=16 → offset=48) ----

// ROW-LABEL: fn compute() void
// ROW: @get_dsd(mem1d_dsd, .{ .base_address = &M + 48, .extent = 16 });
// ROW-NOT: .stride

// ---- Column access: stride=W, offset=j  (here j=5, W=16 → stride=16, offset=5) ----

// COL-LABEL: fn compute() void
// COL: @get_dsd(mem1d_dsd, .{ .base_address = &M + 5, .extent = 8, .stride = 16 });

// ---- Mixed: row i=2 + col j=7 from one buffer, plus even output view ----

// MIX-LABEL: fn compute() void
// MIX-DAG: @get_dsd(mem1d_dsd, .{ .base_address = &M + 32, .extent = 16 });
// MIX-DAG: @get_dsd(mem1d_dsd, .{ .base_address = &M + 7, .extent = 8, .stride = 16 });
// MIX-DAG: @get_dsd(mem1d_dsd, .{ .base_address = &out, .extent = 4, .stride = 2 });

module {
  // Row picker: output = row[3] of an 8x16 row-major matrix.
  csl.wafer @pick_row {arch = "wse3"} {
    csl.program @pe {
      %M   = csl.var @M   : memref<128xf32>   // 8 x 16 flat
      %row = csl.var @row : memref<16xf32>
      csl.func @compute {
        %n128 = arith.constant 128 : index
        %n16  = arith.constant  16 : index
        %s1   = arith.constant   1 : index
        %off  = arith.constant  48 : index   // i=3, W=16 → 3*16
        %rv   = csl.view.strided %n16, %s1, %off : !csl.view
        %Mr   = csl.get_mem_dsd %M, %n128 view %rv
                  : memref<128xf32>, index, !csl.view -> !csl.dsd
        %rd   = csl.get_mem_dsd %row, %n16 : memref<16xf32>, index -> !csl.dsd
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

  // Column picker: output = col[5] of the same shape.
  csl.wafer @pick_col {arch = "wse3"} {
    csl.program @pe {
      %M   = csl.var @M   : memref<128xf32>
      %col = csl.var @col : memref<8xf32>
      csl.func @compute {
        %n128 = arith.constant 128 : index
        %n8   = arith.constant   8 : index
        %sW   = arith.constant  16 : index   // stride = W
        %off  = arith.constant   5 : index   // j = 5
        %cv   = csl.view.strided %n8, %sW, %off : !csl.view
        %Mc   = csl.get_mem_dsd %M, %n128 view %cv
                  : memref<128xf32>, index, !csl.view -> !csl.dsd
        %cd   = csl.get_mem_dsd %col, %n8 : memref<8xf32>, index -> !csl.dsd
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

  // Row (i=2) + column (j=7) + a stride-2 output view — three distinct
  // views in one kernel, reading from the same matrix storage.
  csl.wafer @row_and_col {arch = "wse3"} {
    csl.program @pe {
      %M   = csl.var @M   : memref<128xf32>
      %out = csl.var @out : memref<8xf32>
      csl.func @compute {
        %n128 = arith.constant 128 : index
        %n16  = arith.constant  16 : index
        %n8   = arith.constant   8 : index
        %n4   = arith.constant   4 : index
        %s1   = arith.constant   1 : index
        %sW   = arith.constant  16 : index
        %s2   = arith.constant   2 : index
        %row_off = arith.constant 32 : index  // i=2, W=16 → 32
        %col_off = arith.constant  7 : index  // j=7
        %even_off= arith.constant  0 : index

        %row_v = csl.view.strided %n16, %s1, %row_off : !csl.view
        %col_v = csl.view.strided %n8,  %sW, %col_off : !csl.view
        %out_v = csl.view.strided %n4,  %s2, %even_off : !csl.view

        %Mr = csl.get_mem_dsd %M, %n128 view %row_v
                : memref<128xf32>, index, !csl.view -> !csl.dsd
        %Mc = csl.get_mem_dsd %M, %n128 view %col_v
                : memref<128xf32>, index, !csl.view -> !csl.dsd
        %od = csl.get_mem_dsd %out, %n8 view %out_v
                : memref<8xf32>, index, !csl.view -> !csl.dsd

        // out[even] := M_row (truncated to the first 4 elems by extent)
        csl.builtin_call "fmovs"(%od, %Mr) : (!csl.dsd, !csl.dsd) -> ()
        // out[even] += M_col   (again truncated; just to exercise two
        // strided-read DSDs in the same op sequence)
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
