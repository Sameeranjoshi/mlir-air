// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/diag/pe.csl
//
// 2-D access via 1-D strided view: extract the diagonal of a 16x16 matrix
// stored as a flat 256-element buffer. For a row-major N-by-N matrix, the
// diagonal lives at indices 0, N+1, 2(N+1), ... → stride = N+1, extent = N.

// CHECK-LABEL: fn compute() void
// CHECK: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &M, .extent = 16, .stride = 17 });
// CHECK: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &d, .extent = 16 });
// CHECK: @fmovs(

module {
  csl.wafer @diag {arch = "wse3"} {
    csl.program @pe {
      %M = csl.var @M : memref<256xf32>   // flattened 16x16 row-major
      %d = csl.var @d : memref<16xf32>    // output diagonal (contiguous)
      csl.func @compute {
        %n256 = arith.constant 256 : index
        %n16  = arith.constant  16 : index
        %str  = arith.constant  17 : index  // N+1 for N=16
        %off  = arith.constant   0 : index
        %diag_v = csl.view.strided %n16, %str, %off : !csl.view
        %Md = csl.get_mem_dsd %M, %n256 view %diag_v
                : memref<256xf32>, index, !csl.view -> !csl.dsd
        %dd = csl.get_mem_dsd %d, %n16 : memref<16xf32>, index -> !csl.dsd
        // d := diag(M)
        csl.builtin_call "fmovs"(%dd, %Md) : (!csl.dsd, !csl.dsd) -> ()
        csl.return
      }
      csl.export @M {alias = "M"}
      csl.export @d {alias = "d"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%M_in: memref<256xf32>, %d_out: memref<16xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %M_in to @layout::@M
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@d to %d_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<16xf32>
    }
  }
}
