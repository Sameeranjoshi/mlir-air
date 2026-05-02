// REQUIRES: cerebras-sdk
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-dataflow-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: cd %t/east_west_2row && bash commands_wse3.sh 2>&1 | FileCheck %s
//
// 2x2 grid with two simultaneous single-hop streams in opposite directions:
//
//   (0,0) ── ch_east ──▶ (1,0)       row 0 flows EAST
//   (0,1) ◀── ch_west ── (1,1)       row 1 flows WEST
//
// This exercises the routing pass allocating two colors (one EAST, one WEST)
// and the host emitter pairing H2D buffers with D2H buffers correctly when
// two independent fabric streams are active simultaneously.
//
// The passthrough heuristic fires (2 H2D, 2 D2H, fabric ops present):
//   pair 0: a_in (buf_l0) ↔ a_out (buf_r0)  — east flow verification
//   pair 1: b_in (buf_r1) ↔ b_out (buf_l1)  — west flow verification
//
// CHECK: SUCCESS!

csl.wafer @east_west_2row {arch = "wse3"} {
  csl.program @pe_l0 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.put @ch_east source(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_l0", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @pe_r0 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.get @ch_east target(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_r0", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @pe_r1 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.put @ch_west source(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_r1", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @pe_l1 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.get @ch_west target(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_l1", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.layout {width = 2 : i64, height = 2 : i64} @layout {
    csl_layout.dataflow @ch_east from(0, 0) to(1, 0)
    csl_layout.dataflow @ch_west from(1, 1) to(0, 1)
    csl_layout.place  @pe_l0 at (0, 0)
    csl_layout.place  @pe_r0 at (1, 0)
    csl_layout.place  @pe_l1 at (0, 1)
    csl_layout.place  @pe_r1 at (1, 1)
    csl_layout.export "buf_l0" from @pe_l0::@buf
    csl_layout.export "buf_r0" from @pe_r0::@buf
    csl_layout.export "buf_r1" from @pe_r1::@buf
    csl_layout.export "buf_l1" from @pe_l1::@buf
    csl_layout.export "compute" from @pe_l0::@compute  {kind = "func"}
  }
  csl.host @main(%a_in: memref<64xf32>, %a_out: memref<64xf32>,
                 %b_in: memref<64xf32>, %b_out: memref<64xf32>)
                 {layout = @layout} {
    csl_host.memcpy_h2d %a_in to @layout::@buf_l0
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_h2d %b_in to @layout::@buf_r1
        {px = 1 : i64, py = 1 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@buf_r0 to %a_out
        {px = 1 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_d2h @layout::@buf_l1 to %b_out
        {px = 0 : i64, py = 1 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
  }
}
