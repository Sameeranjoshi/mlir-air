// REQUIRES: cerebras-sdk
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-dataflow-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: cd %t/north_south_2col && bash commands_wse3.sh 2>&1 | FileCheck %s
//
// 2x2 grid with two simultaneous single-hop streams in perpendicular directions:
//
//   (0,0) ── ch_south ──▼──▶ (0,1)       col 0 flows SOUTH
//   (1,0) ◀── ch_north ────── (1,1)       col 1 flows NORTH
//
// This exercises the routing pass allocating two colors (one SOUTH, one NORTH)
// and two independent vertical stream paths running concurrently.
//
// The passthrough heuristic fires (2 H2D, 2 D2H, fabric ops present):
//   pair 0: s_in (buf_top0) ↔ s_out (buf_bot0)  — south flow verification
//   pair 1: n_in (buf_bot1) ↔ n_out (buf_top1)  — north flow verification
//
// CHECK: SUCCESS!

csl.wafer @north_south_2col {arch = "wse3"} {
  csl.program @pe_top0 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.put @ch_south source(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_top0", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @pe_bot0 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.get @ch_south target(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_bot0", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @pe_bot1 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.put @ch_north source(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_bot1", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @pe_top1 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.get @ch_north target(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_top1", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.layout {width = 2 : i64, height = 2 : i64} @layout {
    csl_layout.dataflow @ch_south from(0, 0) to(0, 1)
    csl_layout.dataflow @ch_north from(1, 1) to(1, 0)
    csl_layout.place  @pe_top0 at (0, 0)
    csl_layout.place  @pe_bot0 at (0, 1)
    csl_layout.place  @pe_top1 at (1, 0)
    csl_layout.place  @pe_bot1 at (1, 1)
    csl_layout.export "buf_top0" from @pe_top0::@buf
    csl_layout.export "buf_bot0" from @pe_bot0::@buf
    csl_layout.export "buf_bot1" from @pe_bot1::@buf
    csl_layout.export "buf_top1" from @pe_top1::@buf
    csl_layout.export "compute"  from @pe_top0::@compute  {kind = "func"}
  }
  csl.host @main(%s_in: memref<64xf32>, %s_out: memref<64xf32>,
                 %n_in: memref<64xf32>, %n_out: memref<64xf32>)
                 {layout = @layout} {
    csl_host.memcpy_h2d %s_in to @layout::@buf_top0
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_h2d %n_in to @layout::@buf_bot1
        {px = 1 : i64, py = 1 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@buf_bot0 to %s_out
        {px = 0 : i64, py = 1 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_d2h @layout::@buf_top1 to %n_out
        {px = 1 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
  }
}
