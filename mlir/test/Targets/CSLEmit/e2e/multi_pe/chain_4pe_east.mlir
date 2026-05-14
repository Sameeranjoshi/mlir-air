// REQUIRES: cerebras-sdk
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-dataflow-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: cd %t/chain_4pe_east && bash commands_wse3.sh 2>&1 | FileCheck %s
//
// 4-PE chain: producer → pass-through → pass-through → consumer (3 hops EAST).
//
//   left(0,0) ── ch ──▶ mid1(1,0) ──▶ mid2(2,0) ──▶ right(3,0)
//
// A single csl_layout.dataflow spanning 3 hops exercises the multi-hop routing
// pass. The two intermediate PEs are empty pass-throughs (just csl.return).
// The passthrough heuristic fires (1 H2D, 1 D2H, fabric ops present) and
// verifies that the output buffer equals the input buffer element-wise.
//
// CHECK: SUCCESS!

csl.wafer @chain_4pe_east {arch = "wse3"} {
  csl.program @left {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.put @ch source(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_left", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @mid1 {
    csl.func @compute { csl.return }
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @mid2 {
    csl.func @compute { csl.return }
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @right {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.get @ch target(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_right", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.layout {width = 4 : i64, height = 1 : i64} @layout {
    csl_layout.dataflow @ch from(0, 0) to(3, 0)
    csl_layout.place  @left  at (0, 0)
    csl_layout.place  @mid1  at (1, 0)
    csl_layout.place  @mid2  at (2, 0)
    csl_layout.place  @right at (3, 0)
    csl_layout.export "buf_left"  from @left::@buf
    csl_layout.export "buf_right" from @right::@buf
    csl_layout.export "compute"   from @left::@compute  {kind = "func"}
  }
  csl.host @main(%in: memref<64xf32>, %out: memref<64xf32>) {layout = @layout} {
    csl_host.memcpy_h2d %in to @layout::@buf_left
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@buf_right to %out
        {px = 3 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
  }
}
