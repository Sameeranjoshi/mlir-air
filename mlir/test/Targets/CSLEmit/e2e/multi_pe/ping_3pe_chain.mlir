// REQUIRES: cerebras-sdk
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-streams-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: cd %t/ping_3pe_chain && bash commands_wse3.sh 2>&1 | FileCheck %s
//
// 3-PE chain: producer → pass-through middle → consumer.
//
//   left (0,0) ── ch ──▶ mid (1,0) ── ch ──▶ right (2,0)
//
// The middle PE does NOT issue any stream put/get — it only contributes
// router pass-through (rx=WEST, tx=EAST) for color `ch`. The routing pass
// emits a set_color_config triplet (one per hop), and the middle PE
// program is empty (no fabric DSDs, just unblock_cmd_stream).
//
// CHECK: SUCCESS!

csl.wafer @ping_3pe_chain {arch = "wse3"} {
  csl.program @left {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.stream.put @ch source(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_left", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @mid {
    csl.func @compute { csl.return }
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @right {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.stream.get @ch target(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_right", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.layout {width = 3 : i64, height = 1 : i64} @layout {
    csl_layout.stream @ch from(0, 0) to(2, 0)
    csl_layout.place  @left  at (0, 0)
    csl_layout.place  @mid   at (1, 0)
    csl_layout.place  @right at (2, 0)
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
        {px = 2 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
  }
}
