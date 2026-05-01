// REQUIRES: cerebras-sdk
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-streams-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: cd %t/ping_2pe_north && bash commands_wse3.sh 2>&1 | FileCheck %s
//
// 2-PE vertical-north variant of ping_2pe:
//
//   top_pe    (0,0)
//      ▲
//      │ send_ch
//   bot_pe    (0,1)
//
// Producer at (0,1), consumer at (0,0); routing pass infers tx=NORTH/rx=SOUTH.
//
// CHECK: SUCCESS!

csl.wafer @ping_2pe_north {arch = "wse3"} {
  csl.program @bot_pe {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @compute {
      %n = arith.constant 128 : index
      csl.stream.put @send_ch source(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_bot", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @top_pe {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @compute {
      %n = arith.constant 128 : index
      csl.stream.get @send_ch target(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_top", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.layout {width = 1 : i64, height = 2 : i64} @layout {
    csl_layout.stream @send_ch from(0, 1) to(0, 0)
    csl_layout.place  @bot_pe at (0, 1)
    csl_layout.place  @top_pe at (0, 0)
    csl_layout.export "buf_bot" from @bot_pe::@buf
    csl_layout.export "buf_top" from @top_pe::@buf
    csl_layout.export "compute" from @bot_pe::@compute  {kind = "func"}
  }
  csl.host @main(%in: memref<128xf32>, %out: memref<128xf32>) {layout = @layout} {
    csl_host.memcpy_h2d %in to @layout::@buf_bot
        {px = 0 : i64, py = 1 : i64, width = 1 : i64, height = 1 : i64}
        : memref<128xf32>
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@buf_top to %out
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<128xf32>
  }
}
