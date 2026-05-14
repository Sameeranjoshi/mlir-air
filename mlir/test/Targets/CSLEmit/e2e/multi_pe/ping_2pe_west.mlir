// REQUIRES: cerebras-sdk
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-dataflow-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: cd %t/ping_2pe_west && bash commands_wse3.sh 2>&1 | FileCheck %s
//
// 2-PE westward variant of ping_2pe:
//
//   right_pe ◀── send_ch ── left_pe
//   (0,0)                   (1,0)
//
// Producer at (1,0), consumer at (0,0); routing pass infers tx=WEST/rx=EAST.
//
// CHECK: SUCCESS!

csl.wafer @ping_2pe_west {arch = "wse3"} {
  csl.program @right_pe {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @compute {
      %n = arith.constant 128 : index
      csl.dataflow.put @send_ch source(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_right", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @left_pe {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @compute {
      %n = arith.constant 128 : index
      csl.dataflow.get @send_ch target(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_left", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl_layout.dataflow @send_ch from(1, 0) to(0, 0)
    csl_layout.place  @right_pe at (1, 0)
    csl_layout.place  @left_pe  at (0, 0)
    csl_layout.export "buf_right" from @right_pe::@buf
    csl_layout.export "buf_left"  from @left_pe::@buf
    csl_layout.export "compute"   from @right_pe::@compute  {kind = "func"}
  }
  csl.host @main(%in: memref<128xf32>, %out: memref<128xf32>) {layout = @layout} {
    csl_host.memcpy_h2d %in to @layout::@buf_right
        {px = 1 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<128xf32>
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@buf_left to %out
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<128xf32>
  }
}
