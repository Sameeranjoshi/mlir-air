// REQUIRES: cerebras-sdk
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-streams-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: cd %t/ping_2pe_extents_small && bash commands_wse3.sh 2>&1 | FileCheck %s
//
// Smaller-extent variant: validates that extent threading through the
// streams pipeline (csl.stream.put/get extent → csl.get_fab_dsd extent →
// .extent = N in fabric DSD) works for sizes other than the milestone's
// 128 elements.
//
// CHECK: SUCCESS!

csl.wafer @ping_2pe_extents_small {arch = "wse3"} {
  csl.program @left_pe {
    %buf = csl.var @buf : memref<32xf32>
    csl.func @compute {
      %n = arith.constant 32 : index
      csl.stream.put @send_ch source(%buf) extent(%n : index) : memref<32xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_left", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @right_pe {
    %buf = csl.var @buf : memref<32xf32>
    csl.func @compute {
      %n = arith.constant 32 : index
      csl.stream.get @send_ch target(%buf) extent(%n : index) : memref<32xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_right", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl_layout.stream @send_ch from(0, 0) to(1, 0)
    csl_layout.place  @left_pe  at (0, 0)
    csl_layout.place  @right_pe at (1, 0)
    csl_layout.export "buf_left"  from @left_pe::@buf
    csl_layout.export "buf_right" from @right_pe::@buf
    csl_layout.export "compute"   from @left_pe::@compute  {kind = "func"}
  }
  csl.host @main(%in: memref<32xf32>, %out: memref<32xf32>) {layout = @layout} {
    csl_host.memcpy_h2d %in to @layout::@buf_left
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<32xf32>
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@buf_right to %out
        {px = 1 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<32xf32>
  }
}
