// REQUIRES: cerebras-sdk
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-streams-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: cd %t/ping_2pe_two_streams && bash commands_wse3.sh 2>&1 | FileCheck %s
//
// Two independent fabric streams in one wafer. Layout is 2 rows x 2 cols;
// each row is its own producer→consumer pair using a distinct color:
//
//   row0:  l0 (0,0) ── ch_a ──▶ r0 (1,0)
//   row1:  l1 (0,1) ── ch_b ──▶ r1 (1,1)
//
// Each PE has exactly one stream op (no multi-async-per-PE), so the
// existing single-task-per-program unblock pattern continues to work.
// This test exercises:
//   - color-id allocator emits two distinct ids (ch_b=0, ch_a=1)
//   - routing pass emits two independent set_color_config pairs
//   - host emitter pairs N H2D buffers to N D2H buffers in body order
//   - WSE-3 @initialize_queue binding wires the non-zero color (ch_a=1)
//     to its queue; without this binding the consumer's fabin DSD never
//     sees the wavelets and the kernel stalls.
//
// CHECK: SUCCESS!

csl.wafer @ping_2pe_two_streams {arch = "wse3"} {
  csl.program @l0 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.stream.put @ch_a source(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_l0", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @r0 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.stream.get @ch_a target(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_r0", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @l1 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.stream.put @ch_b source(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_l1", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @r1 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.stream.get @ch_b target(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_r1", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.layout {width = 2 : i64, height = 2 : i64} @layout {
    csl_layout.stream @ch_a from(0, 0) to(1, 0)
    csl_layout.stream @ch_b from(0, 1) to(1, 1)
    csl_layout.place  @l0 at (0, 0)
    csl_layout.place  @r0 at (1, 0)
    csl_layout.place  @l1 at (0, 1)
    csl_layout.place  @r1 at (1, 1)
    csl_layout.export "buf_l0" from @l0::@buf
    csl_layout.export "buf_r0" from @r0::@buf
    csl_layout.export "buf_l1" from @l1::@buf
    csl_layout.export "buf_r1" from @r1::@buf
    csl_layout.export "compute" from @l0::@compute  {kind = "func"}
  }
  csl.host @main(%a_in: memref<64xf32>, %a_out: memref<64xf32>,
                 %b_in: memref<64xf32>, %b_out: memref<64xf32>)
                 {layout = @layout} {
    csl_host.memcpy_h2d %a_in to @layout::@buf_l0
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_h2d %b_in to @layout::@buf_l1
        {px = 0 : i64, py = 1 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@buf_r0 to %a_out
        {px = 1 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_d2h @layout::@buf_r1 to %b_out
        {px = 1 : i64, py = 1 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
  }
}
