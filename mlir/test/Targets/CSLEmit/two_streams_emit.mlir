// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-streams-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck --check-prefix=LAYOUT %s < %t/two_streams/layout.csl
// RUN: FileCheck --check-prefix=L0     %s < %t/two_streams/l0.csl
// RUN: FileCheck --check-prefix=R0     %s < %t/two_streams/r0.csl
// RUN: FileCheck --check-prefix=L1     %s < %t/two_streams/l1.csl
// RUN: FileCheck --check-prefix=R1     %s < %t/two_streams/r1.csl
//
// Compiler-level coverage for the two-stream case: 2x2 grid with two
// independent left→right fabric streams, one per row. Each PE has exactly
// one csl.stream op, so no multi-async-task-per-PE concerns.
//
// This test exercises:
//   - color id allocator emits two distinct ids
//   - routing pass emits two independent set_color_config pairs
//   - per-PE source files re-declare both colors
//   - per-PE queue allocation gives each PE one input or output queue
//
// NOTE: the simulator e2e form of this test is currently blocked by an
// SDK constraint: with `cslc --memcpy --channels 1`, color routing
// configured on PE rows other than the one carrying the cmd-stream
// path appears to disrupt memcpy unblock signaling, causing a stall.
// Until the constraint is understood (likely raising --channels or
// shifting user color ids), we ship this as an emitter-only check.

// LAYOUT: const ch_b_color = @get_color(0);
// LAYOUT: const ch_a_color = @get_color(1);
// LAYOUT: layout {
// LAYOUT:   @set_rectangle(2, 2);
// LAYOUT:   @set_tile_code(0, 0, "l0.csl"
// LAYOUT:   @set_tile_code(1, 0, "r0.csl"
// LAYOUT:   @set_tile_code(0, 1, "l1.csl"
// LAYOUT:   @set_tile_code(1, 1, "r1.csl"
// LAYOUT:   @set_color_config(0, 0, ch_a_color, .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST} } });
// LAYOUT:   @set_color_config(1, 0, ch_a_color, .{ .routes = .{ .rx = .{WEST}, .tx = .{RAMP} } });
// LAYOUT:   @set_color_config(0, 1, ch_b_color, .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST} } });
// LAYOUT:   @set_color_config(1, 1, ch_b_color, .{ .routes = .{ .rx = .{WEST}, .tx = .{RAMP} } });

// L0: const ch_b_color: color = @get_color(0);
// L0: const ch_a_color: color = @get_color(1);
// L0: const ch_a_color_out_q: output_queue = @get_output_queue(2);
// L0: fabout_dsd, .{ .extent = 64, .fabric_color = ch_a_color, .output_queue = ch_a_color_out_q

// R0: const ch_b_color: color = @get_color(0);
// R0: const ch_a_color: color = @get_color(1);
// R0: const ch_a_color_in_q: input_queue = @get_input_queue(2);
// R0: fabin_dsd, .{ .extent = 64, .fabric_color = ch_a_color, .input_queue = ch_a_color_in_q

// L1: const ch_b_color: color = @get_color(0);
// L1: const ch_a_color: color = @get_color(1);
// L1: const ch_b_color_out_q: output_queue = @get_output_queue(2);
// L1: fabout_dsd, .{ .extent = 64, .fabric_color = ch_b_color, .output_queue = ch_b_color_out_q

// R1: const ch_b_color: color = @get_color(0);
// R1: const ch_a_color: color = @get_color(1);
// R1: const ch_b_color_in_q: input_queue = @get_input_queue(2);
// R1: fabin_dsd, .{ .extent = 64, .fabric_color = ch_b_color, .input_queue = ch_b_color_in_q

csl.wafer @two_streams {arch = "wse3"} {
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
  }
}
