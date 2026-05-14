// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports --csl-dataflow-to-csl \
// RUN:   | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck --check-prefix=LAYOUT  %s < %t/w/layout.csl
// RUN: FileCheck --check-prefix=LEFT    %s < %t/w/left.csl
// RUN: FileCheck --check-prefix=RIGHT   %s < %t/w/right.csl
//
// End-to-end check that the streams_to_csl pipeline + emitter produce
// well-formed CSL for a 2-PE single-stream program. layout.csl gets
// `@get_color`, `@set_color_config`, and `@set_tile_code` — one per
// placement, each routed to its own per-program .csl source file.
// The PE source gets the fabric DSD (fabout for put, fabin for get),
// async fmovs, completion task triplet (id const + task body + comptime
// bind), and sys_mod.unblock_cmd_stream inside the task.

// LAYOUT: const ch_color = @get_color(0);
// LAYOUT: layout {
// LAYOUT:   @set_rectangle(2, 1);
// LAYOUT:   @set_tile_code(0, 0, "left.csl"
// LAYOUT:   @set_tile_code(1, 0, "right.csl"
// LAYOUT:   @set_color_config(0, 0, ch_color, .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST} } });
// LAYOUT:   @set_color_config(1, 0, ch_color, .{ .routes = .{ .rx = .{WEST}, .tx = .{RAMP} } });

// LEFT: const ch_color: color = @get_color(0);
// LEFT: const ch_color_out_q: output_queue = @get_output_queue(2);
// LEFT: fn c() void
// LEFT: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &buf, .extent = 32 });
// LEFT: const {{.*}} = @get_dsd(fabout_dsd, .{ .extent = 32, .fabric_color = ch_color, .output_queue = ch_color_out_q });
// LEFT: @fmovs({{.*}}, .{ .async = true, .activate = ch_put_done_0_id });
// LEFT: const ch_put_done_0_id: local_task_id = @get_local_task_id(8);
// LEFT: task ch_put_done_0() void {
// LEFT:   sys_mod.unblock_cmd_stream();
// LEFT: }
// LEFT: comptime { @bind_local_task(ch_put_done_0, ch_put_done_0_id); }

// RIGHT: const ch_color: color = @get_color(0);
// RIGHT: const ch_color_in_q: input_queue = @get_input_queue(2);
// RIGHT: fn c() void
// RIGHT: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &buf, .extent = 32 });
// RIGHT: const {{.*}} = @get_dsd(fabin_dsd, .{ .extent = 32, .fabric_color = ch_color, .input_queue = ch_color_in_q });
// RIGHT: @fmovs({{.*}}, .{ .async = true, .activate = ch_get_done_0_id });
// RIGHT: const ch_get_done_0_id: local_task_id = @get_local_task_id(8);
// RIGHT: task ch_get_done_0() void {
// RIGHT:   sys_mod.unblock_cmd_stream();
// RIGHT: }
// RIGHT: comptime { @bind_local_task(ch_get_done_0, ch_get_done_0_id); }

csl.wafer @w {arch = "wse3"} {
  csl.program @left {
    %buf = csl.var @buf : memref<32xf32>
    csl.func @c {
      %n = arith.constant 32 : index
      csl.dataflow.put @ch source(%buf) extent(%n : index) : memref<32xf32>
      csl.return
    }
    csl.export @buf {alias = "buf"}
  }
  csl.program @right {
    %buf = csl.var @buf : memref<32xf32>
    csl.func @c {
      %n = arith.constant 32 : index
      csl.dataflow.get @ch target(%buf) extent(%n : index) : memref<32xf32>
      csl.return
    }
    csl.export @buf {alias = "buf"}
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl_layout.dataflow @ch from(0, 0) to(1, 0)
    csl_layout.place @left  at (0, 0)
    csl_layout.place @right at (1, 0)
  }
}
