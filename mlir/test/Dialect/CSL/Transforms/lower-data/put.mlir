// RUN: air-opt --csl-lower-dataflow-data %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @left {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: %[[SRC:.*]] = csl.get_mem_dsd %{{.*}} : memref<128xf32> -> !csl.dsd
      // CHECK: %[[OUT:.*]] = csl.get_fab_dsd fabout @ch_color extent(%{{.*}} : index) : !csl.dsd
      // CHECK: csl.builtin_call "fmovs"(%[[OUT]], %[[SRC]]) {activate = @ch_put_done_0, async} : (!csl.dsd, !csl.dsd) -> ()
      csl.dataflow.put @ch source(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
    // CHECK: csl.task @ch_put_done_0 attributes {id = 8 : i32, trigger_kind = "local_task_id"}
    // CHECK: csl.builtin_call "unblock_cmd_stream"() : () -> ()
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl.color @ch_color {id = 0 : i32}
    csl_layout.dataflow @ch from(0, 0) to(1, 0) {color = @ch_color}
    csl_layout.set_color_config @ch_color at(0, 0) rx(RAMP) tx(EAST)
    csl_layout.set_color_config @ch_color at(1, 0) rx(WEST) tx(RAMP)
    csl_layout.place @left at (0, 0)
  }
}
