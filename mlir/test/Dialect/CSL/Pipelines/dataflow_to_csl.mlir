// RUN: air-opt --csl-dataflow-to-csl %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @left {
    %buf = csl.var @buf : memref<32xf32>
    csl.func @c {
      %n = arith.constant 32 : index
      csl.dataflow.put @ch source(%buf) extent(%n : index) : memref<32xf32>
      csl.return
    }
  }
  csl.program @right {
    %buf = csl.var @buf : memref<32xf32>
    csl.func @c {
      %n = arith.constant 32 : index
      csl.dataflow.get @ch target(%buf) extent(%n : index) : memref<32xf32>
      csl.return
    }
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl_layout.dataflow @ch from(0, 0) to(1, 0)
    csl_layout.place @left  at (0, 0)
    csl_layout.place @right at (1, 0)
  }
}

// After full sub-pipeline: streams gone, fabric DSDs + tasks present, set_color_config emitted.
// CHECK-NOT: csl_layout.dataflow
// CHECK-NOT: csl.dataflow.put
// CHECK-NOT: csl.dataflow.get

// Program @left: fabout DSD + async fmovs + completion task.
// CHECK-LABEL: csl.program @left
// CHECK: csl.get_fab_dsd fabout @ch_color
// CHECK: csl.builtin_call "fmovs"({{.*}}) {activate = @ch_put_done_0, async}
// CHECK: csl.task @ch_put_done_0 attributes {id = 8 : i32, trigger_kind = "local_task_id"}

// Program @right: fabin DSD + async fmovs + completion task.
// CHECK-LABEL: csl.program @right
// CHECK: csl.get_fab_dsd fabin @ch_color
// CHECK: csl.builtin_call "fmovs"({{.*}}) {activate = @ch_get_done_0, async}
// CHECK: csl.task @ch_get_done_0 attributes {id = 8 : i32, trigger_kind = "local_task_id"}

// Layout: color allocated + set_color_config on each PE.
// CHECK-LABEL: csl.layout
// CHECK: csl.color @ch_color {id = 0 : i32}
// CHECK: csl_layout.set_color_config @ch_color at(0, 0) rx(RAMP) tx(EAST)
// CHECK: csl_layout.set_color_config @ch_color at(1, 0) rx(WEST) tx(RAMP)
