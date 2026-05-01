// RUN: air-opt %s | air-opt | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    csl.func @c { csl.return }
    // CHECK: csl.task @exit_task attributes {id = 8 : i32, trigger_kind = "local_task_id"}
    csl.task @exit_task attributes {trigger_kind = "local_task_id", id = 8 : i32} {
      csl.return
    }
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    csl_layout.place @p at (0, 0)
  }
}
