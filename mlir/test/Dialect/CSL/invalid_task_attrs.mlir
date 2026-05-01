// RUN: not air-opt -split-input-file %s 2>&1 | FileCheck %s

csl.wafer @w1 {arch = "wse3"} {
  csl.program @p {
    csl.func @c { csl.return }
    // CHECK: error: 'csl.task' op trigger_kind = "local_task_id" requires `id` attribute
    csl.task @t attributes {trigger_kind = "local_task_id"} { csl.return }
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout { csl_layout.place @p at (0,0) }
}

// -----

csl.wafer @w2 {arch = "wse3"} {
  csl.program @p {
    csl.func @c { csl.return }
    // CHECK: error: 'csl.task' op trigger_kind = "color" must not set `id`
    csl.task @t attributes {trigger_kind = "color", color = @x, id = 5 : i32} { csl.return }
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    csl.color @x
    csl_layout.place @p at (0,0)
  }
}

// -----

csl.wafer @w3 {arch = "wse3"} {
  csl.program @p {
    csl.func @c { csl.return }
    // CHECK: error: 'csl.task' op trigger_kind must be "local_task_id" or "color" (got "wavelet")
    csl.task @t attributes {trigger_kind = "wavelet"} { csl.return }
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout { csl_layout.place @p at (0,0) }
}

// -----

csl.wafer @w4 {arch = "wse3"} {
  csl.program @p {
    csl.func @c { csl.return }
    // CHECK: error: 'csl.task' op trigger_kind = "local_task_id" must not set `color`
    csl.task @t attributes {trigger_kind = "local_task_id", id = 1 : i32, color = @x} { csl.return }
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    csl.color @x
    csl_layout.place @p at (0,0)
  }
}

// -----

csl.wafer @w5 {arch = "wse3"} {
  csl.program @p {
    csl.func @c { csl.return }
    // CHECK: error: 'csl.task' op trigger_kind = "color" requires `color` attribute
    csl.task @t attributes {trigger_kind = "color"} { csl.return }
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout { csl_layout.place @p at (0,0) }
}
