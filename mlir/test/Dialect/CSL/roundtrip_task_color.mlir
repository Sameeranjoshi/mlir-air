// RUN: air-opt %s | air-opt | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    csl.func @c { csl.return }
    // CHECK: csl.task @recv attributes {color = @send, trigger_kind = "color"}
    csl.task @recv attributes {trigger_kind = "color", color = @send} {
      csl.return
    }
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    csl.color @send
    csl_layout.place @p at (0, 0)
  }
}
