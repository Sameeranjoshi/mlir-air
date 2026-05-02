// RUN: air-opt --csl-materialize-dataflow-colors %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK: csl.color @send_ch_color
    // CHECK: csl_layout.dataflow @send_ch from(0, 0) to(1, 0) {color = @send_ch_color}
    csl_layout.dataflow @send_ch from(0, 0) to(1, 0)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (1, 0)
  }
}
