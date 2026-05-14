// RUN: air-opt %s | air-opt | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK: csl_layout.dataflow @send_ch from(0, 0) to(1, 0)
    csl_layout.dataflow @send_ch from(0, 0) to(1, 0)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (1, 0)
  }
}
