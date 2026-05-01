// RUN: air-opt --csl-materialize-stream-colors %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 3 : i64, height = 1 : i64} @layout {
    // CHECK-DAG: csl.color @ab_color
    // CHECK-DAG: csl.color @bc_color
    // CHECK: csl_layout.stream @ab from(0, 0) to(1, 0) {color = @ab_color}
    csl_layout.stream @ab from(0, 0) to(1, 0)
    // CHECK: csl_layout.stream @bc from(1, 0) to(2, 0) {color = @bc_color}
    csl_layout.stream @bc from(1, 0) to(2, 0)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (1, 0)
    csl_layout.place @p at (2, 0)
  }
}
