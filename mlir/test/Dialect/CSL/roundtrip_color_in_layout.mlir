// RUN: air-opt %s | air-opt | FileCheck %s

// CHECK-LABEL: csl.wafer @w
csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  // CHECK: csl.layout
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK: csl.color @red
    csl.color @red
    // CHECK: csl.color @blue {id = 5 : i32}
    csl.color @blue {id = 5 : i32}
    csl_layout.place @p at (0, 0)
  }
}
