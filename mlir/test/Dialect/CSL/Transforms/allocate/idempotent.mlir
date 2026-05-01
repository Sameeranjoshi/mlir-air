// RUN: air-opt --csl-allocate-color-ids --csl-allocate-color-ids %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    // CHECK: csl.color @c {id = 0 : i32}
    csl.color @c
    csl_layout.place @p at (0, 0)
  }
}
