// RUN: air-opt --csl-allocate-color-ids %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    // CHECK: csl.color @pinned {id = 0 : i32}
    csl.color @pinned {id = 0 : i32}
    // CHECK: csl.color @v {id = 1 : i32}
    csl.color @v
    csl_layout.place @p at (0, 0)
  }
}
