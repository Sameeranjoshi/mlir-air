// RUN: not air-opt -split-input-file %s 2>&1 | FileCheck %s

csl.wafer @w1 {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 4 : i64, height = 4 : i64} @layout {
    // CHECK: error: 'csl_layout.stream' op requires single-hop cardinal route; got delta (2, 1)
    csl_layout.stream @bad from(0, 0) to(2, 1)
    csl_layout.place @p at (0, 0)
  }
}

// -----

csl.wafer @w2 {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK: error: 'csl_layout.stream' op references undefined color '@nope'
    csl_layout.stream @bad from(0, 0) to(1, 0) {color = @nope}
    csl_layout.place @p at (0, 0)
  }
}
