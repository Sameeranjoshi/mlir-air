// RUN: air-opt --csl-materialize-stream-colors --csl-materialize-stream-colors %s | FileCheck %s
// Running the pass twice is a no-op (no duplicate colors, no errors).

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK: csl.color @ch_color
    // CHECK-NOT: csl.color
    // CHECK: csl_layout.stream @ch from(0, 0) to(1, 0) {color = @ch_color}
    csl_layout.stream @ch from(0, 0) to(1, 0)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (1, 0)
  }
}
