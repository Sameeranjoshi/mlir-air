// RUN: air-opt --csl-lower-stream-routing %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl.color @c0 {id = 0 : i32}
    csl_layout.stream @ch from(1, 0) to(0, 0) {color = @c0}
    // CHECK: csl_layout.set_color_config @c0 at(1, 0) rx(RAMP) tx(WEST)
    // CHECK: csl_layout.set_color_config @c0 at(0, 0) rx(EAST) tx(RAMP)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (1, 0)
  }
}
