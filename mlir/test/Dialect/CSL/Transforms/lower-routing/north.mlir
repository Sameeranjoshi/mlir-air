// RUN: air-opt --csl-lower-stream-routing %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 1 : i64, height = 2 : i64} @layout {
    csl.color @c0 {id = 0 : i32}
    csl_layout.stream @ch from(0, 1) to(0, 0) {color = @c0}
    // CHECK: csl_layout.set_color_config @c0 at(0, 1) rx(RAMP) tx(NORTH)
    // CHECK: csl_layout.set_color_config @c0 at(0, 0) rx(SOUTH) tx(RAMP)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (0, 1)
  }
}
