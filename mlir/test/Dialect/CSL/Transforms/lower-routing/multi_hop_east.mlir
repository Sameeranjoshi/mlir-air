// RUN: air-opt --csl-lower-stream-routing %s | FileCheck %s

// Multi-hop east route: from(0,0) to(3,0). The routing pass should emit:
//   - source endpoint at (0,0): rx=RAMP, tx=EAST
//   - intermediate at  (1,0): rx=WEST, tx=EAST
//   - intermediate at  (2,0): rx=WEST, tx=EAST
//   - dest endpoint   at (3,0): rx=WEST, tx=RAMP

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 4 : i64, height = 1 : i64} @layout {
    csl.color @c0 {id = 0 : i32}
    csl_layout.stream @ch from(0, 0) to(3, 0) {color = @c0}
    // CHECK: csl_layout.set_color_config @c0 at(0, 0) rx(RAMP) tx(EAST)
    // CHECK: csl_layout.set_color_config @c0 at(1, 0) rx(WEST) tx(EAST)
    // CHECK: csl_layout.set_color_config @c0 at(2, 0) rx(WEST) tx(EAST)
    // CHECK: csl_layout.set_color_config @c0 at(3, 0) rx(WEST) tx(RAMP)
    csl_layout.place @p at (0, 0)
  }
}
