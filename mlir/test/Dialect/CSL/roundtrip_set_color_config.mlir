// RUN: air-opt %s | air-opt | FileCheck %s

// CHECK-LABEL: csl.wafer @w
csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl.color @send
    // CHECK: csl_layout.set_color_config @send at(0, 0) rx(RAMP) tx(EAST)
    csl_layout.set_color_config @send at(0, 0) rx(RAMP) tx(EAST)
    // CHECK: csl_layout.set_color_config @send at(1, 0) rx(WEST) tx(RAMP)
    csl_layout.set_color_config @send at(1, 0) rx(WEST) tx(RAMP)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (1, 0)
  }
}
