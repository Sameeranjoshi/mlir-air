// RUN: not air-opt %s 2>&1 | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK: error: 'csl_layout.set_color_config' op references undefined color symbol '@nope'
    csl_layout.set_color_config @nope at(0, 0) rx(RAMP) tx(EAST)
    csl_layout.place @p at (0, 0)
  }
}
