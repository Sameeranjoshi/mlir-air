// RUN: not air-opt %s 2>&1 | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    csl.func @c {
      %n = arith.constant 4 : index
      // CHECK: error: 'csl.get_fab_dsd' op references undefined color symbol '@nope'
      %d = csl.get_fab_dsd fabout @nope extent(%n : index) : !csl.dsd
      csl.return
    }
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    csl_layout.place @p at (0, 0)
  }
}
