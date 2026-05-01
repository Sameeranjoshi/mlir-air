// RUN: not air-opt %s 2>&1 | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    // CHECK: error: 'csl.color' op expects parent op 'csl.layout'
    csl.color @red
    csl.func @c { csl.return }
  }
}
