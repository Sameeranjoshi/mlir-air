// RUN: air-opt %s | air-opt | FileCheck %s

// CHECK-LABEL: csl.wafer @w
csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @c {
      %c128 = arith.constant 128 : index
      // CHECK: csl.get_fab_dsd fabout @send extent(%c128
      %out = csl.get_fab_dsd fabout @send extent(%c128 : index) : !csl.dsd
      // CHECK: csl.get_fab_dsd fabin @send extent(%c128
      %in  = csl.get_fab_dsd fabin  @send extent(%c128 : index) : !csl.dsd
      csl.return
    }
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl.color @send
    csl_layout.place @p at (0, 0)
  }
}
