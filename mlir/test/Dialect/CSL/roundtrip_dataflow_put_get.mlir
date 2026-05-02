// RUN: air-opt %s | air-opt | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @left {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: csl.dataflow.put @ch source(%{{.*}}) extent(%{{.*}}) : memref<128xf32>
      csl.dataflow.put @ch source(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
  }
  csl.program @right {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: csl.dataflow.get @ch target(%{{.*}}) extent(%{{.*}}) : memref<128xf32>
      csl.dataflow.get @ch target(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl_layout.dataflow @ch from(0, 0) to(1, 0)
    csl_layout.place @left  at (0, 0)
    csl_layout.place @right at (1, 0)
  }
}
