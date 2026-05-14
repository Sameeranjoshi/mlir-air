// RUN: not air-opt -split-input-file %s 2>&1 | FileCheck %s

csl.wafer @w1 {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 4 : i64, height = 4 : i64} @layout {
    // CHECK: error: 'csl_layout.dataflow' op requires cardinal route along a single axis; got delta (2, 1)
    csl_layout.dataflow @bad from(0, 0) to(2, 1)
    csl_layout.place @p at (0, 0)
  }
}

// -----

csl.wafer @w2 {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK: error: 'csl_layout.dataflow' op references undefined color '@nope'
    csl_layout.dataflow @bad from(0, 0) to(1, 0) {color = @nope}
    csl_layout.place @p at (0, 0)
  }
}

// -----

csl.wafer @w_self {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2, height = 1} @layout {
    // CHECK: error: 'csl_layout.dataflow' op requires cardinal route along a single axis; got delta (0, 0)
    csl_layout.dataflow @self_edge from(1, 0) to(1, 0)
    csl_layout.place @p at (0, 0)
  }
}

// -----

csl.wafer @w_putty {arch = "wse3"} {
  csl.program @p {
    %buf = csl.var @buf : memref<128xi32>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: error: 'csl.dataflow.put' op source memref element type must be f32
      csl.dataflow.put @ch source(%buf) extent(%n : index) : memref<128xi32>
      csl.return
    }
  }
  csl.layout {width = 2, height = 1} @layout {
    csl_layout.dataflow @ch from(0, 0) to(1, 0)
    csl_layout.place @p at (0, 0)
  }
}

// -----

csl.wafer @w_putnope {arch = "wse3"} {
  csl.program @p {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: error: 'csl.dataflow.put' op references undefined stream '@nope'
      csl.dataflow.put @nope source(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
  }
  csl.layout {width = 1, height = 1} @layout { csl_layout.place @p at (0,0) }
}

// -----

csl.wafer @w_getty {arch = "wse3"} {
  csl.program @p {
    %buf = csl.var @buf : memref<128xf16>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: error: 'csl.dataflow.get' op target memref element type must be f32
      csl.dataflow.get @ch target(%buf) extent(%n : index) : memref<128xf16>
      csl.return
    }
  }
  csl.layout {width = 2, height = 1} @layout {
    csl_layout.dataflow @ch from(0, 0) to(1, 0)
    csl_layout.place @p at (1, 0)
  }
}
