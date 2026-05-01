// RUN: not air-opt -split-input-file %s 2>&1 | FileCheck %s

csl.wafer @w1 {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 4 : i64, height = 4 : i64} @layout {
    // CHECK: error: 'csl_layout.stream' op requires single-hop cardinal route; got delta (2, 1)
    csl_layout.stream @bad from(0, 0) to(2, 1)
    csl_layout.place @p at (0, 0)
  }
}

// -----

csl.wafer @w2 {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK: error: 'csl_layout.stream' op references undefined color '@nope'
    csl_layout.stream @bad from(0, 0) to(1, 0) {color = @nope}
    csl_layout.place @p at (0, 0)
  }
}

// -----

csl.wafer @w_self {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2, height = 1} @layout {
    // CHECK: error: 'csl_layout.stream' op requires single-hop cardinal route; got delta (0, 0)
    csl_layout.stream @self_edge from(1, 0) to(1, 0)
    csl_layout.place @p at (0, 0)
  }
}

// -----

csl.wafer @w_putty {arch = "wse3"} {
  csl.program @p {
    %buf = csl.var @buf : memref<128xi32>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: error: 'csl.stream.put' op source memref element type must be f32
      csl.stream.put @ch source(%buf) extent(%n : index) : memref<128xi32>
      csl.return
    }
  }
  csl.layout {width = 2, height = 1} @layout {
    csl_layout.stream @ch from(0, 0) to(1, 0)
    csl_layout.place @p at (0, 0)
  }
}

// -----

csl.wafer @w_putnope {arch = "wse3"} {
  csl.program @p {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: error: 'csl.stream.put' op references undefined stream '@nope'
      csl.stream.put @nope source(%buf) extent(%n : index) : memref<128xf32>
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
      // CHECK: error: 'csl.stream.get' op target memref element type must be f32
      csl.stream.get @ch target(%buf) extent(%n : index) : memref<128xf16>
      csl.return
    }
  }
  csl.layout {width = 2, height = 1} @layout {
    csl_layout.stream @ch from(0, 0) to(1, 0)
    csl_layout.place @p at (1, 0)
  }
}
