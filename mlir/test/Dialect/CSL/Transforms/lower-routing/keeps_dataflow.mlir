// RUN: air-opt --csl-lower-dataflow-routing %s | FileCheck %s

// After Pass 3 the csl_layout.dataflow op must STILL be present
// (Pass 4 uses it for stream.put/get symbol resolution).
csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl.color @c0 {id = 0 : i32}
    // CHECK: csl_layout.dataflow @ch from(0, 0) to(1, 0) {color = @c0}
    csl_layout.dataflow @ch from(0, 0) to(1, 0) {color = @c0}
    csl_layout.place @p at (0, 0)
  }
}
