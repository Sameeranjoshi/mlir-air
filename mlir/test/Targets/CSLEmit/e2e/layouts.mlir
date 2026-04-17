// RUN: air-opt %s | air-opt | FileCheck %s
//
// Round-trip test for the two forms of csl_layout.place.  Covers:
//   * point form:   `at (x, y)`
//   * range forms:  `over [lo:hi, Y]`, `over [lo:hi, lo:hi]`
//   * range form with `vars (...) params {...}`

// CHECK-LABEL: csl.wafer @place_point
csl.wafer @place_point {arch = "wse3"} {
  csl.program @pe { csl.func @compute { csl.return } }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    // CHECK: csl_layout.place @pe at (0, 0)
    csl_layout.place @pe at (0, 0)
  }
  csl.host @main() {layout = @layout} {
    csl_host.launch @layout::@compute
  }
}

// CHECK-LABEL: csl.wafer @place_row
csl.wafer @place_row {arch = "wse3"} {
  csl.program @pe { csl.func @compute { csl.return } }
  csl.layout {width = 8 : i64, height = 1 : i64} @layout {
    // CHECK: csl_layout.place @pe over [0:8, 0]
    csl_layout.place @pe over [0:8, 0]
  }
  csl.host @main() {layout = @layout} {
    csl_host.launch @layout::@compute
  }
}

// CHECK-LABEL: csl.wafer @place_subgrid
csl.wafer @place_subgrid {arch = "wse3"} {
  csl.program @pe { csl.func @compute { csl.return } }
  csl.layout {width = 4 : i64, height = 4 : i64} @layout {
    // CHECK: csl_layout.place @pe over [0:4, 0:4]
    csl_layout.place @pe over [0:4, 0:4]
  }
  csl.host @main() {layout = @layout} {
    csl_host.launch @layout::@compute
  }
}

// CHECK-LABEL: csl.wafer @place_row_vars
csl.wafer @place_row_vars {arch = "wse3"} {
  csl.program @pe { csl.func @compute { csl.return } }
  csl.layout {width = 8 : i64, height = 1 : i64} @layout {
    // CHECK: csl_layout.place @pe over [0:8, 0] vars (%i : i32) params {pid = %i : i16}
    csl_layout.place @pe over [0:8, 0] vars (%i : i32) params {pid = %i : i16}
  }
  csl.host @main() {layout = @layout} {
    csl_host.launch @layout::@compute
  }
}
