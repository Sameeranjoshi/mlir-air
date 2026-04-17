// RUN: air-opt %s | air-opt | FileCheck %s
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck --check-prefix=POINT-LAYOUT  %s < %t/place_point/layout.csl
// RUN: FileCheck --check-prefix=ROW-LAYOUT    %s < %t/place_row/layout.csl
// RUN: FileCheck --check-prefix=GRID-LAYOUT   %s < %t/place_subgrid/layout.csl
// RUN: FileCheck --check-prefix=PARAMS-LAYOUT %s < %t/place_row_vars/layout.csl
//
// Round-trip test for the two forms of csl_layout.place.  Covers:
//   * point form:   `at (x, y)`
//   * range forms:  `over [lo:hi, Y]`, `over [lo:hi, lo:hi]`
//   * range form with `vars (...) params {...}`
// Also checks the layout.csl emission for each form.

// POINT-LAYOUT: @set_rectangle(1, 1);
// POINT-LAYOUT: @set_tile_code(0, 0, "pe.csl"

// ROW-LAYOUT: @set_rectangle(8, 1);
// ROW-LAYOUT: var i: i16 = 0;
// ROW-LAYOUT: while (i < 8) : (i += 1) {
// ROW-LAYOUT:   @set_tile_code(i, 0, "pe.csl"

// GRID-LAYOUT: @set_rectangle(4, 4);
// GRID-LAYOUT: var j: i16 = 0;
// GRID-LAYOUT: while (j < 4) : (j += 1) {
// GRID-LAYOUT:   var i: i16 = 0;
// GRID-LAYOUT:   while (i < 4) : (i += 1) {
// GRID-LAYOUT:     @set_tile_code(i, j, "pe.csl"

// PARAMS-LAYOUT: var i: i16 = 0;
// PARAMS-LAYOUT: while (i < 8) : (i += 1) {
// PARAMS-LAYOUT:   @set_tile_code(i, 0, "pe.csl", .{
// PARAMS-LAYOUT:     .memcpy_params = memcpy.get_params(i),
// PARAMS-LAYOUT:     .pid = i,

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
