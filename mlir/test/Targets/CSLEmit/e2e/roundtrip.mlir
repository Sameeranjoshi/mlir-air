// RUN: air-opt %s | air-opt | FileCheck %s
//
// Task 10: pure parse-print sanity check for every new form introduced in
// tasks 3 (func.func private / func.call inside csl.program) and 5
// (csl_layout.place subgrid range forms + vars/params).
//
// Each wafer exercises one form. No emit pipeline.

// CHECK-LABEL: csl.wafer @r_place_point
// CHECK:   csl_layout.place @pe at (0, 0)
csl.wafer @r_place_point {arch = "wse3"} {
  csl.program @pe { csl.func @compute { csl.return } }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    csl_layout.place @pe at (0, 0)
  }
  csl.host @main() {layout = @layout} {
    csl_host.launch @layout::@compute
  }
}

// CHECK-LABEL: csl.wafer @r_place_row
// CHECK:   csl_layout.place @pe over [0:8, 0]
csl.wafer @r_place_row {arch = "wse3"} {
  csl.program @pe { csl.func @compute { csl.return } }
  csl.layout {width = 8 : i64, height = 1 : i64} @layout {
    csl_layout.place @pe over [0:8, 0]
  }
  csl.host @main() {layout = @layout} {
    csl_host.launch @layout::@compute
  }
}

// CHECK-LABEL: csl.wafer @r_place_subgrid
// CHECK:   csl_layout.place @pe over [0:4, 0:4]
csl.wafer @r_place_subgrid {arch = "wse3"} {
  csl.program @pe { csl.func @compute { csl.return } }
  csl.layout {width = 4 : i64, height = 4 : i64} @layout {
    csl_layout.place @pe over [0:4, 0:4]
  }
  csl.host @main() {layout = @layout} {
    csl_host.launch @layout::@compute
  }
}

// CHECK-LABEL: csl.wafer @r_place_row_vars
// CHECK:   csl_layout.place @pe over [0:8, 0] vars (%{{.+}} : i32) params {pid = %{{.+}} : i16}
csl.wafer @r_place_row_vars {arch = "wse3"} {
  csl.program @pe { csl.func @compute { csl.return } }
  csl.layout {width = 8 : i64, height = 1 : i64} @layout {
    csl_layout.place @pe over [0:8, 0] vars (%i : i32) params {pid = %i : i16}
  }
  csl.host @main() {layout = @layout} {
    csl_host.launch @layout::@compute
  }
}

// CHECK-LABEL: csl.wafer @r_helper_func
// CHECK:   func.func private @helper
// CHECK:     arith.mulf
// CHECK:     return
// CHECK:   csl.func @compute
// CHECK:     func.call @helper
csl.wafer @r_helper_func {arch = "wse3"} {
  csl.program @pe {
    %a = csl.var @a : memref<16xf32>
    %b = csl.var @b : memref<16xf32>
    func.func private @helper(%x: f32, %y: f32) -> f32 {
      %m = arith.mulf %x, %y : f32
      func.return %m : f32
    }
    csl.func @compute {
      %c0 = arith.constant 0 : index
      %n = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %n step %c1 {
        %va = memref.load %a[%i] : memref<16xf32>
        %vr = func.call @helper(%va, %va) : (f32, f32) -> f32
        memref.store %vr, %b[%i] : memref<16xf32>
      }
      csl.return
    }
    csl.export @a {alias = "a"}
    csl.export @b {alias = "b"}
    csl.export @compute {kind = "func"}
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    csl_layout.place @pe at (0, 0)
  }
  csl.host @main() {layout = @layout} {
    csl_host.launch @layout::@compute
  }
}
