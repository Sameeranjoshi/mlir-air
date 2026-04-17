// RUN: air-opt --verify-roundtrip %s | FileCheck %s
//
// csl_layout dialect round-trip tests.
// Covers: csl_layout.place (point form), csl_layout.export.
// Subgrid/range-form `csl_layout.place` round-trips live in
// mlir/test/Targets/CSLEmit/e2e/layouts.mlir.

// ---- csl_layout.place (no params) ----

// CHECK-LABEL: csl.wafer @w_place_no_params
// CHECK:   csl_layout.place @vecadd_pe at (0, 0)
module {
  csl.wafer @w_place_no_params {arch = "wse3"} {
    csl.program @vecadd_pe {
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
      csl_layout.place @vecadd_pe at (0, 0)
    }
  }
}

// ---- csl_layout.place with params ----

// CHECK-LABEL: csl.wafer @w_place_with_params
// CHECK:   csl_layout.place @gemv_pe at (1, 0) {col = 1 : i16, width = 4 : i16}
module {
  csl.wafer @w_place_with_params {arch = "wse3"} {
    csl.program @gemv_pe(%M: !csl.comptime<i16>) {
    }
    csl.layout {width = 4 : i64, height = 1 : i64} @gemv_layout {
      csl_layout.place @gemv_pe at (1, 0) {col = 1 : i16, width = 4 : i16}
    }
  }
}

// ---- csl_layout.export (var) ----

// CHECK-LABEL: csl.wafer @w_layout_export
// CHECK:   csl_layout.export "a" from @vecadd_pe::@a
module {
  csl.wafer @w_layout_export {arch = "wse3"} {
    csl.program @vecadd_pe {
      %a = csl.var @a : memref<256xf32>
      csl.export @a {alias = "a"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @export_layout {
      csl_layout.place @vecadd_pe at (0, 0)
      csl_layout.export "a" from @vecadd_pe::@a
    }
  }
}

// ---- csl_layout.export (func) ----

// CHECK-LABEL: csl.wafer @w_func_export
// CHECK:   csl_layout.export "compute" from @vecadd_pe::@compute {kind = "func"}
module {
  csl.wafer @w_func_export {arch = "wse3"} {
    csl.program @vecadd_pe {
      csl.func @compute { csl.return }
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @func_layout {
      csl_layout.place @vecadd_pe at (0, 0)
      csl_layout.export "compute" from @vecadd_pe::@compute {kind = "func"}
    }
  }
}

// ---- full vecadd layout ----

// CHECK-LABEL: csl.wafer @full_layout
// CHECK:   csl_layout.place @vecadd_pe at (0, 0)
// CHECK:   csl_layout.export "a" from @vecadd_pe::@a
// CHECK:   csl_layout.export "b" from @vecadd_pe::@b
// CHECK:   csl_layout.export "c" from @vecadd_pe::@c
// CHECK:   csl_layout.export "compute" from @vecadd_pe::@compute {kind = "func"}
module {
  csl.wafer @full_layout {arch = "wse3"} {
    csl.program @vecadd_pe {
      %a = csl.var @a : memref<256xf32>
      %b = csl.var @b : memref<256xf32>
      %c = csl.var @c : memref<256xf32>
      csl.func @compute { csl.return }
      csl.export @a {alias = "a"}
      csl.export @b {alias = "b"}
      csl.export @c {alias = "c"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
      csl_layout.place @vecadd_pe at (0, 0)
      csl_layout.export "a" from @vecadd_pe::@a
      csl_layout.export "b" from @vecadd_pe::@b
      csl_layout.export "c" from @vecadd_pe::@c
      csl_layout.export "compute" from @vecadd_pe::@compute {kind = "func"}
    }
  }
}
