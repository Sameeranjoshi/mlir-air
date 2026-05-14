// RUN: air-opt --verify-roundtrip %s | FileCheck %s
// RUN: air-opt --mlir-print-op-generic %s | FileCheck %s --check-prefix=GENERIC
//
// CSL dialect v2 round-trip tests.
// Covers: !csl.comptime<T>, csl.wafer, csl.program, csl.export,
//         csl.layout, csl.host

// ---- !csl.comptime<T> type ----

// CHECK-LABEL: func.func @test_comptime_i16
// CHECK: %{{.*}}: !csl.comptime<i16>
// GENERIC: !csl.comptime<i16>
func.func @test_comptime_i16(%arg0: !csl.comptime<i16>) -> !csl.comptime<i16> {
  return %arg0 : !csl.comptime<i16>
}

// CHECK-LABEL: func.func @test_comptime_index
// CHECK: %{{.*}}: !csl.comptime<index>
func.func @test_comptime_index(%arg0: !csl.comptime<index>) -> !csl.comptime<index> {
  return %arg0 : !csl.comptime<index>
}

// CHECK-LABEL: func.func @test_comptime_f32
// CHECK: %{{.*}}: !csl.comptime<f32>
func.func @test_comptime_f32(%arg0: !csl.comptime<f32>) -> !csl.comptime<f32> {
  return %arg0 : !csl.comptime<f32>
}

// ---- csl.wafer + csl.program ----

// CHECK-LABEL: module
// CHECK: csl.wafer @wseprog
// CHECK-SAME: arch = "wse3"
// CHECK:   csl.program @vecadd_pe(%{{.*}}: !csl.comptime<i16>)
module {
  csl.wafer @wseprog {arch = "wse3"} {
    csl.program @vecadd_pe(%col: !csl.comptime<i16>) {
    }
  }
}

// CHECK-LABEL: module
// CHECK: csl.wafer @multi_param
// CHECK:   csl.program @gemv_pe(
// CHECK-SAME: !csl.comptime<i16>
// CHECK-SAME: !csl.comptime<i16>
module {
  csl.wafer @multi_param {arch = "wse3"} {
    csl.program @gemv_pe(%M: !csl.comptime<i16>, %N: !csl.comptime<i16>) {
    }
  }
}

// CHECK-LABEL: module
// CHECK: csl.wafer @no_params
// CHECK:   csl.program @simple_pe
module {
  csl.wafer @no_params {arch = "wse3"} {
    csl.program @simple_pe {
    }
  }
}

// ---- csl.export inside csl.program ----

// CHECK-LABEL: module
// CHECK: csl.wafer @w_export
// CHECK:   csl.export @a {alias = "a"}
// CHECK:   csl.export @compute {kind = "func"}
module {
  csl.wafer @w_export {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<256xf32>
      csl.func @compute {
        csl.return
      }
      csl.export @a {alias = "a"}
      csl.export @compute {kind = "func"}
    }
  }
}

// ---- csl.export with direction ----

// CHECK-LABEL: module
// CHECK: csl.wafer @w_export_dir
// CHECK:   csl.export @a {alias = "a", direction = "in"}
// CHECK:   csl.export @c {alias = "c", direction = "out"}
module {
  csl.wafer @w_export_dir {arch = "wse3"} {
    csl.program @pe_dir {
      %a = csl.var @a : memref<256xf32>
      %c = csl.var @c : memref<256xf32>
      csl.export @a {alias = "a", direction = "in"}
      csl.export @c {alias = "c", direction = "out"}
    }
  }
}

// ---- csl.layout ----

// CHECK-LABEL: module
// CHECK: csl.wafer @w_layout
// CHECK:   csl.layout
// CHECK-SAME: height = 1 : i64
// CHECK-SAME: width = 1 : i64
// CHECK-SAME: @main_layout
module {
  csl.wafer @w_layout {arch = "wse3"} {
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
    }
  }
}

// ---- csl.host ----

// CHECK-LABEL: module
// CHECK: csl.wafer @w_host
// CHECK:   csl.host @main
// CHECK-SAME: layout = @main_layout
module {
  csl.wafer @w_host {arch = "wse3"} {
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
    }
    csl.host @main(%a: memref<256xf32>) {layout = @main_layout} {
    }
  }
}

// ---- full v2 structure ----

// CHECK-LABEL: module
// CHECK: csl.wafer @full_vecadd
// CHECK:   csl.program @vecadd_pe
// CHECK:     csl.var @a
// CHECK:     csl.var @b
// CHECK:     csl.var @c
// CHECK:     csl.func @compute
// CHECK:     csl.export @a {alias = "a"}
// CHECK:     csl.export @compute {kind = "func"}
// CHECK:   csl.layout
// CHECK:   csl.host @main
module {
  csl.wafer @full_vecadd {arch = "wse3"} {
    csl.program @vecadd_pe {
      %a = csl.var @a : memref<256xf32>
      %b = csl.var @b : memref<256xf32>
      %c = csl.var @c : memref<256xf32>
      csl.func @compute {
        %lo = arith.constant 0 : index
        %hi = arith.constant 256 : index
        %step = arith.constant 1 : index
        scf.for %i = %lo to %hi step %step {
          %va = memref.load %a[%i] : memref<256xf32>
          %vb = memref.load %b[%i] : memref<256xf32>
          %vc = arith.addf %va, %vb : f32
          memref.store %vc, %c[%i] : memref<256xf32>
        }
        csl.return
      }
      csl.export @a {alias = "a"}
      csl.export @b {alias = "b"}
      csl.export @c {alias = "c"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
    }
    csl.host @main(%a_in: memref<256xf32>, %b_in: memref<256xf32>,
                   %c_out: memref<256xf32>) {layout = @main_layout} {
    }
  }
}
