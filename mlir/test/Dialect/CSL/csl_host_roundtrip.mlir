// RUN: air-opt --verify-roundtrip %s | FileCheck %s
//
// csl_host dialect round-trip tests.
// Covers: csl_host.memcpy_h2d, csl_host.memcpy_d2h, csl_host.launch

// ---- csl_host.memcpy_h2d ----

// CHECK-LABEL: csl.wafer @w_h2d
// CHECK:   csl_host.memcpy_h2d %{{[^ ]*}} to @main_layout::@a {height = 1 : i64, px = 0 : i64, py = 0 : i64, width = 1 : i64} : memref<256xf32>
module {
  csl.wafer @w_h2d {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<256xf32>
      csl.export @a {alias = "a"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
      csl_layout.place @pe at (0, 0)
      csl_layout.export "a" from @pe::@a
    }
    csl.host @main(%a_in: memref<256xf32>) {layout = @main_layout} {
      csl_host.memcpy_h2d %a_in to @main_layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
    }
  }
}

// ---- csl_host.memcpy_d2h ----

// CHECK-LABEL: csl.wafer @w_d2h
// CHECK:   csl_host.memcpy_d2h @main_layout::@c to %{{[^ ]*}} {height = 1 : i64, px = 0 : i64, py = 0 : i64, width = 1 : i64} : memref<256xf32>
module {
  csl.wafer @w_d2h {arch = "wse3"} {
    csl.program @pe {
      %c = csl.var @c : memref<256xf32>
      csl.export @c {alias = "c"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
      csl_layout.place @pe at (0, 0)
      csl_layout.export "c" from @pe::@c
    }
    csl.host @main(%c_out: memref<256xf32>) {layout = @main_layout} {
      csl_host.memcpy_d2h @main_layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
    }
  }
}

// ---- csl_host.launch ----

// CHECK-LABEL: csl.wafer @w_launch
// CHECK:   csl_host.launch @main_layout::@compute
module {
  csl.wafer @w_launch {arch = "wse3"} {
    csl.program @pe {
      csl.func @compute { csl.return }
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
      csl_layout.place @pe at (0, 0)
      csl_layout.export "compute" from @pe::@compute {kind = "func"}
    }
    csl.host @main() {layout = @main_layout} {
      csl_host.launch @main_layout::@compute
    }
  }
}

// ---- full vecadd host: h2d + launch + d2h ----

// CHECK-LABEL: csl.wafer @full_host
// CHECK:   csl_host.memcpy_h2d %{{.*}} to @main_layout::@a
// CHECK:   csl_host.memcpy_h2d %{{.*}} to @main_layout::@b
// CHECK:   csl_host.launch @main_layout::@compute
// CHECK:   csl_host.memcpy_d2h @main_layout::@c to %{{.*}}
module {
  csl.wafer @full_host {arch = "wse3"} {
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
      csl_layout.place @vecadd_pe at (0, 0)
      csl_layout.export "a" from @vecadd_pe::@a
      csl_layout.export "b" from @vecadd_pe::@b
      csl_layout.export "c" from @vecadd_pe::@c
      csl_layout.export "compute" from @vecadd_pe::@compute {kind = "func"}
    }
    csl.host @main(%a_in: memref<256xf32>, %b_in: memref<256xf32>,
                   %c_out: memref<256xf32>) {layout = @main_layout} {
      csl_host.memcpy_h2d %a_in to @main_layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.memcpy_h2d %b_in to @main_layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.launch @main_layout::@compute
      csl_host.memcpy_d2h @main_layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
    }
  }
}
