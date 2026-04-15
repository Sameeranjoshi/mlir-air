// RUN: air-opt -csl-derive-exports %s | FileCheck %s
//
// Tests for -csl-derive-exports pass.
// The pass scans csl.host bodies for memcpy_h2d/memcpy_d2h/launch ops
// and annotates csl.export ops with direction = "in" / "out" / "internal".

// ---- basic vecadd: a,b → in; c → out; compute → internal ----

// CHECK-LABEL: csl.wafer @vecadd
// CHECK:   csl.export @a {alias = "a", direction = "in"}
// CHECK:   csl.export @b {alias = "b", direction = "in"}
// CHECK:   csl.export @c {alias = "c", direction = "out"}
// CHECK:   csl.export @compute {direction = "internal", kind = "func"}
module {
  csl.wafer @vecadd {arch = "wse3"} {
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

// ---- read-only: only memcpy_h2d, no d2h ----

// CHECK-LABEL: csl.wafer @readonly
// CHECK:   csl.export @x {alias = "x", direction = "in"}
// CHECK:   csl.export @y {alias = "y", direction = "internal"}
module {
  csl.wafer @readonly {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<64xf32>
      %y = csl.var @y : memref<64xf32>
      csl.export @x {alias = "x"}
      csl.export @y {alias = "y"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @ro_layout {
      csl_layout.place @pe at (0, 0)
      csl_layout.export "x" from @pe::@x
      csl_layout.export "y" from @pe::@y
    }
    csl.host @main(%x_in: memref<64xf32>) {layout = @ro_layout} {
      csl_host.memcpy_h2d %x_in to @ro_layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
    }
  }
}

// ---- write-only: only memcpy_d2h, no h2d ----

// CHECK-LABEL: csl.wafer @writeonly
// CHECK:   csl.export @result {alias = "result", direction = "out"}
module {
  csl.wafer @writeonly {arch = "wse3"} {
    csl.program @pe {
      %result = csl.var @result : memref<64xf32>
      csl.export @result {alias = "result"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @wo_layout {
      csl_layout.place @pe at (0, 0)
      csl_layout.export "result" from @pe::@result
    }
    csl.host @main(%out: memref<64xf32>) {layout = @wo_layout} {
      csl_host.memcpy_d2h @wo_layout::@result to %out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
    }
  }
}
