// RUN: air-opt %s -csl-infer-exports | FileCheck %s
//
// Verifies -csl-infer-exports auto-generates csl.export and csl_layout.export
// from csl.host ops. Input has NO manual exports.

// CHECK-LABEL: csl.wafer @vecadd
module {
  csl.wafer @vecadd {arch = "wse3"} {
    // CHECK: csl.program @pe
    csl.program @pe {
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
      // No exports here — pass should add them:
      // CHECK: csl.export @a {alias = "a", direction = "in"}
      // CHECK: csl.export @b {alias = "b", direction = "in"}
      // CHECK: csl.export @compute {direction = "internal", kind = "func"}
      // CHECK: csl.export @c {alias = "c", direction = "out"}
    }
    // CHECK: csl.layout
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
      // No layout exports — pass should add them:
      // CHECK: csl_layout.export "a" from @pe::@a
      // CHECK: csl_layout.export "b" from @pe::@b
      // CHECK: csl_layout.export "compute" from @pe::@compute {kind = "func"}
      // CHECK: csl_layout.export "c" from @pe::@c
    }
    csl.host @main(%a_in: memref<256xf32>, %b_in: memref<256xf32>,
                   %c_out: memref<256xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.memcpy_h2d %b_in to @layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
    }
  }
}
