// RUN: air-opt %s -air-to-csl | FileCheck %s
//
// Verifies -air-to-csl lowers a 1x1 vecadd air.herd to csl.wafer v2 IR.

// CHECK-LABEL: csl.wafer @vecadd
// CHECK-SAME:  arch = "wse3"
// CHECK:         csl.program @h
// CHECK:           csl.var @arg0 : memref<256xf32>
// CHECK:           csl.var @arg1 : memref<256xf32>
// CHECK:           csl.var @arg2 : memref<256xf32>
// CHECK:           csl.func @compute
// CHECK:             scf.for
// CHECK:               memref.load
// CHECK:               arith.addf
// CHECK:               memref.store
// CHECK:           csl.export @arg0 {alias = "arg0"}
// CHECK:           csl.export @arg1 {alias = "arg1"}
// CHECK:           csl.export @arg2 {alias = "arg2"}
// CHECK:           csl.export @compute {kind = "func"}
// CHECK:         csl.layout
// CHECK-SAME:    width = 1
// CHECK:           csl_layout.place @h
// CHECK:           csl_layout.export "arg0" from @h::@arg0
// CHECK:           csl_layout.export "arg1" from @h::@arg1
// CHECK:           csl_layout.export "arg2" from @h::@arg2
// CHECK:           csl_layout.export "compute" from @h::@compute {kind = "func"}
// CHECK:         csl.host @vecadd
// CHECK:           csl_host.memcpy_h2d %{{.*}} to @vecadd_layout::@arg0
// CHECK:           csl_host.memcpy_h2d %{{.*}} to @vecadd_layout::@arg1
// CHECK:           csl_host.launch @vecadd_layout::@compute
// CHECK:           csl_host.memcpy_d2h @vecadd_layout::@arg2 to %{{.*}}
//
// Verify the original func.func and air.herd have been erased.
// CHECK-NOT: func.func @vecadd
// CHECK-NOT: air.herd

module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>,
                    %c: memref<256xf32>) {
    %one = arith.constant 1 : index
    air.launch (%tx) in (%sx=%one) args(%la=%a, %lb=%b, %lc=%c)
        : memref<256xf32>, memref<256xf32>, memref<256xf32> {
      air.segment @seg0 args(%sa=%la, %sb=%lb, %sc=%lc)
          : memref<256xf32>, memref<256xf32>, memref<256xf32> {
        %one2 = arith.constant 1 : index
        air.herd @h tile(%htx, %hty) in (%hsx=%one2, %hsy=%one2)
            args(%ha=%sa, %hb=%sb, %hc=%sc)
            : memref<256xf32>, memref<256xf32>, memref<256xf32> {
          %lo = arith.constant 0 : index
          %hi = arith.constant 256 : index
          %step = arith.constant 1 : index
          scf.for %i = %lo to %hi step %step {
            %va = memref.load %ha[%i] : memref<256xf32>
            %vb = memref.load %hb[%i] : memref<256xf32>
            %vc = arith.addf %va, %vb : f32
            memref.store %vc, %hc[%i] : memref<256xf32>
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
