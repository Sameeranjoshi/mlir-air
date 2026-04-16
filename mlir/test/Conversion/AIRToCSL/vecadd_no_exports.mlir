// RUN: air-opt %s -air-to-csl -csl-infer-exports | FileCheck %s
//
// Full pipeline: -air-to-csl creates structure, -csl-infer-exports adds exports.

// CHECK-LABEL: csl.wafer @vecadd
// CHECK: csl.program @h
// CHECK:   csl.var @arg0 : memref<256xf32>
// CHECK:   csl.var @arg1 : memref<256xf32>
// CHECK:   csl.var @arg2 : memref<256xf32>
// CHECK:   csl.func @compute
// CHECK:   csl.export @arg0 {alias = "arg0", direction = "in"}
// CHECK:   csl.export @arg1 {alias = "arg1", direction = "in"}
// CHECK:   csl.export @compute {direction = "internal", kind = "func"}
// CHECK:   csl.export @arg2 {alias = "arg2", direction = "out"}
// CHECK: csl.layout
// CHECK:   csl_layout.place @h at
// CHECK:   csl_layout.export "arg0" from @h::@arg0
// CHECK:   csl_layout.export "arg1" from @h::@arg1
// CHECK:   csl_layout.export "compute" from @h::@compute {kind = "func"}
// CHECK:   csl_layout.export "arg2" from @h::@arg2
// CHECK: csl.host @vecadd
// CHECK:   csl_host.memcpy_h2d
// CHECK:   csl_host.memcpy_h2d
// CHECK:   csl_host.launch
// CHECK:   csl_host.memcpy_d2h

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
