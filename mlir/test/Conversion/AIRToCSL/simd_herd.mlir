// RUN: air-opt %s -air-to-csl | FileCheck %s
//
// Verifies -air-to-csl lowers N-PE air.herd shapes to the subgrid range form
// of csl_layout.place. Covers:
//   * 8x1 herd -> over [0:8, 0], layout width = 8, height = 1
//   * 4x4 herd -> over [0:4, 0:4], layout width = 4, height = 4
//   * 1x4 herd -> over [0, 0:4], layout width = 1, height = 4

// CHECK-LABEL: csl.wafer @simd_1d
// CHECK:         csl.layout
// CHECK-SAME:    height = 1
// CHECK-SAME:    width = 8
// CHECK:           csl_layout.place @{{.*}} over [0:8, 0]
// CHECK:         csl.host @simd_1d
// CHECK:           csl_host.memcpy_h2d
// CHECK-SAME:      height = 1
// CHECK-SAME:      width = 8

// CHECK-LABEL: csl.wafer @simd_2d
// CHECK:         csl.layout
// CHECK-SAME:    height = 4
// CHECK-SAME:    width = 4
// CHECK:           csl_layout.place @{{.*}} over [0:4, 0:4]
// CHECK:         csl.host @simd_2d
// CHECK:           csl_host.memcpy_h2d
// CHECK-SAME:      height = 4
// CHECK-SAME:      width = 4

// CHECK-LABEL: csl.wafer @simd_col
// CHECK:         csl.layout
// CHECK-SAME:    height = 4
// CHECK-SAME:    width = 1
// CHECK:           csl_layout.place @{{.*}} over [0, 0:4]
// CHECK:         csl.host @simd_col
// CHECK:           csl_host.memcpy_h2d
// CHECK-SAME:      height = 4
// CHECK-SAME:      width = 1

module {
  func.func @simd_1d(%a: memref<256xf32>, %b: memref<256xf32>) {
    %one = arith.constant 1 : index
    air.launch (%tx) in (%sx=%one) args(%la=%a, %lb=%b)
        : memref<256xf32>, memref<256xf32> {
      air.segment @seg args(%sa=%la, %sb=%lb)
          : memref<256xf32>, memref<256xf32> {
        %N = arith.constant 8 : index
        %one2 = arith.constant 1 : index
        air.herd @pe tile(%htx, %hty) in (%hsx=%N, %hsy=%one2)
            args(%ha=%sa, %hb=%sb)
            : memref<256xf32>, memref<256xf32> {
          %lo = arith.constant 0 : index
          %hi = arith.constant 256 : index
          %step = arith.constant 1 : index
          scf.for %i = %lo to %hi step %step {
            %va = memref.load %ha[%i] : memref<256xf32>
            memref.store %va, %hb[%i] : memref<256xf32>
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }

  func.func @simd_2d(%a: memref<256xf32>, %b: memref<256xf32>) {
    %one = arith.constant 1 : index
    air.launch (%tx) in (%sx=%one) args(%la=%a, %lb=%b)
        : memref<256xf32>, memref<256xf32> {
      air.segment @seg args(%sa=%la, %sb=%lb)
          : memref<256xf32>, memref<256xf32> {
        %M = arith.constant 4 : index
        air.herd @pe tile(%htx, %hty) in (%hsx=%M, %hsy=%M)
            args(%ha=%sa, %hb=%sb)
            : memref<256xf32>, memref<256xf32> {
          %lo = arith.constant 0 : index
          %hi = arith.constant 256 : index
          %step = arith.constant 1 : index
          scf.for %i = %lo to %hi step %step {
            %va = memref.load %ha[%i] : memref<256xf32>
            memref.store %va, %hb[%i] : memref<256xf32>
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }

  func.func @simd_col(%a: memref<256xf32>, %b: memref<256xf32>) {
    %one = arith.constant 1 : index
    air.launch (%tx) in (%sx=%one) args(%la=%a, %lb=%b)
        : memref<256xf32>, memref<256xf32> {
      air.segment @seg args(%sa=%la, %sb=%lb)
          : memref<256xf32>, memref<256xf32> {
        %one2 = arith.constant 1 : index
        %M = arith.constant 4 : index
        air.herd @pe tile(%htx, %hty) in (%hsx=%one2, %hsy=%M)
            args(%ha=%sa, %hb=%sb)
            : memref<256xf32>, memref<256xf32> {
          %lo = arith.constant 0 : index
          %hi = arith.constant 256 : index
          %step = arith.constant 1 : index
          scf.for %i = %lo to %hi step %step {
            %va = memref.load %ha[%i] : memref<256xf32>
            memref.store %va, %hb[%i] : memref<256xf32>
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
