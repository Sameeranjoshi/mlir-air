// RUN: air-opt %s -air-to-csl-dialect | FileCheck %s

// CHECK-LABEL: func.func @vecadd
// CHECK:   csl.spatial_placement {
// CHECK:     %[[K:.*]] = csl.kernel {
// CHECK-DAG:    csl.var @a_buf : memref<256xf32>
// CHECK-DAG:    csl.var @b_buf : memref<256xf32>
// CHECK-DAG:    csl.var @c_buf : memref<256xf32>
// CHECK:        csl.func @compute {
// CHECK:          scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:            %{{.*}} = memref.load %a_buf[%{{.*}}] : memref<256xf32>
// CHECK:            %{{.*}} = memref.load %b_buf[%{{.*}}] : memref<256xf32>
// CHECK:            %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK:            memref.store %{{.*}}, %c_buf[%{{.*}}] : memref<256xf32>
// CHECK:          }
// CHECK:          csl.return
// CHECK:        }
// CHECK-DAG:    csl.export_symbol @a_buf alias("a")
// CHECK-DAG:    csl.export_symbol @b_buf alias("b")
// CHECK-DAG:    csl.export_symbol @c_buf alias("c")
// CHECK-DAG:    csl.export_symbol @compute
// CHECK:     } {source_file = "vecadd_pe.csl"} : !csl.kernel
// CHECK:     %[[R:.*]] = csl.code_region routes() colors() {
// CHECK:     }{width = 1 : i64, height = 1 : i64} : !csl.code_region
// CHECK:     csl.place %[[R]] %[[K]] {x = 0 : i64, y = 0 : i64}
// CHECK:   }
// CHECK-DAG: csl.export_name "a" : memref<256xf32>{direction = "in"}
// CHECK-DAG: csl.export_name "b" : memref<256xf32>{direction = "in"}
// CHECK-DAG: csl.export_name "c" : memref<256xf32>{direction = "out"}
// CHECK-DAG: csl.export_name "compute" : () -> ()

module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
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
