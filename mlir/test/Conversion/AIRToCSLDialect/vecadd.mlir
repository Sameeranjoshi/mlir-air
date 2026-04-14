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
// CHECK:     } {width = 1 : i64, height = 1 : i64} : !csl.code_region
// CHECK:     csl.place %[[R]] %[[K]] {x = 0 : i64, y = 0 : i64}
// CHECK:   }
// CHECK-DAG: csl.export_name "a" : memref<256xf32>{direction = "in"}
// CHECK-DAG: csl.export_name "b" : memref<256xf32>{direction = "in"}
// CHECK-DAG: csl.export_name "c" : memref<256xf32>{direction = "out"}
// CHECK-DAG: csl.export_name "compute" : () -> ()

module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a0=%a, %b0=%b, %c0=%c)
        : memref<256xf32>, memref<256xf32>, memref<256xf32> {
      air.segment @seg0 args(%a1=%a0, %b1=%b0, %c1_=%c0)
          : memref<256xf32>, memref<256xf32>, memref<256xf32> {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile(%htx, %hty) in (%hsx=%c1_0, %hsy=%c1_0)
            args(%a2=%a1, %b2=%b1, %c2=%c1_)
            : memref<256xf32>, memref<256xf32>, memref<256xf32> {
          %c0 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_1 = arith.constant 1 : index
          scf.for %i = %c0 to %c256 step %c1_1 {
            %va = memref.load %a2[%i] : memref<256xf32>
            %vb = memref.load %b2[%i] : memref<256xf32>
            %vc = arith.addf %va, %vb : f32
            memref.store %vc, %c2[%i] : memref<256xf32>
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
