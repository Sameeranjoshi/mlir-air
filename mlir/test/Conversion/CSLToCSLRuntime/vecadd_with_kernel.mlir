// RUN: air-opt --csl-to-csl-rt %s | FileCheck %s

// Verify that csl-to-csl-rt synthesizes the host-side runtime data-movement
// sequence from csl.export_name ops, leaving csl.spatial_placement and all
// csl.export_name ops in place.

// CHECK-LABEL: func.func @vecadd
// CHECK:       csl.spatial_placement
// CHECK:       csl.export_name "a" : memref<256xf32>{direction = "in"}
// CHECK:       csl.export_name "b" : memref<256xf32>{direction = "in"}
// CHECK:       csl.export_name "c" : memref<256xf32>{direction = "out"}
// CHECK:       csl.export_name "compute" : () -> ()
// CHECK:       %[[L:.*]] = csl_rt.create_layout : !csl_rt.layout
// CHECK:       %[[ART:.*]] = csl_rt.compile %[[L]] : !csl_rt.layout -> !csl_rt.compile_artifacts
// CHECK:       %[[RT:.*]] = csl_rt.runtime_create %[[ART]] : !csl_rt.compile_artifacts -> !csl_rt.runtime
// CHECK:       %[[H1:.*]] = csl_rt.memcpy_h2d %[[RT]] {{.*}} "a"
// CHECK:       %[[H2:.*]] = csl_rt.memcpy_h2d %[[H1]] {{.*}} "b"
// CHECK:       %[[LC:.*]] = csl_rt.launch %[[H2]] "compute"
// CHECK:       %{{.*}} = csl_rt.memcpy_d2h %[[LC]] "c"

module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
    csl.spatial_placement {
      %k = csl.kernel {
        %a_buf = csl.var @a_buf : memref<256xf32>
        %b_buf = csl.var @b_buf : memref<256xf32>
        %c_buf = csl.var @c_buf : memref<256xf32>
        csl.func @compute {
          csl.return
        }
        csl.export_symbol @a_buf alias("a")
        csl.export_symbol @b_buf alias("b")
        csl.export_symbol @c_buf alias("c")
        csl.export_symbol @compute
      } {source_file = "vecadd_pe.csl"} : !csl.kernel
      %r = csl.code_region routes() colors() {
      }{width = 1 : i64, height = 1 : i64} : !csl.code_region
      csl.place %r %k {x = 0 : i64, y = 0 : i64}
    }
    csl.export_name "a" : memref<256xf32>{direction = "in"}
    csl.export_name "b" : memref<256xf32>{direction = "in"}
    csl.export_name "c" : memref<256xf32>{direction = "out"}
    csl.export_name "compute" : () -> ()
    return
  }
}
