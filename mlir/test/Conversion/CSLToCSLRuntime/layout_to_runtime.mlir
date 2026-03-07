// RUN: air-opt --csl-to-csl-rt %s | FileCheck %s

func.func @test_csl_to_csl_rt() {
  csl.spatial_placement {
    %k = csl.kernel "pe.csl" {
      csl.func @compute() : () -> () {
        csl.return
      }
    } : !csl.kernel
    %r = csl.code_region routes() colors() shape(16, 16) {
    } : !csl.code_region
    csl.place %r at(0, 0) kernel(%k)
  }
  return
}

// CHECK: %[[LAYOUT:.*]] = csl_rt.create_layout : !csl_rt.layout
// CHECK: %[[REGION:.*]] = csl_rt.create_code_region %[[LAYOUT]]
// CHECK: %[[PLACED:.*]] = csl_rt.place %[[REGION]]
// CHECK: csl_rt.compile
