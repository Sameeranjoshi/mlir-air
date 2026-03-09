// RUN: air-opt --csl-to-csl-rt %s | FileCheck %s

func.func @test_csl_to_csl_rt() {
  csl.spatial_placement {
    %k = csl.kernel {
    } {source_file = "pe.csl"} : !csl.kernel
    %r = csl.code_region routes() colors() {
    } {width = 16 : i64, height = 16 : i64} : !csl.code_region
    csl.place %r %k {x = 0 : i64, y = 0 : i64}
  }
  return
}

// CHECK: %[[LAYOUT:.*]] = csl_rt.create_layout : !csl_rt.layout
// CHECK: %[[REGION:.*]] = csl_rt.create_code_region %[[LAYOUT]]
// CHECK: %[[PLACED:.*]] = csl_rt.place %[[REGION]]
// CHECK: csl_rt.compile
