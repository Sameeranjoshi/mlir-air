// RUN: air-opt --csl-to-csl-rt %s | FileCheck %s

func.func @test_csl_to_csl_rt() {
  csl.spatial_placement {
    %k = csl.kernel "pe.csl" {
      %f = csl.func @compute() {
        csl.return
      }
    } : !csl.kernel
    %r = csl.code_region shape(16, 16) {
    } : !csl.code_region
    csl.place %r at(0, 0) kernel(%k)
    csl.set_param_all %r "width" = 16 : index
    csl.export_name %r "result" : f32
  }
  return
}

// CHECK: %[[LAYOUT:.*]] = csl_rt.create_layout : () -> !csl_rt.layout
// CHECK: %[[REGION:.*]] = csl_rt.create_code_region %[[LAYOUT]]
// CHECK: %[[PLACED:.*]] = csl_rt.place %[[REGION]]
// CHECK: %[[PARAM:.*]] = csl_rt.set_param_all %[[PLACED]]
// CHECK: %[[EXPORTED:.*]] = csl_rt.export_name %[[LAYOUT]]
// CHECK: %[[ARTIFACTS:.*]] = csl_rt.compile %[[EXPORTED]]
