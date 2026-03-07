// RUN: air-opt %s | FileCheck %s

// Test parsing and printing of CSL Runtime layout operations.

func.func @test_layout_ops() {
  %layout = csl_rt.create_layout : !csl_rt.layout
  // CHECK: %[[LAYOUT:.*]] = csl_rt.create_layout : !csl_rt.layout

  %code_region = csl_rt.create_code_region %layout "pe.csl", "kernel", 16 : index, 16 : index : !csl_rt.layout -> !csl_rt.code_region
  // CHECK: %[[REGION:.*]] = csl_rt.create_code_region %[[LAYOUT]] "pe.csl", "kernel", 16 : index, 16 : index : !csl_rt.layout -> !csl_rt.code_region

  %placed = csl_rt.place %code_region at(0 : index, 0 : index) : !csl_rt.code_region -> !csl_rt.code_region
  // CHECK: %[[PLACED:.*]] = csl_rt.place %[[REGION]] at(0 : index, 0 : index) : !csl_rt.code_region -> !csl_rt.code_region

  %param_set = csl_rt.set_param_all %placed "width" = 16 : !csl_rt.code_region -> !csl_rt.code_region
  // CHECK: %[[PARAM:.*]] = csl_rt.set_param_all %[[PLACED]] "width" = 16 : !csl_rt.code_region -> !csl_rt.code_region

  %exported = csl_rt.export_name %layout "result", "f32" : !csl_rt.layout -> !csl_rt.layout
  // CHECK: %[[EXPORTED:.*]] = csl_rt.export_name %[[LAYOUT]] "result", "f32" : !csl_rt.layout -> !csl_rt.layout

  %artifacts = csl_rt.compile %exported : !csl_rt.layout -> !csl_rt.compile_artifacts
  // CHECK: %[[ARTIFACTS:.*]] = csl_rt.compile %[[EXPORTED]] : !csl_rt.layout -> !csl_rt.compile_artifacts

  return
}
