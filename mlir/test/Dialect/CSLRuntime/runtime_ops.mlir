// RUN: air-opt %s | FileCheck %s

// Test parsing and printing of CSL Runtime operations.

func.func @test_runtime_ops() {
  %layout = csl_rt.create_layout : !csl_rt.layout
  %code_region = csl_rt.create_code_region %layout "pe.csl", "kernel", 16 : index, 16 : index : !csl_rt.layout -> !csl_rt.code_region
  %placed = csl_rt.place %code_region at(0 : index, 0 : index) : !csl_rt.code_region -> !csl_rt.code_region
  %param_set = csl_rt.set_param_all %placed "width" = 16 : !csl_rt.code_region -> !csl_rt.code_region
  %exported = csl_rt.export_name %layout "result", "f32" : !csl_rt.layout -> !csl_rt.layout
  %artifacts = csl_rt.compile %exported : !csl_rt.layout -> !csl_rt.compile_artifacts

  // CHECK: %[[ARTIFACTS:.*]] = csl_rt.compile

  %runtime = csl_rt.runtime_create %artifacts : !csl_rt.compile_artifacts -> !csl_rt.runtime
  // CHECK: %[[RUNTIME:.*]] = csl_rt.runtime_create %[[ARTIFACTS]] : !csl_rt.compile_artifacts -> !csl_rt.runtime

  %loaded = csl_rt.load %runtime : !csl_rt.runtime -> !csl_rt.runtime
  // CHECK: %[[LOADED:.*]] = csl_rt.load %[[RUNTIME]] : !csl_rt.runtime -> !csl_rt.runtime

  %run = csl_rt.run %loaded : !csl_rt.runtime -> !csl_rt.runtime
  // CHECK: %[[RUN:.*]] = csl_rt.run %[[LOADED]] : !csl_rt.runtime -> !csl_rt.runtime

  %id = csl_rt.get_id %run "result" : !csl_rt.runtime -> i32
  // CHECK: %[[ID:.*]] = csl_rt.get_id %[[RUN]] "result" : !csl_rt.runtime -> i32

  %h2d = csl_rt.memcpy_h2d %run 0 "src_data" at(0, 0) with_size(16, 16) elem_per_pe 256 : !csl_rt.runtime -> !csl_rt.runtime
  // CHECK: %[[H2D:.*]] = csl_rt.memcpy_h2d %[[RUN]] 0 "src_data" at(0, 0) with_size(16, 16) elem_per_pe 256 : !csl_rt.runtime -> !csl_rt.runtime

  %d2h = csl_rt.memcpy_d2h %h2d "dest_data" from(0, 0) with_size(16, 16) elem_per_pe 256 {src_id = 1 : i32} : !csl_rt.runtime -> !csl_rt.runtime
  // CHECK: %[[D2H:.*]] = csl_rt.memcpy_d2h %[[H2D]] "dest_data" from(0, 0) with_size(16, 16) elem_per_pe 256 {src_id = 1 : i32} : !csl_rt.runtime -> !csl_rt.runtime

  %launched = csl_rt.launch %d2h "compute" : (!csl_rt.runtime) -> !csl_rt.runtime
  // CHECK: %[[LAUNCHED:.*]] = csl_rt.launch %[[D2H]] "compute" : (!csl_rt.runtime) -> !csl_rt.runtime

  csl_rt.stop %launched : !csl_rt.runtime
  // CHECK: csl_rt.stop %[[LAUNCHED]] : !csl_rt.runtime

  return
}
