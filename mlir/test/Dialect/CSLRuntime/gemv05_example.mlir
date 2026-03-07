// RUN: air-opt %s | FileCheck %s

// GEMV-05 end-to-end example: Multiple PEs with memcpy H2D/D2H and launch
// This encodes the structure of https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-05-multiple-pes

func.func @gemv05_example() {
  // Layout setup
  %layout = csl_rt.create_layout : !csl_rt.layout
  // CHECK: %[[LAYOUT:.*]] = csl_rt.create_layout : !csl_rt.layout

  // Create code region (16x16 PEs with kernel from pe_program.csl)
  %code_region = csl_rt.create_code_region %layout "pe_program.csl", "gemv_kernel", 16 : index, 16 : index : !csl_rt.layout -> !csl_rt.code_region
  // CHECK: %[[REGION:.*]] = csl_rt.create_code_region %[[LAYOUT]] "pe_program.csl", "gemv_kernel", 16 : index, 16 : index : !csl_rt.layout -> !csl_rt.code_region

  // Place the code region at (0, 0)
  %placed = csl_rt.place %code_region at(0 : index, 0 : index) : !csl_rt.code_region -> !csl_rt.code_region
  // CHECK: %[[PLACED:.*]] = csl_rt.place %[[REGION]] at(0 : index, 0 : index) : !csl_rt.code_region -> !csl_rt.code_region

  // Set parameters for the kernel
  %param_width = csl_rt.set_param_all %placed "width" = 16 : !csl_rt.code_region -> !csl_rt.code_region
  // CHECK: %[[PARAM_WIDTH:.*]] = csl_rt.set_param_all %[[PLACED]] "width" = 16 : !csl_rt.code_region -> !csl_rt.code_region

  %param_m = csl_rt.set_param_all %param_width "M" = 256 : !csl_rt.code_region -> !csl_rt.code_region
  // CHECK: %[[PARAM_M:.*]] = csl_rt.set_param_all %[[PARAM_WIDTH]] "M" = 256 : !csl_rt.code_region -> !csl_rt.code_region

  %param_n = csl_rt.set_param_all %param_m "N" = 256 : !csl_rt.code_region -> !csl_rt.code_region
  // CHECK: %[[PARAM_N:.*]] = csl_rt.set_param_all %[[PARAM_M]] "N" = 256 : !csl_rt.code_region -> !csl_rt.code_region

  // Export symbols for host access
  %exported_A = csl_rt.export_name %layout "A", "f32" : !csl_rt.layout -> !csl_rt.layout
  // CHECK: %[[EXP_A:.*]] = csl_rt.export_name %[[LAYOUT]] "A", "f32" : !csl_rt.layout -> !csl_rt.layout

  %exported_x = csl_rt.export_name %exported_A "x", "f32" : !csl_rt.layout -> !csl_rt.layout
  // CHECK: %[[EXP_X:.*]] = csl_rt.export_name %[[EXP_A]] "x", "f32" : !csl_rt.layout -> !csl_rt.layout

  %exported_b = csl_rt.export_name %exported_x "b", "f32" : !csl_rt.layout -> !csl_rt.layout
  // CHECK: %[[EXP_B:.*]] = csl_rt.export_name %[[EXP_X]] "b", "f32" : !csl_rt.layout -> !csl_rt.layout

  %exported_y = csl_rt.export_name %exported_b "y", "f32" : !csl_rt.layout -> !csl_rt.layout
  // CHECK: %[[EXP_Y:.*]] = csl_rt.export_name %[[EXP_B]] "y", "f32" : !csl_rt.layout -> !csl_rt.layout

  %exported_compute = csl_rt.export_name %exported_y "compute", "f32" : !csl_rt.layout -> !csl_rt.layout
  // CHECK: %[[EXP_COMPUTE:.*]] = csl_rt.export_name %[[EXP_Y]] "compute", "f32" : !csl_rt.layout -> !csl_rt.layout

  // Compile layout
  %artifacts = csl_rt.compile %exported_compute : !csl_rt.layout -> !csl_rt.compile_artifacts
  // CHECK: %[[ARTIFACTS:.*]] = csl_rt.compile %[[EXP_COMPUTE]] : !csl_rt.layout -> !csl_rt.compile_artifacts

  // Runtime setup
  %runtime = csl_rt.runtime_create %artifacts : !csl_rt.compile_artifacts -> !csl_rt.runtime
  // CHECK: %[[RUNTIME:.*]] = csl_rt.runtime_create %[[ARTIFACTS]] : !csl_rt.compile_artifacts -> !csl_rt.runtime

  // Load kernel
  %loaded = csl_rt.load %runtime : !csl_rt.runtime -> !csl_rt.runtime
  // CHECK: %[[LOADED:.*]] = csl_rt.load %[[RUNTIME]] : !csl_rt.runtime -> !csl_rt.runtime

  // Get symbol IDs
  %id_A = csl_rt.get_id %loaded "A" : !csl_rt.runtime -> i32
  // CHECK: %[[ID_A:.*]] = csl_rt.get_id %[[LOADED]] "A" : !csl_rt.runtime -> i32

  %id_x = csl_rt.get_id %loaded "x" : !csl_rt.runtime -> i32
  %id_b = csl_rt.get_id %loaded "b" : !csl_rt.runtime -> i32
  %id_y = csl_rt.get_id %loaded "y" : !csl_rt.runtime -> i32

  // Host-to-device transfers
  %after_h2d_A = csl_rt.memcpy_h2d %loaded 0 "A_data" at(0, 0) with_size(16, 16) elem_per_pe 256 : !csl_rt.runtime -> !csl_rt.runtime
  // CHECK: %[[AFTER_H2D_A:.*]] = csl_rt.memcpy_h2d %[[LOADED]] 0 "A_data" at(0, 0) with_size(16, 16) elem_per_pe 256 : !csl_rt.runtime -> !csl_rt.runtime

  %after_h2d_x = csl_rt.memcpy_h2d %after_h2d_A 1 "x_data" at(0, 0) with_size(16, 1) elem_per_pe 16 : !csl_rt.runtime -> !csl_rt.runtime
  %after_h2d_b = csl_rt.memcpy_h2d %after_h2d_x 2 "b_data" at(0, 0) with_size(1, 16) elem_per_pe 16 : !csl_rt.runtime -> !csl_rt.runtime

  // Launch compute kernel
  %after_launch = csl_rt.launch %after_h2d_b "compute" : (!csl_rt.runtime) -> !csl_rt.runtime
  // CHECK: %[[AFTER_LAUNCH:.*]] = csl_rt.launch %{{.*}} "compute" : (!csl_rt.runtime) -> !csl_rt.runtime

  // Device-to-host transfers (read results)
  %after_d2h_y = csl_rt.memcpy_d2h %after_launch "y_result" from(0, 0) with_size(1, 16) elem_per_pe 16 {src_id = 3 : i32} : !csl_rt.runtime -> !csl_rt.runtime
  // CHECK: %[[AFTER_D2H_Y:.*]] = csl_rt.memcpy_d2h %[[AFTER_LAUNCH]] "y_result" from(0, 0) with_size(1, 16) elem_per_pe 16 {src_id = 3 : i32} : !csl_rt.runtime -> !csl_rt.runtime

  // Stop runtime
  csl_rt.stop %after_d2h_y : !csl_rt.runtime
  // CHECK: csl_rt.stop %[[AFTER_D2H_Y]] : !csl_rt.runtime

  return
}
