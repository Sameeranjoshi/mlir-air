//===- csl_runtime.mlir - CSL Runtime dialect tests ----------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Tests for the CSL Runtime (csl_rt) dialect - high-level layout programming
// model that maps to Cerebras SdkLayout Python API.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// ============================================================================
// Test 1: Create layout and code regions
// ============================================================================

// CHECK-LABEL: func.func @test_layout_creation
// CHECK:   %[[LAYOUT:.*]] = csl_rt.create_layout : !csl_rt.layout
// CHECK:   %[[REGION:.*]] = csl_rt.create_code_region %[[LAYOUT]] "pe.csl", "kernel", {{.*}} : index, {{.*}} : index : !csl_rt.layout -> !csl_rt.code_region
func.func @test_layout_creation() {
  %layout = csl_rt.create_layout : !csl_rt.layout
  %region = csl_rt.create_code_region %layout "pe.csl", "kernel", 16 : index, 16 : index : !csl_rt.layout -> !csl_rt.code_region
  return
}

// ============================================================================
// Test 2: Place code region at coordinates
// ============================================================================

// CHECK-LABEL: func.func @test_place
// CHECK:   %[[LAYOUT:.*]] = csl_rt.create_layout : !csl_rt.layout
// CHECK:   %[[REGION:.*]] = csl_rt.create_code_region %[[LAYOUT]]
// CHECK:   %[[PLACED:.*]] = csl_rt.place %[[REGION]] at(0 : index, 0 : index) : !csl_rt.code_region -> !csl_rt.code_region
func.func @test_place() {
  %layout = csl_rt.create_layout : !csl_rt.layout
  %region = csl_rt.create_code_region %layout "pe.csl", "compute", 4 : index, 4 : index : !csl_rt.layout -> !csl_rt.code_region
  %placed = csl_rt.place %region at(0 : index, 0 : index) : !csl_rt.code_region -> !csl_rt.code_region
  return
}

// ============================================================================
// Test 3: Set parameters on code regions
// ============================================================================

// CHECK-LABEL: func.func @test_set_param
// CHECK:   %[[LAYOUT:.*]] = csl_rt.create_layout : !csl_rt.layout
// CHECK:   %[[REGION:.*]] = csl_rt.create_code_region %[[LAYOUT]]
// CHECK:   %[[PARAM:.*]] = csl_rt.set_param_all %[[REGION]] "width" = {{[0-9]+}} : !csl_rt.code_region -> !csl_rt.code_region
func.func @test_set_param() {
  %layout = csl_rt.create_layout : !csl_rt.layout
  %region = csl_rt.create_code_region %layout "pe.csl", "kernel", 8 : index, 8 : index : !csl_rt.layout -> !csl_rt.code_region
  %param_set = csl_rt.set_param_all %region "width" = 256 : !csl_rt.code_region -> !csl_rt.code_region
  return
}

// ============================================================================
// Test 4: Export names (map to host-accessible variables)
// ============================================================================

// CHECK-LABEL: func.func @test_export_name
// CHECK:   %[[LAYOUT:.*]] = csl_rt.create_layout : !csl_rt.layout
// CHECK:   %[[EXPORT:.*]] = csl_rt.export_name %[[LAYOUT]] "result", "f32" : !csl_rt.layout -> !csl_rt.layout
func.func @test_export_name() {
  %layout = csl_rt.create_layout : !csl_rt.layout
  %exported = csl_rt.export_name %layout "result", "f32" : !csl_rt.layout -> !csl_rt.layout
  return
}

// ============================================================================
// Test 5: Complete workflow - layout construction chain
// ============================================================================

// CHECK-LABEL: func.func @test_complete_workflow
func.func @test_complete_workflow() {
  %layout = csl_rt.create_layout : !csl_rt.layout
  
  // Create multiple code regions
  %region1 = csl_rt.create_code_region %layout "sender.csl", "send", 4 : index, 4 : index : !csl_rt.layout -> !csl_rt.code_region
  %placed1 = csl_rt.place %region1 at(0 : index, 0 : index) : !csl_rt.code_region -> !csl_rt.code_region
  %param1 = csl_rt.set_param_all %placed1 "color_id" = 0 : !csl_rt.code_region -> !csl_rt.code_region
  
  // Export symbols
  %exported1 = csl_rt.export_name %layout "output_buffer", "f32" : !csl_rt.layout -> !csl_rt.layout
  
  // Compile
  %artifacts = csl_rt.compile %exported1 : !csl_rt.layout -> !csl_rt.compile_artifacts
  
  return
}
