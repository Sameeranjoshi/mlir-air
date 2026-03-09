//===- basic_layout.mlir - CSL Runtime to CSL text translation test ------===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Tests CSL Runtime dialect → Python code generation via mlir-translate --emit-csl-rt
//
// The --emit-csl-rt translation converts CSL Runtime operations to equivalent Python
// code using the Cerebras SDK's SdkLayout API.
//
// RUN: air-translate --emit-csl-rt %s | FileCheck %s
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: def build_layout(platform):
// CHECK: layout = SdkLayout(platform)
// CHECK: code_region = layout.create_code_region("pe.csl", "kernel", 8, 8)
// CHECK: code_region.place(0, 0)
// CHECK: code_region.set_param_all("width", 8)
// CHECK: compile_artifacts = layout.compile(out_prefix='out')
// CHECK: return compile_artifacts

func.func @test_csl_rt_to_py() {
  %layout = csl_rt.create_layout : !csl_rt.layout

  %code_region = csl_rt.create_code_region %layout "pe.csl", "kernel", 8 : index, 8 : index : !csl_rt.layout -> !csl_rt.code_region

  %placed = csl_rt.place %code_region at(0 : index, 0 : index) : !csl_rt.code_region -> !csl_rt.code_region

  %param_set = csl_rt.set_param_all %placed "width" = 8 : !csl_rt.code_region -> !csl_rt.code_region

  %artifacts = csl_rt.compile %layout : !csl_rt.layout -> !csl_rt.compile_artifacts

  return
}
