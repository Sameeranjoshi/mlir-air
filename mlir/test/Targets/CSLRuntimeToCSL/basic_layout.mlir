//===- basic_layout.mlir - CSL Runtime to CSL text translation test ------===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Smoke test: --emit-csl-rt translates CSL Runtime ops without error and
// emits the expected summary comment to stdout (the SdkLayout-based
// build_layout path has been replaced by HostEmitter / LayoutEmitter).
//
// RUN: air-translate --emit-csl-rt %s | FileCheck %s
//
// CHECK: Generated layout.csl, pe_program.csl, and run.py to
//
//===----------------------------------------------------------------------===//

func.func @test_csl_rt_to_py() {
  %layout = csl_rt.create_layout : !csl_rt.layout

  %code_region = csl_rt.create_code_region %layout "pe.csl", "kernel", 8 : index, 8 : index : !csl_rt.layout -> !csl_rt.code_region

  %placed = csl_rt.place %code_region at(0 : index, 0 : index) : !csl_rt.code_region -> !csl_rt.code_region

  %param_set = csl_rt.set_param_all %placed "width" = 8 : !csl_rt.code_region -> !csl_rt.code_region

  %artifacts = csl_rt.compile %layout : !csl_rt.layout -> !csl_rt.compile_artifacts

  return
}
