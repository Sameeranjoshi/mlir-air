//===- CSLPipelines.cpp -------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Pipelines/Pipelines.h"
#include "air/Dialect/CSL/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {

/// 4-pass chain that lowers user-facing CSL stream ops to emit-ready form.
void buildDataflowToCSL(OpPassManager &pm) {
  pm.addPass(::xilinx::air::createCSLMaterializeDataflowColorsPass());
  pm.addPass(::xilinx::air::createCSLAllocateColorIdsPass());
  pm.addPass(::xilinx::air::createCSLLowerDataflowRoutingPass());
  pm.addPass(::xilinx::air::createCSLLowerDataflowDataPass());
}

} // namespace

void xilinx::air::registerCSLPipelines() {
  static bool registered = false;
  if (registered)
    return;
  registered = true;

  PassPipelineRegistration<>(
      "csl-dataflow-to-csl",
      "Lower CSL dataflow ops to emit-ready form (4-pass chain).",
      buildDataflowToCSL);

  PassPipelineRegistration<>(
      "csl-pipeline",
      "Full CSL pipeline: infer-exports -> auto-vectorize -> dataflow-to-csl.",
      [](OpPassManager &pm) {
        pm.addPass(::xilinx::air::createCSLInferExportsPass());
        pm.addPass(::xilinx::air::createCSLAutoVectorizePass());
        buildDataflowToCSL(pm);
      });
}
