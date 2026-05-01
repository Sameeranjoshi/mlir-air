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
void buildStreamsToCSL(OpPassManager &pm) {
  pm.addPass(::xilinx::air::createCSLMaterializeStreamColorsPass());
  pm.addPass(::xilinx::air::createCSLAllocateColorIdsPass());
  pm.addPass(::xilinx::air::createCSLLowerStreamRoutingPass());
  pm.addPass(::xilinx::air::createCSLLowerStreamDataPass());
}

} // namespace

void xilinx::air::registerCSLPipelines() {
  static bool registered = false;
  if (registered)
    return;
  registered = true;

  PassPipelineRegistration<>(
      "csl-streams-to-csl",
      "Lower CSL stream ops to emit-ready form (4-pass chain).",
      buildStreamsToCSL);

  PassPipelineRegistration<>(
      "csl-pipeline",
      "Full CSL pipeline: infer-exports -> auto-vectorize -> streams-to-csl.",
      [](OpPassManager &pm) {
        pm.addPass(::xilinx::air::createCSLInferExportsPass());
        pm.addPass(::xilinx::air::createCSLAutoVectorizePass());
        buildStreamsToCSL(pm);
      });
}
