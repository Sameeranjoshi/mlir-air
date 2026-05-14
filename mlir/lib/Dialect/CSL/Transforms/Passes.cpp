//===- Passes.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"

void xilinx::air::registerCSLTransformPasses() {
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> {
        return createCSLInferExportsPass();
      });
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> {
        return createCSLAutoVectorizePass();
      });
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> {
        return createCSLMaterializeDataflowColorsPass();
      });
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> {
        return createCSLAllocateColorIdsPass();
      });
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> {
        return createCSLLowerDataflowRoutingPass();
      });
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> {
        return createCSLLowerDataflowDataPass();
      });
}
