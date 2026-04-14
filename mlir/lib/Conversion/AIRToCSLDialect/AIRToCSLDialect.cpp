//===- AIRToCSLDialect.cpp - AIR → csl.* lowering pass ----------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#include "air/Conversion/AIRToCSLDialectPass.h"
#include "air/Dialect/CSL/CSLDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "air-to-csl-dialect"

using namespace mlir;

namespace {

class AIRToCSLDialectPass
    : public PassWrapper<AIRToCSLDialectPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AIRToCSLDialectPass)

  StringRef getArgument() const final { return "air-to-csl-dialect"; }
  StringRef getDescription() const final {
    return "Lower AIR dialect (1x1 herds only) to CSL dialect ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<xilinx::csl::CSLDialect>();
  }

  void runOnOperation() override {
    // Phase 2 task 2.3 will add the real lowering here. Empty for now.
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::air::createAIRToCSLDialectPass() {
  return std::make_unique<AIRToCSLDialectPass>();
}
