//===- CSLAutoVectorize.cpp -------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// -csl-auto-vectorize pass: rewrites scalar scf.for loops over memref buffers
// inside csl.func bodies into csl.get_mem_dsd + csl.builtin_call form where
// the body matches a known DSD idiom.
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/CSLAutoVectorizePass.h"

#include "air/Dialect/CSL/CSLDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "csl-auto-vectorize"

using namespace mlir;

namespace xilinx { namespace air {
  void populateElementwisePatterns(mlir::RewritePatternSet &patterns);
  void populateFmaPattern(mlir::RewritePatternSet &patterns);
  void populateMovePatterns(mlir::RewritePatternSet &patterns);
  void populateScalarBroadcastPatterns(mlir::RewritePatternSet &patterns);
}}

namespace {

struct CSLAutoVectorizePass
    : public PassWrapper<CSLAutoVectorizePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CSLAutoVectorizePass)

  StringRef getArgument() const override { return "csl-auto-vectorize"; }
  StringRef getDescription() const override {
    return "Recognize scalar scf.for loops that match CSL DSD idioms and "
           "rewrite them into csl.get_mem_dsd + csl.builtin_call form";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, memref::MemRefDialect,
                    arith::ArithDialect, xilinx::csl::CSLDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    xilinx::air::populateElementwisePatterns(patterns);
    xilinx::air::populateFmaPattern(patterns);
    xilinx::air::populateMovePatterns(patterns);
    xilinx::air::populateScalarBroadcastPatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::air::createCSLAutoVectorizePass() {
  return std::make_unique<CSLAutoVectorizePass>();
}
