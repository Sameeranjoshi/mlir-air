//===- LoopIdiomAnalysis.cpp ------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/LoopIdiomAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "csl-auto-vectorize"

using namespace mlir;

namespace xilinx {
namespace air {

FailureOr<LoopIdiom> analyzeForLoop(scf::ForOp op) {
  LoopIdiom info;

  // Rule 1 — constant bounds and step.
  std::optional<int64_t> lb = getConstantIntValue(op.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(op.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(op.getStep());
  if (!lb || !ub || !step) {
    LLVM_DEBUG(llvm::dbgs() << "reject: non-constant bounds or step @"
                            << op.getLoc() << "\n");
    return failure();
  }
  info.lb = *lb;
  info.ub = *ub;
  info.step = *step;
  info.extent = info.ub - info.lb;
  info.inductionVar = op.getInductionVar();

  // Rule 2 — step == 1.
  if (info.step != 1) {
    LLVM_DEBUG(llvm::dbgs() << "reject: step != 1 @" << op.getLoc() << "\n");
    return failure();
  }

  // Rule 3 — extent in DSD u16 range.
  if (info.extent <= 0 || info.extent > kMaxDsdExtent) {
    LLVM_DEBUG(llvm::dbgs() << "reject: extent " << info.extent
                            << " outside [1," << kMaxDsdExtent << "] @"
                            << op.getLoc() << "\n");
    return failure();
  }

  // Rule 4 — single-block body, scf.yield with no yielded values.
  if (!op.getRegion().hasOneBlock()) {
    LLVM_DEBUG(llvm::dbgs() << "reject: body has multiple blocks @"
                            << op.getLoc() << "\n");
    return failure();
  }
  if (op.getNumRegionIterArgs() != 0) {
    LLVM_DEBUG(llvm::dbgs() << "reject: loop has iter_args (reduction?) @"
                            << op.getLoc() << "\n");
    return failure();
  }

  // Rule 5 — no nested control flow or calls in body.  (Rank-2 will relax
  // this in Task 7; for now any nested op of these kinds rejects.)
  for (Operation &inner : op.getBody()->without_terminator()) {
    if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(&inner)) {
      LLVM_DEBUG(llvm::dbgs() << "reject: nested control flow ("
                              << inner.getName() << ") @"
                              << inner.getLoc() << "\n");
      return failure();
    }
    if (inner.hasTrait<OpTrait::HasRecursiveMemoryEffects>() ||
        inner.mightHaveTrait<OpTrait::IsTerminator>())
      continue;
    if (!isa<arith::ArithDialect>(inner.getDialect()) &&
        !isa<memref::MemRefDialect>(inner.getDialect())) {
      LLVM_DEBUG(llvm::dbgs() << "reject: disallowed dialect in body ("
                              << inner.getName() << ") @"
                              << inner.getLoc() << "\n");
      return failure();
    }
  }

  // Subsequent rules 6-12 land in later tasks.
  LLVM_DEBUG(llvm::dbgs() << "reject: rules 6-12 not yet implemented @"
                          << op.getLoc() << "\n");
  return failure();
}

} // namespace air
} // namespace xilinx
