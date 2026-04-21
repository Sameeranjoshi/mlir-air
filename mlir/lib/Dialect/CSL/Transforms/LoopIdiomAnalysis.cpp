//===- LoopIdiomAnalysis.cpp ------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/LoopIdiomAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
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

  // Rules 7 & 8 — collect loads / stores / arith ops; enforce single store
  // and allowed body-op set (arith.* + memref.load/store only; constants
  // that are index values are index-typed arith.constant which is already
  // dialect=arith).
  for (Operation &inner : op.getBody()->without_terminator()) {
    if (auto ld = dyn_cast<memref::LoadOp>(&inner)) {
      info.loads.push_back(ld);
      continue;
    }
    if (auto st = dyn_cast<memref::StoreOp>(&inner)) {
      info.stores.push_back(st);
      continue;
    }
    // Anything that passed Rule 5 and isn't load/store must be arith.*.
    // Record it for idiom-signature matching by patterns.
    info.bodyOps.push_back(&inner);
  }
  if (info.stores.size() != 1) {
    LLVM_DEBUG(llvm::dbgs() << "reject: body has "
                            << info.stores.size() << " stores (need 1) @"
                            << op.getLoc() << "\n");
    return failure();
  }
  if (info.loads.empty()) {
    // Trivial move from a scalar constant doesn't match any MVP idiom;
    // patterns always need at least one load to build a DSD from.
    LLVM_DEBUG(llvm::dbgs() << "reject: body has zero loads @"
                            << op.getLoc() << "\n");
    return failure();
  }

  // Rule 6 — induction variable uses restricted to memref index positions,
  // either directly or via arith.addi/arith.muli %iv, %const / affine.apply.
  for (Operation *user : info.inductionVar.getUsers()) {
    if (isa<memref::LoadOp, memref::StoreOp>(user)) {
      // Ok — check the IV is used as an index (not as the memref operand).
      for (auto [idx, operand] : llvm::enumerate(user->getOperands())) {
        if (operand != info.inductionVar) continue;
        if (auto ld = dyn_cast<memref::LoadOp>(user)) {
          if (idx == 0) {
            LLVM_DEBUG(llvm::dbgs() << "reject: IV used as memref operand @"
                                    << user->getLoc() << "\n");
            return failure();
          }
        } else if (auto st = dyn_cast<memref::StoreOp>(user)) {
          if (idx <= 1) {
            LLVM_DEBUG(llvm::dbgs() << "reject: IV used as value/memref @"
                                    << user->getLoc() << "\n");
            return failure();
          }
        }
      }
      continue;
    }
    if (isa<arith::AddIOp, arith::MulIOp, arith::SubIOp>(user)) {
      // OK — the other operand must be a constant for the index to be
      // affine in %iv with constant coefficients (enforced in Task 6).
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "reject: IV has non-index user: "
                            << user->getName() << " @" << user->getLoc()
                            << "\n");
    return failure();
  }

  // Rule 9 — collect loop-invariant scalar f32 values consumed by the body.
  // (Loop-invariant memrefs are captured per-access in Task 6.)
  llvm::DenseSet<Value> seen;
  for (Operation *bodyOp : info.bodyOps) {
    for (Value v : bodyOp->getOperands()) {
      if (seen.contains(v)) continue;
      seen.insert(v);
      // Block arg of the scf.for's body = the IV (already handled).
      if (auto bArg = dyn_cast<BlockArgument>(v)) {
        if (bArg == info.inductionVar) continue;
      }
      // Defined inside the loop? Skip.
      if (Operation *def = v.getDefiningOp()) {
        if (op->isAncestor(def)) continue;
      }
      // Loop-invariant scalar of supported type.
      if (v.getType().isF32()) {
        info.loopInvariantScalars.push_back(v);
      }
      // Non-f32 scalar invariants are allowed through analysis (they may
      // still be valid — e.g. a scalar constant that participates in
      // index math outside our IV).  Patterns that need the scalar will
      // re-check its type.
    }
  }

  // Rules 10-12 land in Task 6.
  LLVM_DEBUG(llvm::dbgs() << "reject: access-pattern check not implemented @"
                          << op.getLoc() << "\n");
  return failure();
}

} // namespace air
} // namespace xilinx
