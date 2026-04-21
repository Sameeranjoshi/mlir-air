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

/// Match index expression `expr` against (coeff * %iv + k) with constant
/// integer `coeff`, `k`.  Returns {coeff, k} on match, failure otherwise.
static FailureOr<std::pair<int64_t, int64_t>>
matchAffineIndexInIV(Value expr, Value iv) {
  // Base case: the IV itself.
  if (expr == iv) return std::make_pair(int64_t(1), int64_t(0));

  // A constant — expressible as (0 * iv + c).
  if (auto c = getConstantIntValue(expr))
    return std::make_pair(int64_t(0), *c);

  Operation *def = expr.getDefiningOp();
  if (!def) return failure();

  // arith.addi a, b : i_or_index  →  coeff(a) + coeff(b), k(a) + k(b)
  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    auto l = matchAffineIndexInIV(add.getLhs(), iv);
    auto r = matchAffineIndexInIV(add.getRhs(), iv);
    if (failed(l) || failed(r)) return failure();
    return std::make_pair(l->first + r->first, l->second + r->second);
  }

  // arith.subi a, b → coeff(a) - coeff(b), k(a) - k(b)
  if (auto sub = dyn_cast<arith::SubIOp>(def)) {
    auto l = matchAffineIndexInIV(sub.getLhs(), iv);
    auto r = matchAffineIndexInIV(sub.getRhs(), iv);
    if (failed(l) || failed(r)) return failure();
    return std::make_pair(l->first - r->first, l->second - r->second);
  }

  // arith.muli iv, c  or  arith.muli c, iv  →  (coeff*c, k*c) IF the OTHER
  // side is a compile-time constant (otherwise reject — we can't multiply
  // two affine expressions and stay affine).
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    auto l = matchAffineIndexInIV(mul.getLhs(), iv);
    auto r = matchAffineIndexInIV(mul.getRhs(), iv);
    if (failed(l) || failed(r)) return failure();
    // One side must have coeff == 0 (be a pure constant).
    if (l->first == 0)
      return std::make_pair(l->second * r->first, l->second * r->second);
    if (r->first == 0)
      return std::make_pair(l->first * r->second, l->second * r->second);
    return failure();
  }

  return failure();
}

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

  // Rules 10, 11, 12 — per-access analysis.
  auto analyzeAccess = [&](Value memRef, ValueRange indices,
                           Operation *accessOp) -> LogicalResult {
    auto ty = dyn_cast<MemRefType>(memRef.getType());
    if (!ty) {
      LLVM_DEBUG(llvm::dbgs() << "reject: non-memref access operand @"
                              << accessOp->getLoc() << "\n");
      return failure();
    }
    if ((unsigned)ty.getRank() != indices.size()) {
      LLVM_DEBUG(llvm::dbgs() << "reject: rank/indices mismatch @"
                              << accessOp->getLoc() << "\n");
      return failure();
    }
    if (ty.getRank() != 1) {
      // Rank-2 handling arrives in Task 7.
      LLVM_DEBUG(llvm::dbgs() << "reject: non-rank-1 access (rank "
                              << ty.getRank() << ") @"
                              << accessOp->getLoc() << "\n");
      return failure();
    }
    if (!ty.hasStaticShape()) {
      LLVM_DEBUG(llvm::dbgs() << "reject: dynamic memref shape @"
                              << accessOp->getLoc() << "\n");
      return failure();
    }

    // Rule 10 — index is affine in IV.
    auto aff = matchAffineIndexInIV(indices[0], info.inductionVar);
    if (failed(aff)) {
      LLVM_DEBUG(llvm::dbgs() << "reject: non-affine index @"
                              << accessOp->getLoc() << "\n");
      return failure();
    }
    int64_t coeff = aff->first;
    int64_t k = aff->second;

    // Rule 11 — access-in-bounds.
    // Iteration range is [lb, ub).
    int64_t ax0 = coeff * info.lb + k;
    int64_t ax1 = coeff * (info.ub - 1) + k;
    int64_t amin = std::min(ax0, ax1);
    int64_t amax = std::max(ax0, ax1);
    int64_t bufExtent = ty.getShape()[0];
    if (amin < 0 || amax >= bufExtent) {
      LLVM_DEBUG(llvm::dbgs() << "reject: access OOB [" << amin << ","
                              << amax << "] on buffer extent " << bufExtent
                              << " @" << accessOp->getLoc() << "\n");
      return failure();
    }

    // Rule 12 — DSD field widths (mem1d: stride i8, offset i16).
    if (coeff < kMinMem1dStride || coeff > kMaxMem1dStride) {
      LLVM_DEBUG(llvm::dbgs() << "reject: stride " << coeff
                              << " outside mem1d i8 range @"
                              << accessOp->getLoc() << "\n");
      return failure();
    }
    if (amin < kMinDsdOffset || amin > kMaxDsdOffset) {
      LLVM_DEBUG(llvm::dbgs() << "reject: effective offset " << amin
                              << " outside i16 range @"
                              << accessOp->getLoc() << "\n");
      return failure();
    }

    DsdAccessPattern ap;
    ap.buffer = memRef;
    ap.strides = {coeff};
    ap.offsets = {amin};                 // effective post-clip start offset
    ap.rank = 1;
    info.accesses.push_back(ap);
    return success();
  };

  for (auto &ld : info.loads)
    if (failed(analyzeAccess(ld.getMemRef(), ld.getIndices(), ld)))
      return failure();
  for (auto &st : info.stores)
    if (failed(analyzeAccess(st.getMemRef(), st.getIndices(), st)))
      return failure();

  LLVM_DEBUG(llvm::dbgs() << "accept: LoopIdiom extent=" << info.extent
                          << " loads=" << info.loads.size()
                          << " stores=" << info.stores.size() << " @"
                          << op.getLoc() << "\n");
  return info;
}

} // namespace air
} // namespace xilinx
