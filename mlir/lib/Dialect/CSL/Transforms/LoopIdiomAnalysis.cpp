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

/// Like matchAffineIndexInIV but handles a pair of IVs; returns
/// (coeff_primary, coeff_secondary, k).  Either coeff may be 0.
static FailureOr<std::tuple<int64_t, int64_t, int64_t>>
matchAffineIndexInIVPair(Value expr, Value ivA, Value ivB) {
  auto a = matchAffineIndexInIV(expr, ivA);
  if (succeeded(a) && ivB == Value{})
    return std::make_tuple(a->first, int64_t(0), a->second);
  // Try treating expr as affine in ivB with ivA contributing via a constant.
  // Simpler: call matchAffineIndexInIV with one IV at a time after splitting.
  // For MVP: require index expression to decompose as (cA*ivA + cB*ivB + k)
  // with additive structure.  We handle it recursively:
  if (expr == ivA) return std::make_tuple(int64_t(1), int64_t(0), int64_t(0));
  if (expr == ivB) return std::make_tuple(int64_t(0), int64_t(1), int64_t(0));
  if (auto c = getConstantIntValue(expr))
    return std::make_tuple(int64_t(0), int64_t(0), *c);

  Operation *def = expr.getDefiningOp();
  if (!def) return failure();
  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    auto l = matchAffineIndexInIVPair(add.getLhs(), ivA, ivB);
    auto r = matchAffineIndexInIVPair(add.getRhs(), ivA, ivB);
    if (failed(l) || failed(r)) return failure();
    return std::make_tuple(std::get<0>(*l) + std::get<0>(*r),
                           std::get<1>(*l) + std::get<1>(*r),
                           std::get<2>(*l) + std::get<2>(*r));
  }
  if (auto sub = dyn_cast<arith::SubIOp>(def)) {
    auto l = matchAffineIndexInIVPair(sub.getLhs(), ivA, ivB);
    auto r = matchAffineIndexInIVPair(sub.getRhs(), ivA, ivB);
    if (failed(l) || failed(r)) return failure();
    return std::make_tuple(std::get<0>(*l) - std::get<0>(*r),
                           std::get<1>(*l) - std::get<1>(*r),
                           std::get<2>(*l) - std::get<2>(*r));
  }
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    auto l = matchAffineIndexInIVPair(mul.getLhs(), ivA, ivB);
    auto r = matchAffineIndexInIVPair(mul.getRhs(), ivA, ivB);
    if (failed(l) || failed(r)) return failure();
    bool lIsConst = std::get<0>(*l) == 0 && std::get<1>(*l) == 0;
    bool rIsConst = std::get<0>(*r) == 0 && std::get<1>(*r) == 0;
    if (lIsConst)
      return std::make_tuple(std::get<2>(*l) * std::get<0>(*r),
                             std::get<2>(*l) * std::get<1>(*r),
                             std::get<2>(*l) * std::get<2>(*r));
    if (rIsConst)
      return std::make_tuple(std::get<0>(*l) * std::get<2>(*r),
                             std::get<1>(*l) * std::get<2>(*r),
                             std::get<2>(*l) * std::get<2>(*r));
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

  // Rule 5 — no nested control flow or calls in body. Exception: exactly
  // one nested scf.for whose body is the rank-1 shape (perfect 2-deep nest).
  scf::ForOp innerFor;
  {
    unsigned nonTermCount = 0;
    for (Operation &inner : op.getBody()->without_terminator()) {
      nonTermCount++;
      if (auto nested = dyn_cast<scf::ForOp>(&inner)) {
        if (innerFor) {
          LLVM_DEBUG(llvm::dbgs() << "reject: multiple nested loops @"
                                  << inner.getLoc() << "\n");
          return failure();
        }
        innerFor = nested;
      }
    }
    if (innerFor && nonTermCount != 1) {
      LLVM_DEBUG(llvm::dbgs() << "reject: non-perfect rank-2 nest @"
                              << innerFor.getLoc() << "\n");
      return failure();
    }
  }

  if (innerFor) {
    // Outer-loop body is the one inner scf.for — no body ops to classify
    // here.  Perform shape + extent checks on the inner loop, then treat
    // the inner loop's body as the "real" body for rules 6-12.
    std::optional<int64_t> ilb = getConstantIntValue(innerFor.getLowerBound());
    std::optional<int64_t> iub = getConstantIntValue(innerFor.getUpperBound());
    std::optional<int64_t> istep = getConstantIntValue(innerFor.getStep());
    if (!ilb || !iub || !istep || *istep != 1) {
      LLVM_DEBUG(llvm::dbgs() << "reject: inner loop non-constant or step!=1 @"
                              << innerFor.getLoc() << "\n");
      return failure();
    }
    info.isRank2 = true;
    info.innerLb = *ilb;
    info.innerUb = *iub;
    info.innerStep = 1;
    info.innerExtent = info.innerUb - info.innerLb;
    info.innerInductionVar = innerFor.getInductionVar();
    if (info.innerExtent <= 0 || info.innerExtent > kMaxDsdExtent ||
        info.innerLb < 0) {
      LLVM_DEBUG(llvm::dbgs() << "reject: inner extent out of range @"
                              << innerFor.getLoc() << "\n");
      return failure();
    }
    if (!innerFor.getRegion().hasOneBlock() ||
        innerFor.getNumRegionIterArgs() != 0) {
      LLVM_DEBUG(llvm::dbgs() << "reject: inner loop iter_args/non-single-block @"
                              << innerFor.getLoc() << "\n");
      return failure();
    }
  } else {
    // No inner loop — check that the flat body has no disallowed ops.
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
  }

  // Loops from here on — iterate over innerFor's body if rank-2, else
  // the outer body.  Bind a helper.
  Block *bodyBlock = innerFor ? innerFor.getBody() : op.getBody();
  Value primaryIV = info.inductionVar;    // outer IV in rank-2
  Value secondaryIV = innerFor ? innerFor.getInductionVar() : Value{};

  // Rules 7 & 8 — collect loads / stores / arith ops; enforce single store
  // and allowed body-op set (arith.* + memref.load/store only; constants
  // that are index values are index-typed arith.constant which is already
  // dialect=arith).
  //
  // Helper: returns true when every use of `v` is as an index operand of
  // a memref.load or memref.store (i.e. the value flows only into index
  // positions, never into value/data positions).  This lets us exclude
  // index-arithmetic arith ops (e.g. `arith.subi %i, %c1` for stencils)
  // from `bodyOps` so that patterns only see "computation" arith ops.
  auto usedOnlyAsIndex = [&](Value v) -> bool {
    for (Operation *user : v.getUsers()) {
      if (auto ld = dyn_cast<memref::LoadOp>(user)) {
        // operand 0 is the memref; operands 1.. are indices.
        bool isIndex = false;
        for (unsigned i = 1, e = ld->getNumOperands(); i < e; ++i)
          if (ld->getOperand(i) == v) isIndex = true;
        if (!isIndex) return false;
        continue;
      }
      if (auto st = dyn_cast<memref::StoreOp>(user)) {
        // operand 0 = value, operand 1 = memref, operands 2.. = indices.
        bool isIndex = false;
        for (unsigned i = 2, e = st->getNumOperands(); i < e; ++i)
          if (st->getOperand(i) == v) isIndex = true;
        if (!isIndex) return false;
        continue;
      }
      // Used by another arith op — recurse: if that op's result is also
      // index-only, it's fine; otherwise this value is a computation value.
      return false;
    }
    return true;
  };
  for (Operation &inner : bodyBlock->without_terminator()) {
    if (auto ld = dyn_cast<memref::LoadOp>(&inner)) {
      info.loads.push_back(ld);
      continue;
    }
    if (auto st = dyn_cast<memref::StoreOp>(&inner)) {
      info.stores.push_back(st);
      continue;
    }
    // Skip arith ops whose sole purpose is to compute a memref index
    // (e.g. `arith.subi %i, %c1` in a stencil).  Such ops are visible to
    // matchAffineIndexInIV() through the load's operand chain; they must not
    // appear in `bodyOps` or patterns will count them as computation ops.
    if (inner.getNumResults() == 1 &&
        inner.getResult(0).getType().isIndex() &&
        usedOnlyAsIndex(inner.getResult(0)))
      continue;
    // Anything else that passed Rule 5 must be arith.*; record it.
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
  auto checkIVUses = [&](Value iv, const char *which) -> LogicalResult {
    for (Operation *user : iv.getUsers()) {
      if (isa<memref::LoadOp, memref::StoreOp>(user)) {
        for (auto [idx, operand] : llvm::enumerate(user->getOperands())) {
          if (operand != iv) continue;
          if (auto ld = dyn_cast<memref::LoadOp>(user)) {
            if (idx == 0) {
              LLVM_DEBUG(llvm::dbgs() << "reject: " << which
                                      << " IV used as memref operand @"
                                      << user->getLoc() << "\n");
              return failure();
            }
          } else if (auto st = dyn_cast<memref::StoreOp>(user)) {
            if (idx <= 1) {
              LLVM_DEBUG(llvm::dbgs() << "reject: " << which
                                      << " IV used as value/memref @"
                                      << user->getLoc() << "\n");
              return failure();
            }
          }
        }
        continue;
      }
      if (isa<arith::AddIOp, arith::MulIOp, arith::SubIOp>(user)) continue;
      // Outer IV used by inner scf.for (as nothing directly — the inner
      // uses ITS OWN iv).  So non-memref, non-arith users should be empty.
      if (isa<scf::ForOp>(user)) continue;
      LLVM_DEBUG(llvm::dbgs() << "reject: " << which
                              << " IV has non-index user: "
                              << user->getName() << "\n");
      return failure();
    }
    return success();
  };
  if (failed(checkIVUses(primaryIV, "outer"))) return failure();
  if (info.isRank2 && failed(checkIVUses(secondaryIV, "inner")))
    return failure();

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
    if (!ty.hasStaticShape()) {
      LLVM_DEBUG(llvm::dbgs() << "reject: dynamic memref shape @"
                              << accessOp->getLoc() << "\n");
      return failure();
    }

    if (ty.getRank() == 1) {
      if (info.isRank2) {
        // Outer rank-2 nest over rank-1 accesses: index must be affine only
        // in the inner IV, outer coefficient must be zero.
        auto aff = matchAffineIndexInIVPair(indices[0], primaryIV, secondaryIV);
        if (failed(aff) || std::get<0>(*aff) != 0) {
          LLVM_DEBUG(llvm::dbgs() << "reject: inner-only-indexed rank-1 access"
                                     " but outer-coeff non-zero @"
                                  << accessOp->getLoc() << "\n");
          return failure();
        }
        int64_t coeff = std::get<1>(*aff);
        int64_t kk = std::get<2>(*aff);
        // Range over [innerLb, innerUb)
        int64_t x0 = coeff * info.innerLb + kk;
        int64_t x1 = coeff * (info.innerUb - 1) + kk;
        int64_t amin = std::min(x0, x1);
        int64_t amax = std::max(x0, x1);
        int64_t bufExtent = ty.getShape()[0];
        if (amin < 0 || amax >= bufExtent) {
          LLVM_DEBUG(llvm::dbgs() << "reject: rank-1 (inside rank-2) OOB @"
                                  << accessOp->getLoc() << "\n");
          return failure();
        }
        // Rule 13 — store target must not be stride-0 (that would be a
        // reduction, deferred to a separate pass — spec §9.2).
        if (isa<memref::StoreOp>(accessOp) && coeff == 0) {
          LLVM_DEBUG(llvm::dbgs()
                     << "reject: store target has stride 0 "
                        "(reduction pattern) @" << accessOp->getLoc() << "\n");
          return failure();
        }
        DsdAccessPattern ap{memRef, {coeff}, {amin}, 1};
        info.accesses.push_back(ap);
        return success();
      }
      // Pure rank-1 (existing logic from Task 6 — keep it).
      auto aff = matchAffineIndexInIV(indices[0], primaryIV);
      if (failed(aff)) {
        LLVM_DEBUG(llvm::dbgs() << "reject: non-affine index @"
                                << accessOp->getLoc() << "\n");
        return failure();
      }
      int64_t coeff = aff->first, k = aff->second;
      int64_t x0 = coeff * info.lb + k, x1 = coeff * (info.ub - 1) + k;
      int64_t amin = std::min(x0, x1), amax = std::max(x0, x1);
      int64_t bufExtent = ty.getShape()[0];
      if (amin < 0 || amax >= bufExtent) {
        LLVM_DEBUG(llvm::dbgs() << "reject: access OOB [" << amin << ","
                                << amax << "] on buffer extent " << bufExtent
                                << " @" << accessOp->getLoc() << "\n");
        return failure();
      }
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
      // Rule 13 — store target must not be stride-0 (that would be a
      // reduction, deferred to a separate pass — spec §9.2).
      if (isa<memref::StoreOp>(accessOp) && coeff == 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "reject: store target has stride 0 "
                      "(reduction pattern) @" << accessOp->getLoc() << "\n");
        return failure();
      }
      DsdAccessPattern ap{memRef, {coeff}, {amin}, 1};
      info.accesses.push_back(ap);
      return success();
    }
    if (ty.getRank() == 2 && info.isRank2) {
      // Rank-2 memref inside rank-2 nest — each dim gets its own (coeff, k).
      int64_t strides[2] = {0, 0};
      int64_t offsets[2] = {0, 0};
      // Dim 0 uses outer IV only; dim 1 uses inner IV only.
      auto affOuter =
          matchAffineIndexInIVPair(indices[0], primaryIV, secondaryIV);
      auto affInner =
          matchAffineIndexInIVPair(indices[1], primaryIV, secondaryIV);
      if (failed(affOuter) || failed(affInner)) return failure();
      if (std::get<1>(*affOuter) != 0 || std::get<0>(*affInner) != 0) {
        LLVM_DEBUG(llvm::dbgs() << "reject: rank-2 access mixes IVs @"
                                << accessOp->getLoc() << "\n");
        return failure();
      }
      int64_t s0 = std::get<0>(*affOuter);
      int64_t k0 = std::get<2>(*affOuter);
      int64_t s1 = std::get<1>(*affInner);
      int64_t k1 = std::get<2>(*affInner);
      // Effective access range start = min over iteration of (stride*iv + k).
      int64_t s0_x0 = s0 * info.lb + k0;
      int64_t s0_x1 = s0 * (info.ub - 1) + k0;
      int64_t amin0 = std::min(s0_x0, s0_x1);
      int64_t amax0 = std::max(s0_x0, s0_x1);
      int64_t s1_x0 = s1 * info.innerLb + k1;
      int64_t s1_x1 = s1 * (info.innerUb - 1) + k1;
      int64_t amin1 = std::min(s1_x0, s1_x1);
      int64_t amax1 = std::max(s1_x0, s1_x1);
      int64_t M = ty.getShape()[0], N = ty.getShape()[1];
      if (amin0 < 0 || amax0 >= M || amin1 < 0 || amax1 >= N) {
        LLVM_DEBUG(llvm::dbgs() << "reject: rank-2 OOB @"
                                << accessOp->getLoc() << "\n");
        return failure();
      }
      strides[0] = s0;  strides[1] = s1;
      offsets[0] = amin0;  offsets[1] = amin1;
      for (int64_t s : strides)
        if (s < kMinMem4dStride || s > kMaxMem4dStride) return failure();
      for (int64_t o : offsets)
        if (o < kMinDsdOffset || o > kMaxDsdOffset) return failure();
      // Rule 13 — store target must not be stride-0 in any dimension (that
      // would be a reduction, deferred to a separate pass — spec §9.2).
      if (isa<memref::StoreOp>(accessOp)) {
        for (int64_t s : strides) {
          if (s == 0) {
            LLVM_DEBUG(llvm::dbgs()
                       << "reject: store target has stride 0 "
                          "(reduction pattern) @" << accessOp->getLoc() << "\n");
            return failure();
          }
        }
      }
      DsdAccessPattern ap;
      ap.buffer = memRef;
      ap.strides = {strides[0], strides[1]};
      ap.offsets = {offsets[0], offsets[1]};
      ap.rank = 2;
      info.accesses.push_back(ap);
      return success();
    }
    LLVM_DEBUG(llvm::dbgs() << "reject: rank mismatch @"
                            << accessOp->getLoc() << "\n");
    return failure();
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
