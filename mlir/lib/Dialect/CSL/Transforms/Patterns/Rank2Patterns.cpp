//===- Rank2Patterns.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Rank-2 (perfect 2-deep scf.for nest over rank-2 memrefs) DSD idiom
// patterns for -csl-auto-vectorize:
//   Rank2FaddsPattern  — C[i,j] = A[i,j] + B[i,j]   (@fadds, benefit=2)
//   Rank2FmacsPattern  — C[i,j]+= A[i,j] * B[i,j]   (@fmacs, benefit=3)
//
//===----------------------------------------------------------------------===//

#include "PatternsCommon.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "csl-auto-vectorize"

using namespace mlir;
using namespace xilinx::air;

namespace {

/// Rank-2 body signature identical to FaddsPattern but gated on
/// `info.isRank2 == true`.  Benefit=2 beats the rank-1 FaddsPattern
/// (benefit=1); rank-1 loops can never satisfy isRank2 anyway, so the
/// ordering only matters for clarity.
struct Rank2FaddsPattern : public OpRewritePattern<scf::ForOp> {
  Rank2FaddsPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/2) {}
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;
    if (!info.isRank2) return failure();
    if (info.loads.size() != 2 || info.stores.size() != 1 ||
        info.bodyOps.size() != 1) return failure();
    auto add = dyn_cast<arith::AddFOp>(info.bodyOps[0]);
    if (!add || !add.getType().isF32()) return failure();
    memref::LoadOp ld0 = info.loads[0], ld1 = info.loads[1];
    memref::StoreOp st0 = info.stores[0];
    Value l0 = ld0.getResult();
    Value l1 = ld1.getResult();
    if (!((add.getLhs() == l0 && add.getRhs() == l1) ||
          (add.getLhs() == l1 && add.getRhs() == l0))) return failure();
    if (st0.getValue() != add.getResult()) return failure();

    LLVM_DEBUG(llvm::dbgs() << "match: Rank2FaddsPattern @" << op.getLoc()
                            << "\n");

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value mrA = buildSubviewForAccess(rewriter, loc, info, info.accesses[0]);
    Value mrB = buildSubviewForAccess(rewriter, loc, info, info.accesses[1]);
    Value mrC = buildSubviewForAccess(rewriter, loc, info, info.accesses[2]);
    Value dA  = buildGetMemDsd(rewriter, loc, mrA);
    Value dB  = buildGetMemDsd(rewriter, loc, mrB);
    Value dC  = buildGetMemDsd(rewriter, loc, mrC);
    buildBuiltinCall(rewriter, loc, "fadds", {dC, dA, dB});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Rank-2 FMA: C[i,j] += A[i,j] * B[i,j].
/// Body signature: 3 loads (A, B, C_acc), mulf + addf, store back to C.
/// Gated on `info.isRank2`.  Benefit=3 beats Rank2FaddsPattern (benefit=2).
struct Rank2FmacsPattern : public OpRewritePattern<scf::ForOp> {
  Rank2FmacsPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/3) {}
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;
    if (!info.isRank2) return failure();

    // 3 loads, 1 store, 2 body ops (mulf then addf).
    if (info.loads.size() != 3 || info.stores.size() != 1 ||
        info.bodyOps.size() != 2) return failure();

    auto mul = dyn_cast<arith::MulFOp>(info.bodyOps[0]);
    auto add = dyn_cast<arith::AddFOp>(info.bodyOps[1]);
    if (!mul || !add) return failure();
    if (!mul.getType().isF32() || !add.getType().isF32()) return failure();

    // addf's operands: one must be mul.result; the other must be a load result.
    Value mulRes = mul.getResult();
    Value accLoadVal;
    if (add.getLhs() == mulRes)
      accLoadVal = add.getRhs();
    else if (add.getRhs() == mulRes)
      accLoadVal = add.getLhs();
    else
      return failure();

    // mul's operands must both be load results.
    Value mA = mul.getLhs(), mB = mul.getRhs();
    auto isLoadResult = [&](Value v) {
      return llvm::any_of(info.loads, [&](memref::LoadOp ld) {
        memref::LoadOp ldMut = ld;
        return ldMut.getResult() == v;
      });
    };
    if (!isLoadResult(mA) || !isLoadResult(mB) || !isLoadResult(accLoadVal))
      return failure();

    // Store target must be the accumulator buffer.
    memref::LoadOp accLoad;
    for (auto ld : info.loads) {
      memref::LoadOp ldMut = ld;
      if (ldMut.getResult() == accLoadVal) { accLoad = ldMut; break; }
    }
    if (!accLoad) return failure();

    memref::StoreOp st0 = info.stores[0];
    if (st0.getMemRef() != accLoad.getMemRef()) return failure();
    if (st0.getValue() != add.getResult()) return failure();

    LLVM_DEBUG(llvm::dbgs() << "match: Rank2FmacsPattern @" << op.getLoc()
                            << "\n");

    // Find access patterns for each buffer.
    auto findAp = [&](Value buf) -> const DsdAccessPattern * {
      for (const auto &ap : info.accesses)
        if (ap.buffer == buf) return &ap;
      return nullptr;
    };
    memref::LoadOp ldA, ldB;
    for (auto ld : info.loads) {
      memref::LoadOp ldMut = ld;
      if (ldMut.getResult() == mA) ldA = ldMut;
      if (ldMut.getResult() == mB) ldB = ldMut;
    }
    if (!ldA || !ldB) return failure();

    const auto *apA = findAp(ldA.getMemRef());
    const auto *apB = findAp(ldB.getMemRef());
    const auto *apC = findAp(accLoad.getMemRef());
    if (!apA || !apB || !apC) return failure();

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value mrA = buildSubviewForAccess(rewriter, loc, info, *apA);
    Value mrB = buildSubviewForAccess(rewriter, loc, info, *apB);
    Value mrC = buildSubviewForAccess(rewriter, loc, info, *apC);
    Value dA  = buildGetMemDsd(rewriter, loc, mrA);
    Value dB  = buildGetMemDsd(rewriter, loc, mrB);
    Value dC  = buildGetMemDsd(rewriter, loc, mrC);
    // @fmacs(dest, src_acc, src_a, src_b) — acc is read AND written.
    buildBuiltinCall(rewriter, loc, "fmacs", {dC, dC, dA, dB});
    rewriter.eraseOp(op);
    return success();
  }
};

} // anonymous namespace

namespace xilinx {
namespace air {

void populateRank2Patterns(RewritePatternSet &patterns) {
  patterns.add<Rank2FaddsPattern, Rank2FmacsPattern>(patterns.getContext());
}

} // namespace air
} // namespace xilinx
