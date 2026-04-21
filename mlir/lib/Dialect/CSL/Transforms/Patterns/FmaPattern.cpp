//===- FmaPattern.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "PatternsCommon.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "csl-auto-vectorize"

using namespace mlir;
using namespace xilinx::air;

namespace {

/// Match body signature:
///   %va = load a[i]; %vb = load b[i]; %vc = load c[i];
///   %m  = mulf %va, %vb : f32
///   %s  = addf %m,  %vc : f32     (or addf %vc, %m -- commutative)
///   store %s, c[i]
/// The store target must match one of the loaded buffers (accumulator).
struct FmacsPattern : public OpRewritePattern<scf::ForOp> {
  FmacsPattern(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/2) {}
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;

    // 3 loads (a, b, accumulator), 1 store, 2 body ops (mulf, addf).
    if (info.loads.size() != 3 || info.stores.size() != 1 ||
        info.bodyOps.size() != 2)
      return failure();

    // Identify the two ops (program order).
    auto mul = dyn_cast<arith::MulFOp>(info.bodyOps[0]);
    auto add = dyn_cast<arith::AddFOp>(info.bodyOps[1]);
    if (!mul || !add) return failure();
    if (!mul.getType().isF32() || !add.getType().isF32()) return failure();

    // addf's operands are (mul.result, X) or (X, mul.result); X must be a
    // load result.
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

    // The store target buffer must be the one accLoadVal was loaded from,
    // at the same index (analyzer ensures index = iv).
    memref::LoadOp accLoad;
    for (auto ld : info.loads) {
      memref::LoadOp ldMut = ld;
      if (ldMut.getResult() == accLoadVal) {
        accLoad = ldMut;
        break;
      }
    }
    if (!accLoad) return failure();

    memref::StoreOp st0 = info.stores[0];
    if (st0.getMemRef() != accLoad.getMemRef()) return failure();
    if (st0.getValue() != add.getResult()) return failure();

    LLVM_DEBUG(llvm::dbgs() << "match: FmacsPattern @" << op.getLoc() << "\n");

    // Identify access-pattern indices for a, b, c_acc.
    auto findAp = [&](Value buf) -> const DsdAccessPattern * {
      for (const auto &ap : info.accesses)
        if (ap.buffer == buf) return &ap;
      return nullptr;
    };
    // mA and mB are load results; get their memrefs.
    memref::LoadOp ldA, ldB;
    for (auto ld : info.loads) {
      memref::LoadOp ldMut = ld;
      if (ldMut.getResult() == mA) { ldA = ldMut; }
      if (ldMut.getResult() == mB) { ldB = ldMut; }
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
    // @fmacs(dest, src_acc, src_a, src_b) -- acc is read *and* written.
    buildBuiltinCall(rewriter, loc, "fmacs", {dC, dC, dA, dB});
    rewriter.eraseOp(op);
    return success();
  }
};

} // anonymous namespace

namespace xilinx {
namespace air {

void populateFmaPattern(RewritePatternSet &patterns) {
  patterns.add<FmacsPattern>(patterns.getContext());
}

} // namespace air
} // namespace xilinx
