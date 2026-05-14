//===- ScalarBroadcastPatterns.cpp ------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Scalar-broadcast DSD idiom patterns for -csl-auto-vectorize:
//   @fmuls  -- c[i] = alpha * a[i]          (Task 15)
//   @fmacs  -- y[i] = alpha * a[i] + y[i]   (Task 16, saxpy)
//
//===----------------------------------------------------------------------===//

#include "PatternsCommon.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "csl-auto-vectorize"

using namespace mlir;
using namespace xilinx::air;

namespace {

/// c[i] = alpha * a[i], alpha loop-invariant f32.
/// Shape: 1 load, 1 store, 1 mulf; one mulf operand is the load result,
/// the other is in loopInvariantScalars.
/// Benefit=2 to beat FmulsPattern (benefit=1) — both match on scf::ForOp
/// with a single mulf body op but FmulsPattern wants 2 loads.
struct FmulsScalarPattern : public OpRewritePattern<scf::ForOp> {
  FmulsScalarPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/2) {}
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;
    if (info.loads.size() != 1 || info.stores.size() != 1 ||
        info.bodyOps.size() != 1)
      return failure();
    auto mul = dyn_cast<arith::MulFOp>(info.bodyOps[0]);
    if (!mul || !mul.getType().isF32()) return failure();

    memref::LoadOp ld0 = info.loads[0];
    Value loadVal = ld0.getResult();
    Value scalar;
    if (mul.getLhs() == loadVal)
      scalar = mul.getRhs();
    else if (mul.getRhs() == loadVal)
      scalar = mul.getLhs();
    else
      return failure();

    if (!llvm::is_contained(info.loopInvariantScalars, scalar))
      return failure();

    memref::StoreOp st0 = info.stores[0];
    if (st0.getValue() != mul.getResult()) return failure();

    LLVM_DEBUG(llvm::dbgs() << "match: FmulsScalarPattern @"
                            << op.getLoc() << "\n");

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value mrA = buildSubviewForAccess(rewriter, loc, info, info.accesses[0]);
    Value mrC = buildSubviewForAccess(rewriter, loc, info, info.accesses[1]);
    Value dA = buildGetMemDsd(rewriter, loc, mrA);
    Value dC = buildGetMemDsd(rewriter, loc, mrC);
    buildBuiltinCall(rewriter, loc, "fmuls", {dC, dA, scalar});
    rewriter.eraseOp(op);
    return success();
  }
};

/// y[i] = alpha * a[i] + y[i] (saxpy), alpha loop-invariant f32.
/// Shape: 2 loads (a, y), 1 store (y), 2 body ops (mulf + addf).
/// The mulf has one load-result operand (a[i]) and one scalar operand (alpha).
/// The addf takes mulf.result and the y[i] load result.
/// The store target buffer equals the accumulator load's buffer (y).
/// Benefit=3 — beats FmacsPattern (benefit=2) and FmulsScalarPattern (benefit=2).
struct FmacsScalarPattern : public OpRewritePattern<scf::ForOp> {
  FmacsScalarPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/3) {}
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;
    if (info.loads.size() != 2 || info.stores.size() != 1 ||
        info.bodyOps.size() != 2)
      return failure();
    auto mul = dyn_cast<arith::MulFOp>(info.bodyOps[0]);
    auto add = dyn_cast<arith::AddFOp>(info.bodyOps[1]);
    if (!mul || !add) return failure();
    if (!mul.getType().isF32() || !add.getType().isF32()) return failure();

    // One of mulf's operands must be a load result; the other must be a
    // loop-invariant scalar.
    memref::LoadOp ld0 = info.loads[0], ld1 = info.loads[1];
    Value l0 = ld0.getResult(), l1 = ld1.getResult();

    // Identify which load is the "a" buffer and which is the accumulator "y".
    Value loadA, loadAcc;
    Value scalar;
    // Try: mul LHS is a load, RHS is scalar.
    if (mul.getLhs() == l0 && llvm::is_contained(info.loopInvariantScalars,
                                                  mul.getRhs())) {
      loadA = l0; scalar = mul.getRhs();
    } else if (mul.getLhs() == l1 && llvm::is_contained(
                                         info.loopInvariantScalars, mul.getRhs())) {
      loadA = l1; scalar = mul.getRhs();
    } else if (mul.getRhs() == l0 && llvm::is_contained(
                                         info.loopInvariantScalars, mul.getLhs())) {
      loadA = l0; scalar = mul.getLhs();
    } else if (mul.getRhs() == l1 && llvm::is_contained(
                                         info.loopInvariantScalars, mul.getLhs())) {
      loadA = l1; scalar = mul.getLhs();
    } else {
      return failure();
    }
    // The accumulator load is the other one.
    loadAcc = (loadA == l0) ? l1 : l0;

    // addf's operands: (mul.result, loadAcc) or (loadAcc, mul.result).
    Value mulRes = mul.getResult();
    if (!((add.getLhs() == mulRes && add.getRhs() == loadAcc) ||
          (add.getRhs() == mulRes && add.getLhs() == loadAcc)))
      return failure();

    // Store value must be addf.result; store target must be acc's buffer.
    memref::StoreOp st0 = info.stores[0];
    if (st0.getValue() != add.getResult()) return failure();

    // Find the accumulator load op to get its memref.
    memref::LoadOp accLoad;
    for (auto ld : info.loads) {
      memref::LoadOp ldMut = ld;
      if (ldMut.getResult() == loadAcc) { accLoad = ldMut; break; }
    }
    if (!accLoad) return failure();
    if (st0.getMemRef() != accLoad.getMemRef()) return failure();

    // Find the "a" load op.
    memref::LoadOp aLoad;
    for (auto ld : info.loads) {
      memref::LoadOp ldMut = ld;
      if (ldMut.getResult() == loadA) { aLoad = ldMut; break; }
    }
    if (!aLoad) return failure();

    // Look up access patterns by buffer.
    auto findAp = [&](Value buf) -> const DsdAccessPattern * {
      for (const auto &ap : info.accesses)
        if (ap.buffer == buf) return &ap;
      return nullptr;
    };
    const auto *apA   = findAp(aLoad.getMemRef());
    const auto *apAcc = findAp(accLoad.getMemRef());
    if (!apA || !apAcc) return failure();

    LLVM_DEBUG(llvm::dbgs() << "match: FmacsScalarPattern @"
                            << op.getLoc() << "\n");

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value mrA   = buildSubviewForAccess(rewriter, loc, info, *apA);
    Value mrAcc = buildSubviewForAccess(rewriter, loc, info, *apAcc);
    Value dA    = buildGetMemDsd(rewriter, loc, mrA);
    Value dAcc  = buildGetMemDsd(rewriter, loc, mrAcc);
    // @fmacs(dest, src_acc, src_a, alpha) — acc is read and written.
    buildBuiltinCall(rewriter, loc, "fmacs", {dAcc, dAcc, dA, scalar});
    rewriter.eraseOp(op);
    return success();
  }
};

} // anonymous namespace

namespace xilinx {
namespace air {

void populateScalarBroadcastPatterns(RewritePatternSet &patterns) {
  patterns.add<FmulsScalarPattern, FmacsScalarPattern>(patterns.getContext());
}

} // namespace air
} // namespace xilinx
