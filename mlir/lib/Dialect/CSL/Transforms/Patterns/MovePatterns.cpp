//===- MovePatterns.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Move and negate DSD idiom patterns for -csl-auto-vectorize:
//   @fmovs  -- memcpy c[i] = a[i]       (Task 13)
//   @fnegs  -- negate c[i] = -a[i]      (Task 14)
//
//===----------------------------------------------------------------------===//

#include "PatternsCommon.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "csl-auto-vectorize"

using namespace mlir;
using namespace xilinx::air;

namespace {

/// c[i] = a[i] on f32 -- one load, one store, zero body ops.
struct FmovsPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;
    if (info.loads.size() != 1 || info.stores.size() != 1 ||
        !info.bodyOps.empty())
      return failure();
    memref::LoadOp ld0 = info.loads[0];
    if (!ld0.getResult().getType().isF32()) return failure();
    memref::StoreOp st0 = info.stores[0];
    if (st0.getValue() != ld0.getResult()) return failure();

    LLVM_DEBUG(llvm::dbgs() << "match: FmovsPattern @" << op.getLoc() << "\n");

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value mrA = buildSubviewForAccess(rewriter, loc, info, info.accesses[0]);
    Value mrC = buildSubviewForAccess(rewriter, loc, info, info.accesses[1]);
    Value dA = buildGetMemDsd(rewriter, loc, mrA);
    Value dC = buildGetMemDsd(rewriter, loc, mrC);
    buildBuiltinCall(rewriter, loc, "fmovs", {dC, dA});
    rewriter.eraseOp(op);
    return success();
  }
};

/// c[i] = -a[i] on f32 -- one load, one store, one arith.negf body op.
struct FnegsPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;
    if (info.loads.size() != 1 || info.stores.size() != 1 ||
        info.bodyOps.size() != 1)
      return failure();
    auto neg = dyn_cast<arith::NegFOp>(info.bodyOps[0]);
    if (!neg || !neg.getType().isF32()) return failure();
    memref::LoadOp ld0 = info.loads[0];
    memref::StoreOp st0 = info.stores[0];
    if (neg.getOperand() != ld0.getResult()) return failure();
    if (st0.getValue() != neg.getResult()) return failure();

    LLVM_DEBUG(llvm::dbgs() << "match: FnegsPattern @" << op.getLoc() << "\n");

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value mrA = buildSubviewForAccess(rewriter, loc, info, info.accesses[0]);
    Value mrC = buildSubviewForAccess(rewriter, loc, info, info.accesses[1]);
    Value dA = buildGetMemDsd(rewriter, loc, mrA);
    Value dC = buildGetMemDsd(rewriter, loc, mrC);
    buildBuiltinCall(rewriter, loc, "fnegs", {dC, dA});
    rewriter.eraseOp(op);
    return success();
  }
};

} // anonymous namespace

namespace xilinx {
namespace air {

void populateMovePatterns(RewritePatternSet &patterns) {
  patterns.add<FmovsPattern, FnegsPattern>(patterns.getContext());
}

} // namespace air
} // namespace xilinx
