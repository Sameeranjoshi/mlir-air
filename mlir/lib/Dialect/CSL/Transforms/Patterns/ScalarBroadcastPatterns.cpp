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

} // anonymous namespace

namespace xilinx {
namespace air {

void populateScalarBroadcastPatterns(RewritePatternSet &patterns) {
  patterns.add<FmulsScalarPattern>(patterns.getContext());
}

} // namespace air
} // namespace xilinx
