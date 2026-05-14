//===- ElementwisePatterns.cpp -----------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Elementwise binary DSD idiom patterns for -csl-auto-vectorize:
//   @fadds  — arith.addf c = a + b
//   @fsubs  — arith.subf c = a - b   (added in Task 10)
//   @fmuls  — arith.mulf c = a * b   (added in Task 11)
//
//===----------------------------------------------------------------------===//

#include "PatternsCommon.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "csl-auto-vectorize"

using namespace mlir;
using namespace xilinx::air;

namespace {

/// Match body signature:
///   %0 = memref.load a[idx] : memref<_xf32>
///   %1 = memref.load b[idx] : memref<_xf32>
///   %2 = arith.addf %0, %1 : f32
///   memref.store %2, c[idx]
struct FaddsPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;

    // Shape: 2 loads, 1 store, 1 body op (arith.addf), all f32.
    if (info.loads.size() != 2 || info.stores.size() != 1 ||
        info.bodyOps.size() != 1)
      return failure();
    auto add = dyn_cast<arith::AddFOp>(info.bodyOps[0]);
    if (!add) return failure();
    if (!add.getType().isF32()) return failure();

    // The store's value must be the addf's result.
    memref::StoreOp st0 = info.stores[0];
    memref::LoadOp ld0 = info.loads[0], ld1 = info.loads[1];
    if (st0.getValue() != add.getResult()) return failure();
    // The addf's operands must be the two loads (order-independent).
    Value l0 = ld0.getResult();
    Value l1 = ld1.getResult();
    Value ra = add.getLhs(), rb = add.getRhs();
    if (!((ra == l0 && rb == l1) || (ra == l1 && rb == l0)))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "match: FaddsPattern @" << op.getLoc() << "\n");

    // accesses[0..1] correspond to loads[0..1]; accesses[2] to stores[0].
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

struct FsubsPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;
    if (info.loads.size() != 2 || info.stores.size() != 1 ||
        info.bodyOps.size() != 1)
      return failure();
    auto sub = dyn_cast<arith::SubFOp>(info.bodyOps[0]);
    if (!sub || !sub.getType().isF32()) return failure();
    memref::StoreOp st0 = info.stores[0];
    memref::LoadOp ld0 = info.loads[0], ld1 = info.loads[1];
    if (st0.getValue() != sub.getResult()) return failure();
    Value l0 = ld0.getResult();
    Value l1 = ld1.getResult();
    if (!(sub.getLhs() == l0 && sub.getRhs() == l1)) return failure();

    LLVM_DEBUG(llvm::dbgs() << "match: FsubsPattern @" << op.getLoc() << "\n");

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value mrA = buildSubviewForAccess(rewriter, loc, info, info.accesses[0]);
    Value mrB = buildSubviewForAccess(rewriter, loc, info, info.accesses[1]);
    Value mrC = buildSubviewForAccess(rewriter, loc, info, info.accesses[2]);
    Value dA  = buildGetMemDsd(rewriter, loc, mrA);
    Value dB  = buildGetMemDsd(rewriter, loc, mrB);
    Value dC  = buildGetMemDsd(rewriter, loc, mrC);
    buildBuiltinCall(rewriter, loc, "fsubs", {dC, dA, dB});
    rewriter.eraseOp(op);
    return success();
  }
};

struct FmulsPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;
    if (info.loads.size() != 2 || info.stores.size() != 1 ||
        info.bodyOps.size() != 1)
      return failure();
    auto mul = dyn_cast<arith::MulFOp>(info.bodyOps[0]);
    if (!mul || !mul.getType().isF32()) return failure();
    memref::StoreOp st0 = info.stores[0];
    memref::LoadOp ld0 = info.loads[0], ld1 = info.loads[1];
    if (st0.getValue() != mul.getResult()) return failure();
    Value l0 = ld0.getResult();
    Value l1 = ld1.getResult();
    Value ra = mul.getLhs(), rb = mul.getRhs();
    if (!((ra == l0 && rb == l1) || (ra == l1 && rb == l0)))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "match: FmulsPattern @" << op.getLoc() << "\n");

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value mrA = buildSubviewForAccess(rewriter, loc, info, info.accesses[0]);
    Value mrB = buildSubviewForAccess(rewriter, loc, info, info.accesses[1]);
    Value mrC = buildSubviewForAccess(rewriter, loc, info, info.accesses[2]);
    Value dA  = buildGetMemDsd(rewriter, loc, mrA);
    Value dB  = buildGetMemDsd(rewriter, loc, mrB);
    Value dC  = buildGetMemDsd(rewriter, loc, mrC);
    buildBuiltinCall(rewriter, loc, "fmuls", {dC, dA, dB});
    rewriter.eraseOp(op);
    return success();
  }
};

} // anonymous namespace

namespace xilinx {
namespace air {

void populateElementwisePatterns(RewritePatternSet &patterns) {
  patterns.add<FaddsPattern, FsubsPattern, FmulsPattern>(patterns.getContext());
}

} // namespace air
} // namespace xilinx
