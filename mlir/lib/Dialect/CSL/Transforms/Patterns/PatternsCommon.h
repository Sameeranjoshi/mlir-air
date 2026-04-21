//===- PatternsCommon.h - DSD IR-construction helpers -----------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Shared IR-construction helpers for -csl-auto-vectorize pattern rewrites.
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_TRANSFORMS_PATTERNS_COMMON_H
#define AIR_DIALECT_CSL_TRANSFORMS_PATTERNS_COMMON_H

#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "air/Dialect/CSL/Transforms/LoopIdiomAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

namespace xilinx {
namespace air {

/// Build the memref SSA value that the DSD will wrap, applying
/// memref.subview where offsets/strides differ from the root buffer's
/// natural layout.  Returns an SSA value of memref type suitable as
/// csl.get_mem_dsd operand.
inline mlir::Value buildSubviewForAccess(mlir::PatternRewriter &rewriter,
                                         mlir::Location loc,
                                         const LoopIdiom &info,
                                         const DsdAccessPattern &ap) {
  using namespace mlir;
  auto rootTy = cast<MemRefType>(ap.buffer.getType());
  bool needsSubview = false;
  for (int64_t o : ap.offsets) if (o != 0) needsSubview = true;
  for (int64_t s : ap.strides) if (s != 1) needsSubview = true;
  // Rank-1: if the effective extent equals the buffer extent AND offset=0
  // AND stride=1, pass the raw memref.
  if (!needsSubview) return ap.buffer;

  // Compose subview offsets/sizes/strides.
  SmallVector<OpFoldResult> offsets, sizes, strides;
  if (ap.rank == 1) {
    offsets.push_back(rewriter.getIndexAttr(ap.offsets[0]));
    sizes.push_back(rewriter.getIndexAttr(info.extent));
    strides.push_back(rewriter.getIndexAttr(ap.strides[0]));
  } else {
    assert(ap.rank == 2);
    offsets.push_back(rewriter.getIndexAttr(ap.offsets[0]));
    offsets.push_back(rewriter.getIndexAttr(ap.offsets[1]));
    sizes.push_back(rewriter.getIndexAttr(info.extent));
    sizes.push_back(rewriter.getIndexAttr(info.innerExtent));
    strides.push_back(rewriter.getIndexAttr(ap.strides[0]));
    strides.push_back(rewriter.getIndexAttr(ap.strides[1]));
  }
  auto resultTy = cast<MemRefType>(
      memref::SubViewOp::inferResultType(rootTy, offsets, sizes, strides));
  return rewriter.create<memref::SubViewOp>(loc, resultTy, ap.buffer,
                                            offsets, sizes, strides);
}

/// Build `csl.get_mem_dsd` on the given memref.
inline mlir::Value buildGetMemDsd(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value memRef) {
  using namespace mlir;
  auto dsdTy = xilinx::csl::DsdType::get(rewriter.getContext());
  return rewriter.create<xilinx::csl::GetMemDsdOp>(loc, dsdTy, memRef);
}

/// Build a `csl.builtin_call "<callee>"(args...)` with no results.
inline void buildBuiltinCall(mlir::PatternRewriter &rewriter,
                             mlir::Location loc, llvm::StringRef callee,
                             mlir::ValueRange args) {
  using namespace mlir;
  rewriter.create<xilinx::csl::BuiltinCallOp>(
      loc,
      /*results=*/TypeRange{},
      /*callee=*/rewriter.getStringAttr(callee),
      /*module=*/Value{},
      /*operands=*/args);
}

} // namespace air
} // namespace xilinx

#endif // AIR_DIALECT_CSL_TRANSFORMS_PATTERNS_COMMON_H
