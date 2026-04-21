//===- LoopIdiomAnalysis.h --------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Shared legality helper for -csl-auto-vectorize pattern rewrites.  Every
// pattern calls analyzeForLoop() first; on success it returns a filled-in
// LoopIdiom struct describing loop shape, body classification, and per-access
// stride/offset data.  Body-content matching (is this loop an @fadds? an
// @fmacs?) is the pattern's responsibility, not the analyzer's.
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_TRANSFORMS_LOOP_IDIOM_ANALYSIS_H
#define AIR_DIALECT_CSL_TRANSFORMS_LOOP_IDIOM_ANALYSIS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace xilinx {
namespace air {

/// One memref access (load or store) classified by its affine pattern.
/// For rank-R memrefs, strides and offsets each have R entries.
///
/// `strides[d]` is the coefficient of the IV for dimension `d` (signed).
/// `offsets[d]` is the effective range-start for dimension `d`: the
/// minimum value of (stride*iv + k) over the iteration range.  This
/// means `memref.subview %buf[offsets] [extents] [strides]` with the
/// loop's extent covers the exact access range, regardless of sign.
/// For positive strides, offsets[d] equals the access at iv = lb.
/// For negative strides, offsets[d] equals the access at iv = ub-1.
struct DsdAccessPattern {
  mlir::Value buffer;                              // root memref SSA value
  llvm::SmallVector<int64_t, 2> strides;           // per-rank coefficient
  llvm::SmallVector<int64_t, 2> offsets;           // per-rank constant term
  unsigned rank = 1;                               // 1 or 2 in MVP
};

/// Outcome of analyzing one scf.for against the DSD legality predicate.
/// Valid only if analyzeForLoop() returned success.
struct LoopIdiom {
  // Loop shape
  int64_t lb = 0;
  int64_t ub = 0;
  int64_t step = 1;
  int64_t extent = 0;         // ub - lb  (always > 0)
  mlir::Value inductionVar;

  // For rank-2 nests, the inner loop's shape is recorded here.
  bool isRank2 = false;
  int64_t innerLb = 0;
  int64_t innerUb = 0;
  int64_t innerStep = 1;
  int64_t innerExtent = 0;
  mlir::Value innerInductionVar;

  // Body classification — all loads and stores inside body, plus the
  // arith-op chain (program order).  Terminators (scf.yield) are excluded.
  llvm::SmallVector<mlir::memref::LoadOp, 4> loads;
  llvm::SmallVector<mlir::memref::StoreOp, 1> stores;     // exactly 1
  llvm::SmallVector<mlir::Operation *, 8> bodyOps;        // arith.* only

  // SSA values defined outside the loop, used inside, classified:
  //   - memref buffers are captured per-access in `accesses` below
  //   - scalar loop-invariants (potential @fmuls/@fmacs scalar-broadcast
  //     operands) are collected here
  llvm::SmallVector<mlir::Value, 2> loopInvariantScalars;

  // One pattern entry per distinct memref-operand of a load/store inside
  // the body.  Populated in load+store order; a pattern can look them up
  // by pointer-equality against `loads[i].getMemRef()` etc.
  llvm::SmallVector<DsdAccessPattern, 4> accesses;
};

/// Run the full MVP legality predicate on `op`.  On failure returns
/// failure() and emits an LLVM_DEBUG(DBG_TYPE("csl-auto-vectorize"))
/// trace with the reject reason.  On success returns a LoopIdiom with
/// every field populated.
mlir::FailureOr<LoopIdiom> analyzeForLoop(mlir::scf::ForOp op);

/// DSD field-width limits straight from the SDK docs
/// (DSDs.md:36 for mem1d, :218-224 for mem4d).
constexpr int64_t kMaxDsdExtent = 65535;            // u16
constexpr int64_t kMaxMem1dStride = 127;            // i8
constexpr int64_t kMinMem1dStride = -128;
constexpr int64_t kMaxDsdOffset = 32767;            // i16
constexpr int64_t kMinDsdOffset = -32768;
constexpr int64_t kMaxMem4dStride = 32767;          // i16
constexpr int64_t kMinMem4dStride = -32768;

} // namespace air
} // namespace xilinx

#endif // AIR_DIALECT_CSL_TRANSFORMS_LOOP_IDIOM_ANALYSIS_H
