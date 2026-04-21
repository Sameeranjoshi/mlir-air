//===- LoopIdiomAnalysis.cpp ------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/LoopIdiomAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "csl-auto-vectorize"

using namespace mlir;

namespace xilinx {
namespace air {

FailureOr<LoopIdiom> analyzeForLoop(scf::ForOp op) {
  LLVM_DEBUG(llvm::dbgs() << "reject: analyzer not yet implemented @"
                          << op.getLoc() << "\n");
  return failure();
}

} // namespace air
} // namespace xilinx
