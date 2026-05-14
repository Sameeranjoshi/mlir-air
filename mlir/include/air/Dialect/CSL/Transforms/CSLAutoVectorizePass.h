//===- CSLAutoVectorizePass.h -----------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Declares the -csl-auto-vectorize pass factory.
//
// The pass walks csl.program bodies for scf.for loops whose body matches a
// known CSL DSD idiom (elementwise f32 add/sub/mul, FMA, mov, neg, plus
// scalar-broadcast variants) and rewrites the loop into
// csl.get_mem_dsd + csl.builtin_call.  Loops that fail the legality
// predicate are preserved unchanged (Tier-4 scalar fall-through).
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_TRANSFORMS_CSL_AUTO_VECTORIZE_PASS_H
#define AIR_DIALECT_CSL_TRANSFORMS_CSL_AUTO_VECTORIZE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;
} // namespace mlir

namespace xilinx {
namespace air {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCSLAutoVectorizePass();

} // namespace air
} // namespace xilinx

#endif // AIR_DIALECT_CSL_TRANSFORMS_CSL_AUTO_VECTORIZE_PASS_H
