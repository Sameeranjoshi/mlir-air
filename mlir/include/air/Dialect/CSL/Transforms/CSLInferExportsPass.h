//===- CSLInferExportsPass.h ------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Declares the -csl-infer-exports pass factory.
//
// The pass walks csl.host regions for csl_host.memcpy_h2d / memcpy_d2h /
// launch ops and auto-generates the corresponding csl.export ops in
// csl.program and csl_layout.export ops in csl.layout.  Pre-existing exports
// not referenced by any host op receive direction = "internal".
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_TRANSFORMS_CSL_INFER_EXPORTS_PASS_H
#define AIR_DIALECT_CSL_TRANSFORMS_CSL_INFER_EXPORTS_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;
} // namespace mlir

namespace xilinx {
namespace air {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCSLInferExportsPass();

} // namespace air
} // namespace xilinx

#endif // AIR_DIALECT_CSL_TRANSFORMS_CSL_INFER_EXPORTS_PASS_H
