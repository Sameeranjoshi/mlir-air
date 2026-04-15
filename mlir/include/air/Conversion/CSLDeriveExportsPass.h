//===- CSLDeriveExportsPass.h -----------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Declares the -csl-derive-exports pass factory.
//
// The pass scans csl.host regions for csl_host.memcpy_h2d / memcpy_d2h /
// launch ops and annotates the corresponding csl.export ops in csl.program
// regions with direction = "in" or "out".  Exports that are not referenced
// by any host transfer ops receive direction = "internal".
//
//===----------------------------------------------------------------------===//

#ifndef AIR_CONVERSION_CSL_DERIVE_EXPORTS_PASS_H
#define AIR_CONVERSION_CSL_DERIVE_EXPORTS_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;
} // namespace mlir

namespace xilinx {
namespace air {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCSLDeriveExportsPass();

} // namespace air
} // namespace xilinx

#endif // AIR_CONVERSION_CSL_DERIVE_EXPORTS_PASS_H
