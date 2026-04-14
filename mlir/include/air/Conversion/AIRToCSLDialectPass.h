//===- AIRToCSLDialectPass.h ------------------------------------*- C++ -*-===//
//
// Lowers AIR dialect ops (1x1 herds only, milestone scope) to the CSL
// dialect (csl.spatial_placement, csl.kernel, csl.code_region, csl.place,
// csl.export_name).
//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#ifndef AIR_CONVERSION_AIRTOCSLDIALECT_PASS_H
#define AIR_CONVERSION_AIRTOCSLDIALECT_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;
} // namespace mlir

namespace xilinx {
namespace air {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIRToCSLDialectPass();

} // namespace air
} // namespace xilinx

#endif // AIR_CONVERSION_AIRTOCSLDIALECT_PASS_H
