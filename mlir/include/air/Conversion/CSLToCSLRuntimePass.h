//===- CSLToCSLRuntimePass.h - CSL to CSL Runtime lowering pass -*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_CONVERSION_CSLTO_CSLRUNTIME_PASS_H
#define AIR_CONVERSION_CSLTO_CSLRUNTIME_PASS_H

#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace air {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCSLToCSLRuntimePass();

} // namespace air
} // namespace xilinx

#endif // AIR_CONVERSION_CSLTO_CSLRUNTIME_PASS_H
