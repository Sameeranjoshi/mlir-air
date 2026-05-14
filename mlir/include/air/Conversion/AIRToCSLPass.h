//===- AIRToCSLPass.h - Lower AIR to CSL v2 wafer IR ----------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//
//
// Declares the -air-to-csl pass factory. Converts func.func bodies containing
// air.launch > air.segment > air.herd hierarchies into csl.wafer ops with
// three sibling regions:
//   csl.program — PE kernel template (cloned herd body)
//   csl.layout  — placement grid (1x1 for V1)
//   csl.host    — host data movement (memcpy_h2d/d2h + launch)
//
// V1 scope: 1x1 herds only, no channels, no DMA, static memrefs.
//===----------------------------------------------------------------------===//

#ifndef AIR_CONVERSION_AIRTOCSL_PASS_H
#define AIR_CONVERSION_AIRTOCSL_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;
} // namespace mlir

namespace xilinx {
namespace air {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAIRToCSLPass();

} // namespace air
} // namespace xilinx

#endif // AIR_CONVERSION_AIRTOCSL_PASS_H
