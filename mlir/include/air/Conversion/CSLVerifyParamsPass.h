//===- CSLVerifyParamsPass.h ------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Declares the -csl-verify-params pass factory.
//
// The pass validates that attributes carried on each `csl_layout.place` op
// match the block-argument names of the referenced `csl.program` template.
// Extra or missing parameters are reported as op errors so mis-wirings are
// caught before we emit CSL source.
//
//===----------------------------------------------------------------------===//

#ifndef AIR_CONVERSION_CSL_VERIFY_PARAMS_PASS_H
#define AIR_CONVERSION_CSL_VERIFY_PARAMS_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;
} // namespace mlir

namespace xilinx {
namespace air {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCSLVerifyParamsPass();

} // namespace air
} // namespace xilinx

#endif // AIR_CONVERSION_CSL_VERIFY_PARAMS_PASS_H
