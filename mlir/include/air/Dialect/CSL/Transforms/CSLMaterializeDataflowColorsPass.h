//===- CSLMaterializeDataflowColorsPass.h ---------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Declares the -csl-materialize-dataflow-colors pass factory.
//
// Pass 1 of csl-dataflow-to-csl pipeline.
// Pre:  every csl_layout.dataflow has no `color` attr.
// Post: every csl_layout.dataflow has {color = @<sym>}; matching csl.color
//       @<sym> exists in same csl.layout body (no id yet).
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_TRANSFORMS_CSL_MATERIALIZE_DATAFLOW_COLORS_PASS_H
#define AIR_DIALECT_CSL_TRANSFORMS_CSL_MATERIALIZE_DATAFLOW_COLORS_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createCSLMaterializeDataflowColorsPass();

} // namespace air
} // namespace xilinx

#endif // AIR_DIALECT_CSL_TRANSFORMS_CSL_MATERIALIZE_DATAFLOW_COLORS_PASS_H
