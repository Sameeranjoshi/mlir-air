//===- CSLLowerDataflowDataPass.h ---------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Declares the -csl-lower-dataflow-data pass factory.
//
// Pass 4 of csl-dataflow-to-csl pipeline.
// Pre:  Stage-3 form (set_color_configs in place; streams + put/get still
//       present).
// Post: all csl.dataflow.put/get and csl_layout.dataflow ops erased; each program
//       contains the matching fabric DSDs + auto-generated tasks + async
//       builtin calls.
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_TRANSFORMS_CSL_LOWER_DATAFLOW_DATA_PASS_H
#define AIR_DIALECT_CSL_TRANSFORMS_CSL_LOWER_DATAFLOW_DATA_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createCSLLowerDataflowDataPass();

} // namespace air
} // namespace xilinx

#endif // AIR_DIALECT_CSL_TRANSFORMS_CSL_LOWER_DATAFLOW_DATA_PASS_H
