//===- CSLAllocateColorIdsPass.h ---------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Declares the -csl-allocate-color-ids pass factory.
//
// Pass 2 of csl-dataflow-to-csl pipeline.
// Pre:  every csl.color exists (with or without `id`).
// Post: every csl.color has an `id` attribute.
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_TRANSFORMS_CSL_ALLOCATE_COLOR_IDS_PASS_H
#define AIR_DIALECT_CSL_TRANSFORMS_CSL_ALLOCATE_COLOR_IDS_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createCSLAllocateColorIdsPass();

} // namespace air
} // namespace xilinx

#endif // AIR_DIALECT_CSL_TRANSFORMS_CSL_ALLOCATE_COLOR_IDS_PASS_H
