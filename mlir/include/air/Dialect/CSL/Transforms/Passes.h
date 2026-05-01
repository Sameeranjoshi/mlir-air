//===- Passes.h ---------------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Aggregate header + registration entry point for every pass that operates
// intra-CSL-dialect (CSL -> CSL rewrites).  Air-opt calls
// registerCSLTransformPasses() in addition to registerConversionPasses().
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_TRANSFORMS_PASSES_H
#define AIR_DIALECT_CSL_TRANSFORMS_PASSES_H

#include "air/Dialect/CSL/Transforms/CSLAllocateColorIdsPass.h"
#include "air/Dialect/CSL/Transforms/CSLAutoVectorizePass.h"
#include "air/Dialect/CSL/Transforms/CSLInferExportsPass.h"
#include "air/Dialect/CSL/Transforms/CSLLowerStreamRoutingPass.h"
#include "air/Dialect/CSL/Transforms/CSLMaterializeStreamColorsPass.h"

namespace xilinx {
namespace air {

void registerCSLTransformPasses();

} // namespace air
} // namespace xilinx

#endif // AIR_DIALECT_CSL_TRANSFORMS_PASSES_H
