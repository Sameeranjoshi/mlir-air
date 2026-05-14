//===- Pipelines.h ------------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Registration entry point for the named CSL pass pipelines:
//   --csl-dataflow-to-csl  : 4-pass stream-lowering chain
//   --csl-pipeline        : full chain (infer-exports + auto-vectorize +
//                           dataflow-to-csl)
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_PIPELINES_H
#define AIR_DIALECT_CSL_PIPELINES_H

namespace xilinx {
namespace air {

void registerCSLPipelines();

} // namespace air
} // namespace xilinx

#endif // AIR_DIALECT_CSL_PIPELINES_H
