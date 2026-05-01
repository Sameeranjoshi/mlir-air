//===- CSLLowerStreamRoutingPass.h ---------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Declares the -csl-lower-stream-routing pass factory.
//
// Pass 3 of csl-streams-to-csl pipeline.
// Pre:  every csl_layout.stream has {color = @<x>}; matching color has an `id`.
// Post: each stream has two csl_layout.set_color_config siblings (one at
//       `from` coord, one at `to` coord); stream op is kept (still needed by
//       Pass 4 for put/get symbol resolution).
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_TRANSFORMS_CSL_LOWER_STREAM_ROUTING_PASS_H
#define AIR_DIALECT_CSL_TRANSFORMS_CSL_LOWER_STREAM_ROUTING_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createCSLLowerStreamRoutingPass();

} // namespace air
} // namespace xilinx

#endif // AIR_DIALECT_CSL_TRANSFORMS_CSL_LOWER_STREAM_ROUTING_PASS_H
