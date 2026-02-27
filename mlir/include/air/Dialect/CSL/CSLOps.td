//===- CSLOps.td - Top-level CSL dialect ops ---------------*- tablegen -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Top-level TableGen file that includes all CSL sub-dialect operation
// categories. This is the single file used by `add_mlir_dialect` to
// generate the complete set of CSL operations.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_OPS
#define CSL_OPS

include "air/Dialect/CSL/CSLBase.td"
include "air/Dialect/CSL/CSLLayoutOps.td"
include "air/Dialect/CSL/CSLRoutingOps.td"
include "air/Dialect/CSL/CSLKernelOps.td"
include "air/Dialect/CSL/CSLDataMovementOps.td"
include "air/Dialect/CSL/CSLRuntimeOps.td"

#endif // CSL_OPS
