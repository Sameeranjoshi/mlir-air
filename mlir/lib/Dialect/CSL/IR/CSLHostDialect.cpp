//===- CSLHostDialect.cpp - csl_host dialect implementation -------C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/CSLHostOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "air/Dialect/CSL/CSLHostOpsDialect.cpp.inc"

namespace xilinx::csl_host {

void CSLHostDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "air/Dialect/CSL/CSLHostOps.cpp.inc"
      >();
}

} // namespace xilinx::csl_host
