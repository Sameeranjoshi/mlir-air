//===- CSLLayoutDialect.cpp - csl_layout dialect implementation -*-C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "air/Dialect/CSL/CSLLayoutOpsDialect.cpp.inc"

namespace xilinx::csl_layout {

void CSLLayoutDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "air/Dialect/CSL/CSLLayoutOps.cpp.inc"
      >();
}

} // namespace xilinx::csl_layout
