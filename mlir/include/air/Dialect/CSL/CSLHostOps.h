//===- CSLHostOps.h - csl_host dialect operations ---------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef CSL_HOST_OPS_H
#define CSL_HOST_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Bytecode/BytecodeImplementation.h"

#include "air/Dialect/CSL/CSLHostOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "air/Dialect/CSL/CSLHostOps.h.inc"

#endif // CSL_HOST_OPS_H
