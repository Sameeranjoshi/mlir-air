//===- CSLLayoutOps.h - csl_layout dialect operations -----------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef CSL_LAYOUT_OPS_H
#define CSL_LAYOUT_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Bytecode/BytecodeImplementation.h"

#include "air/Dialect/CSL/CSLLayoutOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "air/Dialect/CSL/CSLLayoutOps.h.inc"

#endif // CSL_LAYOUT_OPS_H
