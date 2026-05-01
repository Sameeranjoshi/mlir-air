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

// Direction enum is generated from CSLBase.td and shared with the csl dialect;
// CSLOps.h has the full include guard around the .inc, so pulling it in here
// avoids a duplicate-definition error when both headers end up in the same TU.
#include "air/Dialect/CSL/CSLOps.h"

#include "air/Dialect/CSL/CSLLayoutOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "air/Dialect/CSL/CSLLayoutOps.h.inc"

#endif // CSL_LAYOUT_OPS_H
