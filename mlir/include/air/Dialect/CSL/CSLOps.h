//===- CSLOps.h - CSL dialect operations -----------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef CSL_OPS_H
#define CSL_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "air/Dialect/CSL/CSLDialect.h"

#include "air/Dialect/CSL/CSLEnums.h.inc"

#define GET_OP_CLASSES
#include "air/Dialect/CSL/CSLOps.h.inc"

#endif // CSL_OPS_H
