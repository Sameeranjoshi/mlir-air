//===- ChipletOps.h - Chiplet dialect ops -----------------------*- C++ -*-===//
#ifndef CHIPLET_OPS_H
#define CHIPLET_OPS_H

#include "chiplet/Dialect/ChipletAttrs.h"
#include "chiplet/Dialect/ChipletDialect.h"
#include "chiplet/Dialect/ChipletTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "chiplet/Dialect/ChipletOps.h.inc"

#endif // CHIPLET_OPS_H
