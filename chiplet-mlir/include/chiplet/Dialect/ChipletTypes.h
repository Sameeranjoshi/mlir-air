//===- ChipletTypes.h - Chiplet dialect types -------------------*- C++ -*-===//
#ifndef CHIPLET_TYPES_H
#define CHIPLET_TYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "chiplet/Dialect/ChipletOpsTypes.h.inc"

#endif // CHIPLET_TYPES_H
