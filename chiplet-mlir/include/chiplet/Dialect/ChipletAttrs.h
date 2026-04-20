//===- ChipletAttrs.h - Chiplet dialect attrs -------------------*- C++ -*-===//
#ifndef CHIPLET_ATTRS_H
#define CHIPLET_ATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

#include "chiplet/Dialect/ChipletEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "chiplet/Dialect/ChipletAttrs.h.inc"

#endif // CHIPLET_ATTRS_H
