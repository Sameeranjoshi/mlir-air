//===- ChipletAttrs.cpp - Chiplet dialect attrs -----------------*- C++ -*-===//
#include "chiplet/Dialect/ChipletAttrs.h"
#include "chiplet/Dialect/ChipletDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::chiplet;

#include "chiplet/Dialect/ChipletEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "chiplet/Dialect/ChipletAttrs.cpp.inc"

void ChipletDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "chiplet/Dialect/ChipletAttrs.cpp.inc"
      >();
}
