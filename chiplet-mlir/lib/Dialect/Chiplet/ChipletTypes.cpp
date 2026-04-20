//===- ChipletTypes.cpp - Chiplet dialect types -----------------*- C++ -*-===//
#include "chiplet/Dialect/ChipletTypes.h"
#include "chiplet/Dialect/ChipletDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::chiplet;

#define GET_TYPEDEF_CLASSES
#include "chiplet/Dialect/ChipletOpsTypes.cpp.inc"

void ChipletDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "chiplet/Dialect/ChipletOpsTypes.cpp.inc"
      >();
}
