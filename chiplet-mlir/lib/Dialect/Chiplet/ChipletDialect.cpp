//===- ChipletDialect.cpp - Chiplet dialect ---------------------*- C++ -*-===//
#include "chiplet/Dialect/ChipletDialect.h"
#include "chiplet/Dialect/ChipletAttrs.h"
#include "chiplet/Dialect/ChipletOps.h"
#include "chiplet/Dialect/ChipletTypes.h"

using namespace mlir;
using namespace mlir::chiplet;

#include "chiplet/Dialect/ChipletOpsDialect.cpp.inc"

void ChipletDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "chiplet/Dialect/ChipletOps.cpp.inc"
      >();
  registerTypes();
  registerAttributes();
}
