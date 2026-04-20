//===- ChipletOps.cpp - Chiplet dialect ops ---------------------*- C++ -*-===//
#include "chiplet/Dialect/ChipletOps.h"
#include "chiplet/Dialect/ChipletDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::chiplet;

//===----------------------------------------------------------------------===//
// Trivial verifiers: partition_id / worker_id must appear lexically inside
// the appropriate region kind. Walk parent chain; reject if no match.
//===----------------------------------------------------------------------===//

LogicalResult PartitionIdOp::verify() {
  Operation *parent = (*this)->getParentOp();
  while (parent) {
    if (isa<LaunchOp>(parent))
      return success();
    parent = parent->getParentOp();
  }
  return emitOpError(
      "must appear lexically inside a 'chiplet.launch' region");
}

LogicalResult WorkerIdOp::verify() {
  Operation *parent = (*this)->getParentOp();
  while (parent) {
    if (isa<TaskOp>(parent))
      return success();
    parent = parent->getParentOp();
  }
  return emitOpError(
      "must appear lexically inside a 'chiplet.task' region");
}

#define GET_OP_CLASSES
#include "chiplet/Dialect/ChipletOps.cpp.inc"
