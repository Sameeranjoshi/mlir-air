//===- CSLOps.cpp - CSL dialect operation implementation -------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/CSLOps.h"
#include "air/Dialect/CSL/CSLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "air/Dialect/CSL/CSLEnums.cpp.inc"
using namespace mlir;

//===----------------------------------------------------------------------===//
// TableGen-generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "air/Dialect/CSL/CSLOps.cpp.inc"
