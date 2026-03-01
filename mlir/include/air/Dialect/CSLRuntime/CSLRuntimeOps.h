//===- CSLRuntimeOps.h - CSL Runtime operations declaration -----*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef CSLRUNTIME_OPS_H
#define CSLRUNTIME_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "air/Dialect/CSLRuntime/CSLRuntimeDialect.h"

#define GET_OP_CLASSES
#include "air/Dialect/CSLRuntime/CSLRuntimeOps.h.inc"

#endif // CSLRUNTIME_OPS_H
