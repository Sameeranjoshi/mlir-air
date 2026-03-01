//===- CSLRuntimeOps.cpp - CSL Runtime operations implementation -*-C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSLRuntime/CSLRuntimeOps.h"
#include "air/Dialect/CSLRuntime/CSLRuntimeDialect.h"

using namespace xilinx::csl_rt;

#define GET_OP_CLASSES
#include "air/Dialect/CSLRuntime/CSLRuntimeOps.cpp.inc"
