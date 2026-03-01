//===- InitAll.cpp ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/InitAll.h"

#include "air/Conversion/Passes.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIR/AIRTransformOps.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "air/Dialect/CSLRuntime/CSLRuntimeDialect.h"
#include "air/Dialect/CSLRuntime/CSLRuntimeOps.h"
#include "air/Interfaces/AIRInterfaces.h"
#include "air/Transform/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllPasses.h"

void xilinx::air::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<xilinx::air::airDialect, xilinx::airrt::AIRRtDialect,
                  xilinx::csl::CSLDialect, xilinx::csl_rt::CSLRuntimeDialect>();
  xilinx::air::registerTransformDialectExtension(registry);
  xilinx::air::registerCodegenInterfaces(registry);
}

void xilinx::air::registerAllPasses() {
  xilinx::air::registerTransformPasses();
  xilinx::air::registerConversionPasses();
}
