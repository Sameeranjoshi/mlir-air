//===- AIRToCSLPass.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_TO_CSL_PASS_H
#define AIR_TO_CSL_PASS_H

#include "air/Conversion/PassDetail.h"

#include "mlir/Pass/Pass.h"

#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRToCSLPass();

} // namespace air
} // namespace xilinx

#endif // AIR_TO_CSL_PASS_H
