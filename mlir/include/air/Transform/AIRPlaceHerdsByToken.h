//===- AIRPlaceHerdsByToken.h -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
#ifndef AIR_PLACE_HERDS_BY_TOKEN_H
#define AIR_PLACE_HERDS_BY_TOKEN_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRPlaceHerdsByTokenPass();
std::unique_ptr<mlir::Pass>
createAIRPlaceHerdsByTokenPass(const AIRPlaceHerdsByTokenOptions &options);

} // namespace air
} // namespace xilinx

#endif // AIR_PLACE_HERDS_BY_TOKEN_H
