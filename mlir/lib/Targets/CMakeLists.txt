# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_mlir_library(
  AIRTargets
  AIRTargets.cpp
  AIRHerdToJSON.cpp
  AIRToSpada.cpp

  LINK_LIBS PUBLIC
  AIRRtDialect
  AIRDialect
  AIRUtil
  MLIRIR
  MLIRSupport
  MLIRTransforms)
