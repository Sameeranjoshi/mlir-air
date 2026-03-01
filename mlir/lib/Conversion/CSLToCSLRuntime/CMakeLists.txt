# Part of the air-to-csl project.
# SPDX-License-Identifier: MIT

add_air_library(AIRCSLToCSLRuntimePass
  CSLToCSLRuntime.cpp

  DEPENDS
  AIRConversionPasses

  LINK_LIBS PUBLIC
  AIRDialect
  CSLDialect
  CSLRuntimeDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)
