# Part of the air-to-csl project.
# SPDX-License-Identifier: MIT

add_mlir_dialect_library(
  CSLRuntimeDialect
  CSLRuntimeDialect.cpp
  CSLRuntimeOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/air/Dialect/CSLRuntime

  DEPENDS
  MLIRCSLRuntimeOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR)
