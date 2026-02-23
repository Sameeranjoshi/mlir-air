# Part of the air-to-csl project.
# SPDX-License-Identifier: MIT

add_mlir_dialect_library(
  CSLDialect
  CSLDialect.cpp
  CSLOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/air/Dialect/CSL

  DEPENDS
  MLIRCSLOpsIncGen
  MLIRCSLEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRIR)
