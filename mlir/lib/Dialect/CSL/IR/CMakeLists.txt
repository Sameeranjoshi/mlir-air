# Part of the air-to-csl project.
# SPDX-License-Identifier: MIT

add_mlir_dialect_library(
  CSLDialect
  CSLDialect.cpp
  CSLOps.cpp
  CSLLayoutDialect.cpp
  CSLLayoutOps.cpp
  CSLHostDialect.cpp
  CSLHostOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/air/Dialect/CSL

  DEPENDS
  MLIRCSLOpsIncGen
  MLIRCSLEnumsIncGen
  MLIRCSLLayoutOpsIncGen
  MLIRCSLHostOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR)
