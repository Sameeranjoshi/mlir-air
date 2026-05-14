# Part of the air-to-csl project.
# SPDX-License-Identifier: MIT

add_mlir_library(
  CSLPipelines
  CSLPipelines.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/air/Dialect/CSL/Pipelines

  LINK_LIBS PUBLIC
  CSLTransforms
  MLIRIR
  MLIRPass
  MLIRSupport)
