# Part of the air-to-csl project.
# SPDX-License-Identifier: MIT

add_mlir_library(
  CSLTransforms
  CSLInferExports.cpp
  CSLAutoVectorize.cpp
  LoopIdiomAnalysis.cpp
  Passes.cpp
  Patterns/ElementwisePatterns.cpp
  Patterns/FmaPattern.cpp
  Patterns/MovePatterns.cpp
  Patterns/Rank2Patterns.cpp
  Patterns/ScalarBroadcastPatterns.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/air/Dialect/CSL/Transforms

  LINK_LIBS PUBLIC
  CSLDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRArithDialect)
