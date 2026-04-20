add_mlir_dialect_library(MLIRChiplet
  ChipletDialect.cpp
  ChipletAttrs.cpp
  ChipletTypes.cpp
  ChipletOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/chiplet/Dialect

  DEPENDS
  MLIRChipletOpsIncGen
  MLIRChipletAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect
)
