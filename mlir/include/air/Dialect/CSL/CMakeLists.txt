# Part of the air-to-csl project.
# SPDX-License-Identifier: MIT

add_mlir_dialect(CSLOps csl)
add_mlir_doc(CSLOps CSLDialect ./ -gen-dialect-doc -dialect=csl)

set(LLVM_TARGET_DEFINITIONS CSLBase.td)
mlir_tablegen(CSLEnums.h.inc -gen-enum-decls)
mlir_tablegen(CSLEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRCSLEnumsIncGen)
add_dependencies(csl-headers MLIRCSLEnumsIncGen)
