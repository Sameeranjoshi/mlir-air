# Part of the air-to-csl project.
# SPDX-License-Identifier: MIT

add_mlir_dialect(CSLRuntimeOps csl_rt)
add_mlir_doc(CSLRuntimeOps CSLRuntimeDialect ./ -gen-dialect-doc -dialect=csl_rt)
