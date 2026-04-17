# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

if(AIR_ENABLE_AIE AND NOT AIE_BINARY_DIR)
  find_package(AIE REQUIRED CONFIG)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(TEST_DEPENDS FileCheck count not air-opt air-runner air-translate)

add_lit_testsuite(check-air-mlir "Running the air mlir regression tests"
                  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${TEST_DEPENDS})
set_target_properties(check-air-mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(AIRMLIR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TEST_DEPENDS})

#===----------------------------------------------------------------------===//
# CSL Test Suites
#
#   ninja check-csl              — all CSL tests
#   ninja check-csl-dialect      — Dialect/CSL/         (op round-trip, diagnostics)
#   ninja check-csl-conversion   — Conversion/AIRToCSL/ (-air-to-csl, derive-exports, e2e)
#   ninja check-csl-emitters     — Targets/CSLEmit/      (--emit-csl-program/layout/host)
#===----------------------------------------------------------------------===//

# Locate the lit runner — use LLVM_EXTERNAL_LIT (set by LLVM build) with -v for
# verbose output (shows individual test names instead of progress dots).
set(CSL_LIT_COMMAND ${LLVM_EXTERNAL_LIT} -v --timeout 30)

# 1. CSL dialect tests (csl.*, csl_layout.*, csl_host.* ops)
add_custom_target(check-csl-dialect
  COMMAND ${CSL_LIT_COMMAND} ${CMAKE_CURRENT_BINARY_DIR}/Dialect/CSL
  DEPENDS ${TEST_DEPENDS}
  COMMENT "Running CSL dialect tests")
set_target_properties(check-csl-dialect PROPERTIES FOLDER "Tests/CSL")

# 2. AIR → CSL conversion tests (-air-to-csl, -csl-infer-exports, e2e pipeline)
add_custom_target(check-csl-conversion
  COMMAND ${CSL_LIT_COMMAND} ${CMAKE_CURRENT_BINARY_DIR}/Conversion/AIRToCSL
  DEPENDS ${TEST_DEPENDS}
  COMMENT "Running AIR to CSL conversion tests")
set_target_properties(check-csl-conversion PROPERTIES FOLDER "Tests/CSL")

# 3. CSL emitter tests (--emit-csl-program, --emit-csl-layout, --emit-csl-host)
add_custom_target(check-csl-emitters
  COMMAND ${CSL_LIT_COMMAND} ${CMAKE_CURRENT_BINARY_DIR}/Targets/CSLEmit
  DEPENDS ${TEST_DEPENDS}
  COMMENT "Running CSL emitter tests")
set_target_properties(check-csl-emitters PROPERTIES FOLDER "Tests/CSL")

# Aggregate: check-csl runs all three suites
add_custom_target(check-csl
  COMMAND ${CSL_LIT_COMMAND}
    ${CMAKE_CURRENT_BINARY_DIR}/Dialect/CSL
    ${CMAKE_CURRENT_BINARY_DIR}/Conversion/AIRToCSL
    ${CMAKE_CURRENT_BINARY_DIR}/Targets/CSLEmit
  DEPENDS ${TEST_DEPENDS}
  COMMENT "Running all CSL tests")
set_target_properties(check-csl PROPERTIES FOLDER "Tests/CSL")

add_dependencies(check-all check-air-mlir check-csl-dialect)

add_subdirectory(CppUnitTests)
add_subdirectory(Targets)
