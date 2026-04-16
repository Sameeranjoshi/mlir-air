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
# CSL Dialect Test Suites — Simplified Naming
#===----------------------------------------------------------------------===//

# CSL Dialect tests (csl.* operations, parsing, round-trip)
add_lit_testsuite(check-csld "Running CSL dialect tests"
                  ${CMAKE_CURRENT_BINARY_DIR}/Dialect/CSL
                  DEPENDS ${TEST_DEPENDS})
set_target_properties(check-csld PROPERTIES FOLDER "Tests/CSL")

# AIR to CSL conversion tests (V2 pipeline)
add_lit_testsuite(check-air-to-csl "Running AIR to CSL conversion tests"
                  ${CMAKE_CURRENT_BINARY_DIR}/Conversion/AIRToCSL
                  DEPENDS ${TEST_DEPENDS})
set_target_properties(check-air-to-csl PROPERTIES FOLDER "Tests/CSL")

# Convenience alias: check-csl runs all CSL tests
add_custom_target(check-csl DEPENDS
  check-csld
  check-air-to-csl
)
set_target_properties(check-csl PROPERTIES FOLDER "Tests/CSL")

# Legacy alias for compatibility
add_custom_target(check-csl-all DEPENDS check-csl)
set_target_properties(check-csl-all PROPERTIES FOLDER "Tests/CSL")

add_dependencies(check-all check-air-mlir check-csld)

add_subdirectory(CppUnitTests)
add_subdirectory(Targets)
