configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CHIPLET_TEST_DEPENDS
  FileCheck count not
  chiplet-opt
)

add_lit_testsuite(check-chiplet "Running the chiplet regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CHIPLET_TEST_DEPENDS}
)
set_target_properties(check-chiplet PROPERTIES FOLDER "Tests")

add_lit_testsuites(CHIPLET ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${CHIPLET_TEST_DEPENDS})
