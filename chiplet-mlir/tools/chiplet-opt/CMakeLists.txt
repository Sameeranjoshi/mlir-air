get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRArithDialect
  MLIRFuncDialect
  MLIROptLib
  MLIRChiplet
)

add_llvm_executable(chiplet-opt chiplet-opt.cpp)
llvm_update_compile_flags(chiplet-opt)
target_link_libraries(chiplet-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(chiplet-opt)
