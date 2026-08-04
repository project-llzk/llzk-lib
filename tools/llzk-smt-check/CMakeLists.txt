add_executable(llzk-smt-check llzk-smt-check.cpp)
target_link_libraries(
  llzk-smt-check
  PRIVATE
    LLVMHeaders
    LLVMSupport
)
llzk_target_add_mlir_link_settings(llzk-smt-check)

set_target_properties(llzk-smt-check PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

install(
  TARGETS llzk-smt-check
  EXPORT LLZKTargets
  COMPONENT devtools)
