add_executable(r1cs-opt r1cs-opt.cpp)
target_link_libraries(r1cs-opt PRIVATE
  LLVMHeaders
  MLIRHeaders
  MLIRIR
  MLIRParser
  MLIROptLib
  MLIRPass
  MLIRTransforms
  R1CSDialect
  R1CS::DialectRegistration
)
llzk_target_add_mlir_link_settings(r1cs-opt)
set_target_properties(r1cs-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                          ${CMAKE_BINARY_DIR}/bin)
