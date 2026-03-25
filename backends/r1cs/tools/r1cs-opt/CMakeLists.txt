add_executable(r1cs-opt r1cs-opt.cpp)
target_link_libraries(r1cs-opt PRIVATE
  ${LLZK_DEP_DIALECT_LIBS}
  MLIROptLib
  LLZK::DialectRegistration
  R1CS::DialectRegistration
  LLZKDialectHeaders
  R1CSDialectHeaders
  MLIRIR
  LLZKUtil
  LLZKAnalysis
  LLZKValidators
  LLZKTransforms
  R1CSTransforms
  $<$<BOOL:${LLZK_WITH_PCL_BOOL}>:PCL::PCLDialect>
  $<$<BOOL:${LLZK_WITH_PCL_BOOL}>:PCL::PCLTransforms>
)
llzk_target_add_mlir_link_settings(r1cs-opt)
set_target_properties(r1cs-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                          ${CMAKE_BINARY_DIR}/bin)

install(
  TARGETS r1cs-opt
  EXPORT LLZKTargets
  COMPONENT devtools)
