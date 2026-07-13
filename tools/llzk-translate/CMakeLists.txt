get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_executable(llzk-translate llzk-translate.cpp)
target_link_libraries(llzk-translate PRIVATE ${LLZK_DEP_DIALECT_LIBS} MLIRTranslateLib
                      LLZKDialect
                      LLZKDialectHeaders MLIRIR 
                       SMTTarget 
                      ZKLeanTarget
                      ${translation_libs}
                      # $<$<BOOL:${LLZK_WITH_PCL_BOOL}>:PCLConvTransforms>
                      )
llzk_target_add_mlir_link_settings(llzk-translate)
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} ${R1CS_INCLUDE_DIR}
                    ${ZKLEAN_INCLUDE_DIR})

set_target_properties(llzk-translate PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                          ${CMAKE_BINARY_DIR}/bin)
install(
  TARGETS llzk-translate
  EXPORT LLZKTargets
  COMPONENT devtools)
