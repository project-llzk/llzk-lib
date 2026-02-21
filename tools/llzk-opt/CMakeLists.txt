add_executable(llzk-opt llzk-opt.cpp)
target_link_libraries(llzk-opt PRIVATE ${LLZK_DEP_DIALECT_LIBS} MLIROptLib
		      LLZK::DialectRegistration R1CS::DialectRegistration
		      LLZKAnalysis LLZKTransforms LLZKConversions 
		      R1CSTransforms LLZKUtil LLZKValidators 
		      LLZKDialectHeaders R1CSDialectHeaders MLIRIR 
		      $<$<BOOL:${LLZK_WITH_PCL_BOOL}>:PCL::PCLDialect> 
		      $<$<BOOL:${LLZK_WITH_PCL_BOOL}>:PCL::PCLTransforms> 
		      ZKLean::DialectRegistration ZKLeanDialectHeaders 
		      ZKLeanTransforms ZKLean::Conversions)
llzk_target_add_mlir_link_settings(llzk-opt)
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} ${R1CS_INCLUDE_DIR}
                    ${ZKLEAN_INCLUDE_DIR})

set_target_properties(llzk-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                          ${CMAKE_BINARY_DIR}/bin)
install(
  TARGETS llzk-opt
  EXPORT LLZKTargets
  COMPONENT devtools)
