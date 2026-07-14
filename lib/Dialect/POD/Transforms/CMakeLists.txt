
add_library(LLZKPODTransforms)
add_library(LLZK::LLZKPODTransforms ALIAS LLZKPODTransforms)
target_link_libraries(LLZKAllDialects INTERFACE LLZKPODTransforms)

target_sources(LLZKPODTransforms PRIVATE
  PodToScalarPass.cpp
)
target_link_libraries(
  LLZKPODTransforms PUBLIC LLZKPODDialect LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
                     LLVMHeaders MLIRHeaders LLZKDialect LLZKArrayDialect)
llzk_target_add_mlir_link_settings(LLZKPODTransforms)

install(TARGETS LLZKPODTransforms EXPORT LLZKTargets)
