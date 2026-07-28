add_library(LLZKFunctionDialect)
add_library(LLZK::LLZKFunctionDialect ALIAS LLZKFunctionDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKFunctionDialect)

file(GLOB_RECURSE LLZKFunctionDialect_SOURCES "**/*.cpp")
target_sources(LLZKFunctionDialect PRIVATE ${LLZKFunctionDialect_SOURCES})
target_link_libraries(
  LLZKFunctionDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
                     LLVMHeaders MLIRHeaders LLZKPolymorphicDialect)
llzk_target_add_mlir_link_settings(LLZKFunctionDialect)

install(TARGETS LLZKFunctionDialect EXPORT LLZKTargets)
