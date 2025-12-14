add_library(LLZKZKBuilderDialect)
add_library(LLZK::LLZKZKBuilderDialect ALIAS LLZKZKBuilderDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKZKBuilderDialect)

file(GLOB_RECURSE LLZKZKBuilderDialect_SOURCES "**/*.cpp")
target_sources(LLZKZKBuilderDialect PRIVATE ${LLZKZKBuilderDialect_SOURCES})
target_link_libraries(
  LLZKZKBuilderDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils LLVMHeaders
                     MLIRHeaders LLZKZKExprDialect)
llzk_target_add_mlir_link_settings(LLZKZKBuilderDialect)

install(TARGETS LLZKZKBuilderDialect EXPORT LLZKTargets)
