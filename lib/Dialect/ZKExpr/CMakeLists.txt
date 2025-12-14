add_library(LLZKZKExprDialect)
add_library(LLZK::LLZKZKExprDialect ALIAS LLZKZKExprDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKZKExprDialect)

file(GLOB_RECURSE LLZKZKExprDialect_SOURCES "**/*.cpp")
target_sources(LLZKZKExprDialect PRIVATE ${LLZKZKExprDialect_SOURCES})
target_link_libraries(
  LLZKZKExprDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils LLVMHeaders
                     MLIRHeaders LLZKFeltDialect)
llzk_target_add_mlir_link_settings(LLZKZKExprDialect)

install(TARGETS LLZKZKExprDialect EXPORT LLZKTargets)
