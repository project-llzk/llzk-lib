add_library(LLZKZKLeanStructDialect)
add_library(LLZK::LLZKZKLeanStructDialect ALIAS LLZKZKLeanStructDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKZKLeanStructDialect)

file(GLOB_RECURSE LLZKZKLeanStructDialect_SOURCES "**/*.cpp")
target_sources(LLZKZKLeanStructDialect PRIVATE ${LLZKZKLeanStructDialect_SOURCES})
target_link_libraries(
  LLZKZKLeanStructDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS}
                           MLIRIR MLIRPass MLIRParser MLIRTransformUtils
                           LLVMHeaders MLIRHeaders LLZKZKExprDialect)
llzk_target_add_mlir_link_settings(LLZKZKLeanStructDialect)
add_dependencies(LLZKZKLeanStructDialect ZKLeanStructDialectIncGen)

install(TARGETS LLZKZKLeanStructDialect EXPORT LLZKTargets)
