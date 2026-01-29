add_library(LLZKZKLeanLeanDialect)
add_library(LLZK::LLZKZKLeanLeanDialect ALIAS LLZKZKLeanLeanDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKZKLeanLeanDialect)

file(GLOB_RECURSE LLZKZKLeanLeanDialect_SOURCES "**/*.cpp")
target_sources(LLZKZKLeanLeanDialect PRIVATE ${LLZKZKLeanLeanDialect_SOURCES})
target_link_libraries(
  LLZKZKLeanLeanDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS}
                           MLIRIR MLIRPass MLIRParser MLIRTransformUtils
                           LLVMHeaders MLIRHeaders LLZKZKExprDialect)
llzk_target_add_mlir_link_settings(LLZKZKLeanLeanDialect)
add_dependencies(LLZKZKLeanLeanDialect ZKLeanLeanDialectIncGen)

install(TARGETS LLZKZKLeanLeanDialect EXPORT LLZKTargets)
