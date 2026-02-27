add_library(ZKLeanZKLeanLeanDialect)
add_library(ZKLean::ZKLeanLeanDialect ALIAS ZKLeanZKLeanLeanDialect)

file(GLOB_RECURSE ZKLeanZKLeanLeanDialect_SOURCES CONFIGURE_DEPENDS "**/*.cpp")
target_sources(ZKLeanZKLeanLeanDialect PRIVATE ${ZKLeanZKLeanLeanDialect_SOURCES})
target_link_libraries(
  ZKLeanZKLeanLeanDialect
  PUBLIC
    ZKLeanDialectHeaders
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR MLIRPass MLIRParser MLIRTransformUtils LLVMHeaders MLIRHeaders
    ZKLeanZKExprDialect)
llzk_target_add_mlir_link_settings(ZKLeanZKLeanLeanDialect)
add_dependencies(ZKLeanZKLeanLeanDialect ZKLeanLeanDialectIncGen)

install(TARGETS ZKLeanZKLeanLeanDialect EXPORT LLZKTargets)
