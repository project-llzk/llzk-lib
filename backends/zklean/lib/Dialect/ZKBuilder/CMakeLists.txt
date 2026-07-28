add_library(ZKLeanZKBuilderDialect)
add_library(ZKLean::ZKBuilderDialect ALIAS ZKLeanZKBuilderDialect)

file(GLOB_RECURSE ZKLeanZKBuilderDialect_SOURCES CONFIGURE_DEPENDS "**/*.cpp")
target_sources(ZKLeanZKBuilderDialect PRIVATE ${ZKLeanZKBuilderDialect_SOURCES})
target_link_libraries(
  ZKLeanZKBuilderDialect
  PUBLIC
    ZKLeanDialectHeaders
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR MLIRPass MLIRParser MLIRTransformUtils LLVMHeaders MLIRHeaders
    ZKLeanZKExprDialect)
llzk_target_add_mlir_link_settings(ZKLeanZKBuilderDialect)

install(TARGETS ZKLeanZKBuilderDialect EXPORT LLZKTargets)
