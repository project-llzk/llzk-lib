add_library(ZKLeanZKExprDialect)
add_library(ZKLean::ZKExprDialect ALIAS ZKLeanZKExprDialect)

file(GLOB_RECURSE ZKLeanZKExprDialect_SOURCES CONFIGURE_DEPENDS "**/*.cpp")
target_sources(ZKLeanZKExprDialect PRIVATE ${ZKLeanZKExprDialect_SOURCES})
target_link_libraries(
  ZKLeanZKExprDialect
  PUBLIC
    ZKLeanDialectHeaders
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR MLIRPass MLIRParser MLIRTransformUtils LLVMHeaders MLIRHeaders
    LLZKFeltDialect)
llzk_target_add_mlir_link_settings(ZKLeanZKExprDialect)
add_dependencies(ZKLeanZKExprDialect ZKExprDialectIncGen)

install(TARGETS ZKLeanZKExprDialect EXPORT LLZKTargets)
