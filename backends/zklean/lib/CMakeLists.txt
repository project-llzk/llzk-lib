add_subdirectory(Dialect)
add_subdirectory(Conversions)
add_subdirectory(Transforms)

add_library(ZKLeanDialectRegistration)
add_library(ZKLean::DialectRegistration ALIAS ZKLeanDialectRegistration)

target_sources(ZKLeanDialectRegistration PRIVATE DialectRegistration.cpp)

target_link_libraries(
  ZKLeanDialectRegistration
  PUBLIC
    ZKLeanZKExprDialect
    ZKLeanZKBuilderDialect
    ZKLeanZKLeanLeanDialect
    ZKLeanDialectHeaders
    LLZKAllDialects
    ${LLZK_DEP_DIALECT_LIBS})
llzk_target_add_mlir_link_settings(ZKLeanDialectRegistration)

install(TARGETS ZKLeanDialectRegistration EXPORT LLZKTargets)
