add_library(LLZKPolymorphicDialect)
add_library(LLZK::LLZKPolymorphicDialect ALIAS LLZKPolymorphicDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKPolymorphicDialect)

file(GLOB_RECURSE LLZKPolymorphicDialect_SOURCES "**/*.cpp")
target_sources(LLZKPolymorphicDialect PRIVATE ${LLZKPolymorphicDialect_SOURCES})
target_link_libraries(
  LLZKPolymorphicDialect
  PUBLIC
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR
    MLIRParser
    LLVMHeaders
    MLIRHeaders
  PRIVATE
    LLZKTransforms)
llzk_target_add_mlir_link_settings(LLZKPolymorphicDialect)

install(TARGETS LLZKPolymorphicDialect EXPORT LLZKTargets)
