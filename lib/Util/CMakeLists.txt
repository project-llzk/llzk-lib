add_library(LLZKUtil)
add_library(LLZK::LLZKUtil ALIAS LLZKUtil)

file(GLOB_RECURSE LLZKUtil_SOURCES "*.cpp")
target_sources(LLZKUtil PRIVATE ${LLZKUtil_SOURCES})
target_link_libraries(
  LLZKUtil
  PUBLIC
    LLVMHeaders
    MLIRHeaders
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR
  PRIVATE
    LLZKIncludeDialect
    LLZKPODDialect
)
add_dependencies(LLZKUtil LLZKDialect)
llzk_target_add_mlir_link_settings(LLZKUtil)

install(TARGETS LLZKUtil EXPORT LLZKTargets)
