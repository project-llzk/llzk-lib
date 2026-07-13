add_library(LLZKArrayDialect)
add_library(LLZK::LLZKArrayDialect ALIAS LLZKArrayDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKArrayDialect)

file(GLOB_RECURSE LLZKArrayDialect_SOURCES "**/*.cpp")
target_sources(LLZKArrayDialect PRIVATE ${LLZKArrayDialect_SOURCES})
target_link_libraries(
  LLZKArrayDialect
  PUBLIC
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR
    MLIRPass
    MLIRParser
    MLIRTransformUtils
    MLIRSCFTransforms
    LLVMHeaders
    MLIRHeaders
    LLZKDialect
    LLZKDialectShared
)
llzk_target_add_mlir_link_settings(LLZKArrayDialect)

install(TARGETS LLZKArrayDialect EXPORT LLZKTargets)
