add_library(LLZKBoolDialect)
add_library(LLZK::LLZKBoolDialect ALIAS LLZKBoolDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKBoolDialect)

file(GLOB_RECURSE LLZKBoolDialect_SOURCES "**/*.cpp")
target_sources(LLZKBoolDialect PRIVATE ${LLZKBoolDialect_SOURCES})
target_link_libraries(
  LLZKBoolDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
                     MLIRArithDialect LLVMHeaders MLIRHeaders)
llzk_target_add_mlir_link_settings(LLZKBoolDialect)

install(TARGETS LLZKBoolDialect EXPORT LLZKTargets)
