add_library(LLZKVerifDialect)
add_library(LLZK::LLZKVerifDialect ALIAS LLZKVerifDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKVerifDialect)

file(GLOB_RECURSE LLZKVerifDialect_SOURCES "**/*.cpp")
target_sources(LLZKVerifDialect PRIVATE ${LLZKVerifDialect_SOURCES})
target_link_libraries(
  LLZKVerifDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
                     LLVMHeaders MLIRHeaders LLZKDialect)
llzk_target_add_mlir_link_settings(LLZKVerifDialect)

install(TARGETS LLZKVerifDialect EXPORT LLZKTargets)
