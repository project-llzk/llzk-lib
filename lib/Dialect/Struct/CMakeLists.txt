add_library(LLZKStructDialect)
add_library(LLZK::LLZKStructDialect ALIAS LLZKStructDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKStructDialect)

file(GLOB_RECURSE LLZKStructDialect_SOURCES "**/*.cpp")
target_sources(LLZKStructDialect PRIVATE ${LLZKStructDialect_SOURCES})
target_link_libraries(
  LLZKStructDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
                     LLVMHeaders MLIRHeaders LLZKAnalysis LLZKUtil LLZKDialect)
llzk_target_add_mlir_link_settings(LLZKStructDialect)

install(TARGETS LLZKStructDialect EXPORT LLZKTargets)
