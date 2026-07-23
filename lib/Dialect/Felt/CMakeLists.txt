add_library(LLZKFeltDialect)
add_library(LLZK::LLZKFeltDialect ALIAS LLZKFeltDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKFeltDialect)

file(GLOB_RECURSE LLZKFeltDialect_SOURCES "**/*.cpp")
target_sources(LLZKFeltDialect PRIVATE ${LLZKFeltDialect_SOURCES})
target_link_libraries(
  LLZKFeltDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
                     LLVMHeaders MLIRHeaders LLZKUtil)
llzk_target_add_mlir_link_settings(LLZKFeltDialect)

install(TARGETS LLZKFeltDialect EXPORT LLZKTargets)
