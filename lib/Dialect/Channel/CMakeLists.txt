add_library(LLZKChannelDialect)
add_library(LLZK::LLZKChannelDialect ALIAS LLZKChannelDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKChannelDialect)

file(GLOB_RECURSE LLZKChannelDialect_SOURCES "**/*.cpp")
target_sources(LLZKChannelDialect PRIVATE ${LLZKChannelDialect_SOURCES})
target_link_libraries(
  LLZKChannelDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils
                     LLVMHeaders MLIRHeaders)
llzk_target_add_mlir_link_settings(LLZKChannelDialect)

install(TARGETS LLZKChannelDialect EXPORT LLZKTargets)
