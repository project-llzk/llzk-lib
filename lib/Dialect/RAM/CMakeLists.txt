add_library(LLZKRAMDialect)
add_library(LLZK::LLZKRAMDialect ALIAS LLZKRAMDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKRAMDialect)

file(GLOB_RECURSE LLZKRAMDialect_SOURCES "**/*.cpp")
target_sources(LLZKRAMDialect PRIVATE ${LLZKRAMDialect_SOURCES})
target_link_libraries(
  LLZKRAMDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRParser LLVMHeaders MLIRHeaders)
llzk_target_add_mlir_link_settings(LLZKRAMDialect)

install(TARGETS LLZKRAMDialect EXPORT LLZKTargets)
