add_library(R1CSDialect)

file(GLOB_RECURSE R1CSDialect_SOURCES "*.cpp")
target_sources(R1CSDialect PRIVATE ${R1CSDialect_SOURCES})
target_link_libraries(
  R1CSDialect PUBLIC R1CSDialectHeaders
                     MLIRIR MLIRSupport
                     LLVMHeaders MLIRHeaders)
add_dependencies(R1CSDialect R1CSDialectIncGen)
llzk_target_add_mlir_link_settings(R1CSDialect)

install(TARGETS R1CSDialect EXPORT LLZKTargets)
