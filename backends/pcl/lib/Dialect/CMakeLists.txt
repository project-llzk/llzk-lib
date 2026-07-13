add_library(PCLDialect)
add_library(LLZK::PCLDialect ALIAS PCLDialect)

file(GLOB_RECURSE PCLDialect_SOURCES "*.cpp")
target_sources(PCLDialect PRIVATE ${PCLDialect_SOURCES})
target_link_libraries(
  PCLDialect PUBLIC MLIRFuncDialect PCLDialectHeaders
                     MLIRIR MLIRSupport
                     LLVMHeaders MLIRHeaders)
add_dependencies(PCLDialect PCLDialectIncGen)
llzk_target_add_mlir_link_settings(PCLDialect)

install(TARGETS PCLDialect EXPORT LLZKTargets)
