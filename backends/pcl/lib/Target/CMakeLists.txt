add_library(PCLTarget)
add_library(LLZK::PCLTarget ALIAS PCLTarget)

file(GLOB_RECURSE PCLTarget_SOURCES "*.cpp")
target_sources(PCLTarget PRIVATE ${PCLTarget_SOURCES})
target_link_libraries(
  PCLTarget PUBLIC MLIRFuncDialect PCLDialect PCLDialectHeaders
                     MLIRIR MLIRSupport
                     LLVMHeaders MLIRHeaders)
llzk_target_add_mlir_link_settings(PCLTarget)

install(TARGETS PCLTarget EXPORT LLZKTargets)
