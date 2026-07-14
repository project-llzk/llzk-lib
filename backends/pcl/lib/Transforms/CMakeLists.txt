add_library(PCLTransforms)
add_library(LLZK::PCLTransforms ALIAS PCLTransforms)

file(GLOB PCLTransforms_SOURCES "*.cpp")
target_sources(PCLTransforms PRIVATE ${PCLTransforms_SOURCES})
target_link_libraries(
  PCLTransforms PUBLIC PCLDialect PCLTarget MLIRFuncDialect MLIRIR
                    LLZKDialectHeaders
                    MLIRPass
                    MLIRTransforms
                    MLIRDialect
                    MLIRSupport
                    MLIRTransformUtils
                    LLVMHeaders MLIRHeaders)
add_dependencies(PCLTransforms PCLDialect PCLDialectHeaders)
llzk_target_add_mlir_link_settings(PCLTransforms)

install(TARGETS PCLTransforms EXPORT LLZKTargets)
