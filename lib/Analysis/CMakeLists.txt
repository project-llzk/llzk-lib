add_library(LLZKAnalysis)
add_library(LLZK::LLZKAnalysis ALIAS LLZKAnalysis)

file(GLOB LLZKAnalysis_SOURCES "*.cpp")
target_sources(LLZKAnalysis PRIVATE ${LLZKAnalysis_SOURCES})
target_link_libraries(
  LLZKAnalysis PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRAnalysis
                      MLIRIR MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
                      LLVMHeaders MLIRHeaders
                      LLZKUtil LLZKDialect)
llzk_target_add_mlir_link_settings(LLZKAnalysis)

llzk_check_cxx_feature(LLZKAnalysis nontype_template_args)
install(TARGETS LLZKAnalysis EXPORT LLZKTargets)
