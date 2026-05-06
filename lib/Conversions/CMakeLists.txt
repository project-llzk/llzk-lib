add_library(LLZKConversions)
add_library(LLZK::LLZKConversions ALIAS LLZKConversions)

file(GLOB LLZKConversions_SOURCES "*.cpp")

target_sources(LLZKConversions PRIVATE ${LLZKConversions_SOURCES})

target_link_libraries(
  LLZKConversions
  PUBLIC
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
    LLVMHeaders MLIRHeaders
  PRIVATE
    LLZKDialect
    LLZKAnalysis
    LLVMDialect
    MLIRLLVMCommonConversion
)

llzk_target_add_mlir_link_settings(LLZKConversions)
install(TARGETS LLZKConversions EXPORT LLZKTargets)
