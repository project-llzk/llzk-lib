add_library(PCLConversion)
add_library(LLZK::PCLConversion ALIAS PCLConversion)


target_sources(PCLConversion PRIVATE 
  CAPI.cpp
  PCLLoweringPass.cpp 
  PCLLoweringPass/UsedFreeFunctions.cpp
  PCLLoweringPass/Patterns.cpp
  PCLLoweringPass/Modes.cpp
  PCLLoweringPass/TypeConverter.cpp
  PCLLoweringPass/Temporary.cpp
)

target_link_libraries(
  PCLConversion
  PUBLIC
    PCLDialect
    LLZKDialectHeaders
    PCLDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
    LLVMHeaders MLIRHeaders
  PRIVATE
    LLZKDialect
    LLZKAnalysis
)

llzk_target_add_mlir_link_settings(PCLConversion)
install(TARGETS PCLConversion EXPORT LLZKTargets)
