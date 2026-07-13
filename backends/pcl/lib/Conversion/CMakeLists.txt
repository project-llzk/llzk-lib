add_library(PCLConversion)
add_library(LLZK::PCLConversion ALIAS PCLConversion)

file(GLOB PCLConversion_SOURCES "*.cpp")

target_sources(PCLConversion PRIVATE ${PCLConversion_SOURCES})

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
