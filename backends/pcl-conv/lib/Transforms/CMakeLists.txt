add_library(PCLConvTransforms)
add_library(PCLConv::Transforms ALIAS PCLConvTransforms)

file(GLOB PCLConvTransforms_SOURCES "*.cpp")

target_sources(PCLConvTransforms PRIVATE ${PCLConvTransforms_SOURCES})

target_link_libraries(
  PCLConvTransforms
  PUBLIC
    PCL::PCLDialect
    LLZKDialectHeaders
    PCLConvDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
    LLVMHeaders MLIRHeaders
  PRIVATE
    LLZKDialect
    LLZKAnalysis
)

llzk_target_add_mlir_link_settings(PCLConvTransforms)
install(TARGETS PCLConvTransforms EXPORT LLZKTargets)
