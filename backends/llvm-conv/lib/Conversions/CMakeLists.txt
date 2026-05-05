add_library(LLVMConversions)

target_sources(LLVMConversions PRIVATE LLVMLoweringPass.cpp)

target_link_libraries(
  LLVMConversions
  PUBLIC
    LLVMConvHeaders
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    LLVMHeaders
    MLIRHeaders
  PRIVATE
    LLZKDialect
)


llzk_target_add_mlir_link_settings(LLVMConversions)
install(TARGETS LLVMConversions EXPORT LLZKTargets)
