add_library(PCLTransforms)
add_library(LLZK::PCLTransforms ALIAS PCLTransforms)

target_sources(PCLTransforms PRIVATE TrimExprSizePass.cpp)
target_link_libraries(
  PCLTransforms
  PUBLIC
    PCLDialect
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR
    MLIRPass
    MLIRParser
    MLIRTransformUtils
    MLIRSCFTransforms
    LLVMHeaders
    MLIRHeaders
    LLZKDialect
    LLZKDialectShared
)
llzk_target_add_mlir_link_settings(PCLTransforms)

install(TARGETS PCLTransforms EXPORT LLZKTargets)
