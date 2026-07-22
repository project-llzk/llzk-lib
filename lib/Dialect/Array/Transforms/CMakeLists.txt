add_library(LLZKArrayTransforms)
add_library(LLZK::LLZKArrayTransforms ALIAS LLZKArrayTransforms)

target_sources(LLZKArrayTransforms PRIVATE ArrayToScalarPass.cpp)
target_link_libraries(
  LLZKArrayTransforms
  PUBLIC
    LLZKArrayDialect
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
llzk_target_add_mlir_link_settings(LLZKArrayTransforms)

install(TARGETS LLZKArrayTransforms EXPORT LLZKTargets)
