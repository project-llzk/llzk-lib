add_library(ZKLeanConversions)
add_library(ZKLean::Conversions ALIAS ZKLeanConversions)

target_sources(ZKLeanConversions PRIVATE LLZKToZKLean.cpp ZKLeanToLLZK.cpp)

target_link_libraries(
  ZKLeanConversions
  PUBLIC
    ZKLeanDialectHeaders
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    LLVMHeaders
    MLIRHeaders
  PRIVATE
    LLZKBoolDialect
    LLZKCastDialect
    LLZKFeltDialect
    ZKLeanZKExprDialect
    ZKLeanZKLeanLeanDialect
)

llzk_target_add_mlir_link_settings(ZKLeanConversions)
install(TARGETS ZKLeanConversions EXPORT LLZKTargets)
