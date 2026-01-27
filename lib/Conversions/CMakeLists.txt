add_library(LLZKConversions)
add_library(LLZK::LLZKConversions ALIAS LLZKConversions)

target_sources(LLZKConversions PRIVATE LLZKToZKLean.cpp ZKLeanToLLZK.cpp)

target_link_libraries(
  LLZKConversions
  PUBLIC
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    LLVMHeaders
    MLIRHeaders
  PRIVATE
    LLZKFeltDialect
    LLZKZKExprDialect
    LLZKZKLeanStructDialect
)

llzk_target_add_mlir_link_settings(LLZKConversions)
install(TARGETS LLZKConversions EXPORT LLZKTargets)
