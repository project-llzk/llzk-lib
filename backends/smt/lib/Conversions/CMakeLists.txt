add_library(SMTConversions)

target_sources(SMTConversions PRIVATE SMTLoweringPass.cpp)

target_link_libraries(
  SMTConversions
  PUBLIC
    SMTDialectHeaders
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
)


llzk_target_add_mlir_link_settings(SMTConversions)
install(TARGETS SMTConversions EXPORT LLZKTargets)
