add_library(SMTTarget)
add_library(SMT::Target ALIAS SMTTarget)

file(GLOB SMTTarget_SOURCES "*.cpp")
target_sources(SMTTarget PRIVATE ${SMTTarget_SOURCES})

target_link_libraries(
  SMTTarget
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
    LLZKDialect
    SMTDialect
    LLZKBoolDialect
)

llzk_target_add_mlir_link_settings(SMTTarget)
install(TARGETS SMTTarget EXPORT LLZKTargets)
