add_library(SMTTransforms)
add_library(SMT::Transforms ALIAS SMTTransforms)

file(GLOB SMTTransforms_SOURCES "*.cpp")
target_sources(SMTTransforms PRIVATE ${SMTTransforms_SOURCES})

target_link_libraries(
  SMTTransforms
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

llzk_target_add_mlir_link_settings(SMTTransforms)
install(TARGETS SMTTransforms EXPORT LLZKTargets)
