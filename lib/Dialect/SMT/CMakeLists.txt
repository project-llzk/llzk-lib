add_library(SMTDialect)
target_link_libraries(LLZKAllDialects INTERFACE SMTDialect)

file(GLOB_RECURSE SMTDialect_SOURCES CONFIGURE_DEPENDS "*.cpp")
target_sources(SMTDialect PRIVATE ${SMTDialect_SOURCES})
target_link_libraries(
  SMTDialect
  PUBLIC
    SMTDialectHeaders
    LLZKDialectHeaders
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
    MLIRControlFlowInterfaces
    LLVMHeaders
    MLIRHeaders)
llzk_target_add_mlir_link_settings(SMTDialect)
add_dependencies(SMTDialect SMTDialectIncGen)

install(TARGETS SMTDialect EXPORT LLZKTargets)
