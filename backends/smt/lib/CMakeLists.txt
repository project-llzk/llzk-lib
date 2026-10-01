add_subdirectory(Conversions)
add_subdirectory(Target)

add_library(SMTDialectRegistration)
add_library(SMT::DialectRegistration ALIAS SMTDialectRegistration)
target_sources(SMTDialectRegistration PRIVATE DialectRegistration.cpp)
target_link_libraries(
  SMTDialectRegistration
  PUBLIC SMTConversions SMTDialectHeaders MLIRSMT
  PRIVATE MLIRFuncInlinerExtension)
llzk_target_add_mlir_link_settings(SMTDialectRegistration)

install(TARGETS SMTDialectRegistration EXPORT LLZKTargets)
