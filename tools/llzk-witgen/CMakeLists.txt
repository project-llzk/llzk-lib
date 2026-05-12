add_library(LLZKWitgenTool STATIC)
target_sources(
  LLZKWitgenTool
  PRIVATE
    Interpreter.cpp
    JSON.cpp
    ValueModel.cpp
    WitgenDriver.cpp
)
target_include_directories(LLZKWitgenTool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  LLZKWitgenTool
  PUBLIC
    LLVMHeaders
    MLIRHeaders
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRArithDialect
    MLIRIR
    MLIRParser
    MLIRSCFDialect
  PRIVATE
    LLZK::DialectRegistration
    LLZK::LLZKUtil
    LLZKIncludeDialect
    LLZKTransforms
)
llzk_target_add_mlir_link_settings(LLZKWitgenTool)

add_executable(llzk-witgen llzk-witgen.cpp)
target_link_libraries(llzk-witgen PRIVATE LLZKWitgenTool)
llzk_target_add_mlir_link_settings(llzk-witgen)

set_target_properties(llzk-witgen PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

install(
  TARGETS llzk-witgen
  EXPORT LLZKTargets
  COMPONENT devtools)
