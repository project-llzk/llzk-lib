file(GLOB_RECURSE LLZK_WITGEN_TEST_SOURCES "*.cpp")
add_executable(LLZK_Witgen_Tests ${LLZK_WITGEN_TEST_SOURCES})

target_link_libraries(
  LLZK_Witgen_Tests
  PUBLIC
    LLZK::DialectRegistration
    LLZK::AllDialects
    ${GTEST_LIB_TARGETS}
    ${GTEST_EXE_TARGET}
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRParser
    LLZKWitgenTool)
target_include_directories(LLZK_Witgen_Tests PRIVATE ${PROJECT_SOURCE_DIR}/tools/llzk-witgen)

llzk_gtest_suite(LLZK_Witgen_Tests)
