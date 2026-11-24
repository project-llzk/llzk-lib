file(GLOB_RECURSE LLZK_ANALYSIS_TEST_SOURCES "*.cpp")
add_executable(LLZK_Analysis_Tests ${LLZK_ANALYSIS_TEST_SOURCES})

target_link_libraries(
  LLZK_Analysis_Tests
  PUBLIC LLZK::DialectRegistration LLZK::AllDialects ${GTEST_LIB_TARGETS}
  ${GTEST_EXE_TARGET} ${LLZK_DEP_DIALECT_LIBS} MLIRParser
  LLZK::LLZKAnalysis)

llzk_gtest_suite(LLZK_Analysis_Tests)
