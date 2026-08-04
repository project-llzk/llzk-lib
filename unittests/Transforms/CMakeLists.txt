file(GLOB_RECURSE LLZK_TRANSFORMS_TEST_SOURCES "*.cpp")
add_executable(LLZK_Transforms_Tests ${LLZK_TRANSFORMS_TEST_SOURCES})

target_link_libraries(
  LLZK_Transforms_Tests
  PUBLIC LLZK::DialectRegistration LLZK::AllDialects ${GTEST_LIB_TARGETS} ${GTEST_EXE_TARGET}
  ${LLZK_DEP_DIALECT_LIBS} MLIRParser LLZK::LLZKTransforms)

llzk_gtest_suite(LLZK_Transforms_Tests)
