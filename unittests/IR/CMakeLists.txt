file(GLOB_RECURSE LLZK_IR_TEST_SOURCES "*.cpp")
add_executable(LLZK_IR_Tests ${LLZK_IR_TEST_SOURCES})

target_link_libraries(
  LLZK_IR_Tests
  PUBLIC LLZK::DialectRegistration LLZK::AllDialects ${GTEST_LIB_TARGETS} ${GTEST_EXE_TARGET}
  ${LLZK_DEP_DIALECT_LIBS} MLIRParser
  LLZKUtil)

llzk_gtest_suite(LLZK_IR_Tests)
