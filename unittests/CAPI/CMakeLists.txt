file(GLOB_RECURSE LLZK_CAPI_TEST_SOURCES "*.cpp")
add_executable(LLZK_CAPI_Tests ${LLZK_CAPI_TEST_SOURCES})

add_subdirectory(Dialect)
target_link_libraries(
  LLZK_CAPI_Tests
  PUBLIC LLZK::DialectRegistration LLZK::AllDialects ${GTEST_LIB_TARGETS}
  ${GTEST_EXE_TARGET} ${LLZK_DEP_DIALECT_LIBS} MLIRParser
  LLZK::LLZKAnalysis LLZK::LLZKCAPI LLZK::LLZKTransforms LLZK::LLZKValidators)

llzk_gtest_suite(LLZK_CAPI_Tests)

# C compatibility test - verifies the C API can be consumed from pure C code
add_executable(LLZK_CAPI_C_Test c_compatibility_test.c)
set_target_properties(LLZK_CAPI_C_Test PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(
  LLZK_CAPI_C_Test
  PRIVATE
  LLZK::LLZKCAPI
  MLIRCAPIIR
  MLIRCAPIRegisterEverything)
add_test(NAME LLZK_CAPI_C_Test COMMAND LLZK_CAPI_C_Test)
