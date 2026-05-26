function(llzk_gtest_suite target)
  gtest_discover_tests(${target})
  list(APPEND LLZK_UNITTESTS ${target})
  set(LLZK_UNITTESTS
      ${LLZK_UNITTESTS}
      CACHE INTERNAL "")
endfunction()

add_subdirectory(Analysis)
add_subdirectory(IR)
add_subdirectory(Util)
add_subdirectory(Witgen)
add_subdirectory(CAPI)

add_custom_target(
  check-unit
  DEPENDS ${LLZK_UNITTESTS}
  COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target test
  COMMENT "Running the gtest-based unit tests")
