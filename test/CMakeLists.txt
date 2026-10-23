# llvm-lit is named lit in external builds
find_program(LLVM_EXTERNAL_LIT NAMES lit llvm-lit ${LLVM_EXTERNAL_LIT} REQUIRED)
message(STATUS "Using lit: ${LLVM_EXTERNAL_LIT}")

# LLVM runs LLVM_EXTERNAL_LIT through Python when Python3_EXECUTABLE is set.
# Nix's lit is a shell wrapper that must run to establish its Python environment.
# A Python trampoline also works with the executable provided by a manual build.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(LLZK_LIT_EXECUTABLE "${LLVM_EXTERNAL_LIT}")
configure_file(lit-launcher.py.in lit-launcher.py @ONLY)
set(LLVM_EXTERNAL_LIT "${CMAKE_CURRENT_BINARY_DIR}/lit-launcher.py")

set(LLZK_TEST_DEPENDS llzk-opt llzk-smt-check llzk-witgen llzk-tblgen llzk-translate r1cs-opt)

set(LLZK_ENABLE_BINDINGS_PYTHON_STRING "False")
if(LLZK_ENABLE_BINDINGS_PYTHON)
  set(LLZK_ENABLE_BINDINGS_PYTHON_STRING "True")
  list(APPEND LLZK_TEST_DEPENDS LLZKPythonDialectModules
       LLZKPythonExtensionModules)
endif()

set(LLZK_LIT_COMPATIBLE_COVERAGE_STRING "False")
if(LLZK_LIT_COMPATIBLE_COVERAGE)
  message(STATUS "Lit will generate coverage information for every test")
  set(LLZK_LIT_COMPATIBLE_COVERAGE_STRING "True")
endif()

set(LIT_ARGS "--xunit-xml-output report.xml")

# creating this guard and setting to 0-1
# because python doesn't like the ON/OFF stuff
set(LLZK_WITH_PCL_GUARD 0)
if(LLZK_WITH_PCL_BOOL)
  set(LLZK_WITH_PCL_GUARD 1)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

add_lit_testsuite(
  check-lit
  "Running the lit-based regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  ${LLZK_TEST_DEPENDS}
  ARGS
  ${LIT_ARGS})
