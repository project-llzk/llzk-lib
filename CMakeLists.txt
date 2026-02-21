cmake_minimum_required(VERSION 3.23)

if (NOT DEFINED LLZK_VERSION_OVERRIDE)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/GitVersion.cmake)
  get_git_version(LLZK_VERSION "0.1.0") # default version
else()
  message(STATUS "LLZK_VERSION_OVERRIDE specified, setting version to ${LLZK_VERSION_OVERRIDE}")
  set(LLZK_VERSION "${LLZK_VERSION_OVERRIDE}")
endif()

project(
  LLZK
  # Keeping this description short since it's also used by doxygen
  DESCRIPTION
    "An open-source IR for Zero Knowledge (ZK) circuits"
  VERSION "${LLZK_VERSION}"
  HOMEPAGE_URL https://github.com/project-llzk/llzk-lib)

option(LLZK_TEMPLATE_BACKTRACE_LIMIT
       "Add -ftemplate-backtrace-limit=0 to the compile flags")

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD
    20
    CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED
    ON
    CACHE STRING "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Use same policy as LLVM to suppress warnings
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FeatureTests.cmake)
# Dependency setup

find_package(LLVM 20.1.8 REQUIRED CONFIG)
find_package(MLIR 20.1.8 REQUIRED CONFIG)
find_package(Clang 20.1.8 REQUIRED CONFIG)

# Optional PCL dialect
set(LLZK_WITH_PCL "AUTO" CACHE STRING "Enable PCL integration (AUTO, ON, OFF)")
set_property(CACHE LLZK_WITH_PCL PROPERTY STRINGS "AUTO;ON;OFF")

# boolean used to guard imports in other CMakeLists.txt
set(LLZK_WITH_PCL_BOOL OFF)

if(LLZK_WITH_PCL STREQUAL "ON")
  find_package(PCL CONFIG REQUIRED)
  set(LLZK_WITH_PCL_BOOL ON)
elseif(LLZK_WITH_PCL STREQUAL "AUTO")
  find_package(PCL CONFIG QUIET)
  if(PCL_FOUND)
    set(LLZK_WITH_PCL_BOOL ON)
  endif()
elseif(LLZK_WITH_PCL STREQUAL "OFF")
  # do nothing
else()
  message(FATAL_ERROR "LLZK_WITH_PCL must be AUTO, ON, or OFF")
endif()

message(STATUS "LLZK_WITH_PCL=${LLZK_WITH_PCL} (found pcl=${LLZK_WITH_PCL_BOOL})")

message(STATUS "Using LLVM in: ${LLVM_DIR}")
message(STATUS "Using MLIR in: ${MLIR_DIR}")

# Adds an INTERFACE library that only includes headers
function(add_header_library _name _include_dirs _export_name)
  add_library(${_name} INTERFACE)
  target_include_directories(${_name} SYSTEM INTERFACE ${_include_dirs})
  install(TARGETS ${_name} EXPORT ${_export_name})
endfunction()

set(LLZK_EXPORT_TARGETS "LLZKTargets")
# LLVM & MLIR do not propagate their include dirs correctly. We define them as
# INTERFACE libraries and link against them
add_header_library(LLVMHeaders ${LLVM_INCLUDE_DIRS} ${LLZK_EXPORT_TARGETS})
add_header_library(MLIRHeaders ${MLIR_INCLUDE_DIRS} ${LLZK_EXPORT_TARGETS})

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
add_compile_options(
  -Werror
  -Wall
  -Wextra
  -Wno-unused-parameter
  -Wshadow
  $<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>
  -pedantic
  -fno-omit-frame-pointer)

if(CMAKE_CONFIGURATION_TYPES)
  list(APPEND CMAKE_CONFIGURATION_TYPES DebWithSans)
  list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
endif()

set(LLZK_LIT_COMPATIBLE_COVERAGE
    OFF
    CACHE INTERNAL "")

if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DebWithSans")
  if(LLZK_LIT_COMPATIBLE_COVERAGE)
    message(FATAL_ERROR "Don't manually set LLZK_LIT_COMPATIBLE_COVERAGE.")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    message(STATUS "Adding coverage flags")
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
    set(LLZK_LIT_COMPATIBLE_COVERAGE ON)
    message(
      STATUS "LLZK_LIT_COMPATIBLE_COVERAGE=${LLZK_LIT_COMPATIBLE_COVERAGE}")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(WARNING "Gcov reports are not supported by the test environment.")
    message(WARNING "Set LLZK_FORCE_GCC_COVERAGE to enable coverage in GCC.")
    if(LLZK_FORCE_GCC_COVERAGE)
      add_compile_options(--coverage -O0 -g)
      add_link_options(--coverage)
    endif()
  else()
    message(
      WARNING
        "Coverage requested but ${CMAKE_CXX_COMPILER_ID} is not a supported compiler and coverage will not be applied."
    )
  endif()

  add_compile_options(-O0 -g)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "DebWithSans")
  # NOTE: undefined behavior sanitizer ("undefined") has been removed due to its
  # poor interaction with gtest fixtures and other standard MLIR/LLVM operations.
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
  # LeakSanitizer is not supported on all platforms (notably, MacOS)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-fsanitize=leak" HAS_FSANITIZE_LEAK)
  if(HAS_FSANITIZE_LEAK)
    add_compile_options(-fsanitize=leak)
    add_link_options(-fsanitize=leak)
  endif()
endif()

if(LLZK_TEMPLATE_BACKTRACE_LIMIT)
  add_compile_options(-ftemplate-backtrace-limit=0)
endif()

# MAYBE FIXME: If we want to hide inline functions we need to setup macros See
# https://github.com/project-llzk/llzk-lib/pull/3#discussion_r1726140891

# Python bindings

include(CMakeDependentOption)
cmake_dependent_option(LLZK_ENABLE_BINDINGS_PYTHON "Enable Python bindings" ON
                       MLIR_ENABLE_BINDINGS_PYTHON OFF)

message(CHECK_START "Checking LLZK Python bindings")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
message(CHECK_START "Checking MLIR Python bindings")
if(MLIR_ENABLE_BINDINGS_PYTHON AND LLZK_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  include(AddMLIRPython)
  mlir_configure_python_dev_packages()
  message(CHECK_PASS "found")

  list(POP_BACK CMAKE_MESSAGE_INDENT)
  message(CHECK_PASS "enabling LLZK Python bindings")
elseif(MLIR_ENABLE_BINDINGS_PYTHON)
  message(CHECK_PASS "found")
  list(POP_BACK CMAKE_MESSAGE_INDENT)
  message(CHECK_FAIL "LLZK_ENABLE_BINDINGS_PYTHON is set to off")
else()
  message(CHECK_FAIL "not found") # mlir bindings
  list(POP_BACK CMAKE_MESSAGE_INDENT)
  message(CHECK_FAIL "missing dependencies") # overall status
endif()

# Project files

set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doc")
set(LLZK_MLIR_DOC_OUTPUT_DIR "${DOXYGEN_OUTPUT_DIRECTORY}/mlir")

# We need the include dir so we can pass it to mlir-tblgen
set(LLZK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# We need the r1cs include dir so we can pass it to mlir-tblgen
set(R1CS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/backends/r1cs/include")
# We need the zklean include dir so we can pass it to mlir-tblgen
set(ZKLEAN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/backends/zklean/include")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/LLZKMacros.cmake)

# Build `llzk-tblgen` first since it generates files used by the `include` directory below
add_subdirectory(tools/llzk-tblgen)
set(LLZK_TABLEGEN_EXE llzk-tblgen)
set(LLZK_TABLEGEN_TARGET llzk-tblgen)

# Create an empty list for the dialect libraries, to be populated in the lib
# subdirectory.
add_library(LLZKAllDialects INTERFACE)
add_library(LLZK::AllDialects ALIAS LLZKAllDialects)
# Ensures that the target can be imported as "LLZK::AllDialects" and not
# "LLZK::LLZKAllDialects"
set_target_properties(LLZKAllDialects PROPERTIES EXPORT_NAME "AllDialects")
install(TARGETS LLZKAllDialects EXPORT LLZKTargets)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(backends)
add_subdirectory(tools)
if(LLZK_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()

# Documentation
find_package(Doxygen OPTIONAL_COMPONENTS dot)
if(Doxygen_FOUND)
  message(STATUS "Doxygen found, enabling documentation and `doc` target...")
  add_custom_target(doc)

  # Fetch style document
  include(FetchContent)
  set(DoxygenAwesomeCSS_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/deps/doxygen-awesome-css")
  FetchContent_Declare(
    DoxygenAwesomeCSS
    GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
    GIT_TAG v2.4.1
    SOURCE_DIR "${DoxygenAwesomeCSS_SOURCE_DIR}"
  )
  FetchContent_MakeAvailable(DoxygenAwesomeCSS)

  # - doxygen awesome settings
  set(DOXYGEN_HTML_EXTRA_STYLESHEET
    "${DoxygenAwesomeCSS_SOURCE_DIR}/doxygen-awesome.css"
    "${DoxygenAwesomeCSS_SOURCE_DIR}/doxygen-awesome-sidebar-only.css"
    "${DoxygenAwesomeCSS_SOURCE_DIR}/doxygen-awesome-sidebar-only-darkmode-toggle.css")
  set(DOXYGEN_HTML_EXTRA_FILES
    "${DoxygenAwesomeCSS_SOURCE_DIR}/doxygen-awesome-darkmode-toggle.js"
    "${DoxygenAwesomeCSS_SOURCE_DIR}/doxygen-awesome-paragraph-link.js"
    "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/interactive-toc.js"
    "${DoxygenAwesomeCSS_SOURCE_DIR}/doxygen-awesome-fragment-copy-button.js")
  set(DOXYGEN_HTML_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/header.html")
  set(DOXYGEN_HTML_FOOTER "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/footer.html")
  set(DOXYGEN_GENERATE_TREEVIEW YES)
  set(DOXYGEN_DISABLE_INDEX NO)
  set(DOXYGEN_FULL_SIDEBAR NO)
  set(DOXYGEN_HTML_COLORSTYLE "LIGHT")
  set(DOXYGEN_DOT_IMAGE_FORMAT "svg")
  set(DOXYGEN_DOT_TRANSPARENT YES)
  set(DOXYGEN_TREEVIEW_WIDTH 700)
  # set(DOXYGEN_INTERACTIVE_SVG YES) # This is an option, but it's kinda clunky

  set(DOXYGEN_MARKDOWN_SUPPORT YES)
  set(DOXYGEN_MARKDOWN_ID_STYLE "GITHUB")

  if (DOXYGEN_VERSION VERSION_GREATER_EQUAL "1.14.0")
    # https://jothepro.github.io/doxygen-awesome-css/md_docs_2extensions.html#extension-toc
    set(DOXYGEN_PAGE_OUTLINE_PANEL NO)
  endif()

  # Remaining doxygen setup
  set(DOXYGEN_EXTRACT_ALL YES)
  set(DOXYGEN_INCLUDE_PATH "${CMAKE_CURRENT_BINARY_DIR}/include/")
  set(DOXYGEN_EXCLUDE_PATTERNS
    "${CMAKE_CURRENT_BINARY_DIR}/include/*/*.md"
    # We ignore the passes because we aggregate the documentation in doc/doxygen/2_tools.md under `llzk-opt`
    "${LLZK_MLIR_DOC_OUTPUT_DIR}/passes/*.md"
    # We ignore the dialects because we aggregate the documentation in doc/doxygen/7_dialects.md
    "${LLZK_MLIR_DOC_OUTPUT_DIR}/dialect/*.md"
  )
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE
      ${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/index.md)
  set(DOXYGEN_FILE_PATTERNS
      *.cpp
      *.cpp.inc
      *.h.inc
      *.hpp
      *.h
      *.td
      *.md
      *.py
      *.txt)
  set(DOXYGEN_EXTENSION_MAPPING inc=C++)
  set(DOXYGEN_MACRO_EXPANSION YES)
  set(DOXYGEN_EXPAND_ONLY_PREDEF YES)
  set(DOXYGEN_PREDEFINED GET_OP_CLASSES GET_TYPEDEF_CLASSES GET_ATTR_CLASSES)
  set(DOXYGEN_SOURCE_BROWSER YES)
  set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
  set(DOXYGEN_ALIASES "llzkVersion=${LLZK_VERSION}")
  # The order that docs are added affects the sorting in the navigation bar.
  doxygen_add_docs(
    doxygen
    "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/"
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/"
    "${CMAKE_CURRENT_BINARY_DIR}/include/"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    "${LLZK_MLIR_DOC_OUTPUT_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
  add_dependencies(doxygen LLZKDialectHeaders mlir-doc)
  add_dependencies(doc doxygen)
endif()

# Tests

# lit tests
include(CTest)
enable_testing()
set(CMAKE_CTEST_ARGUMENTS
    "--output-on-failure"
    CACHE STRING "CTest arguments")
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

find_package(GTest)
if(BUILD_TESTING AND GTest_FOUND)
  message(STATUS "gtest found, enabling unit tests...")
  include(GoogleTest)

  add_custom_target(LLZKUnitTests)

  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20")
    set(GTEST_LIB_TARGETS "GTest::gtest;GTest::gmock")
    set(GTEST_EXE_TARGET GTest::gtest_main)
  else()
    set(GTEST_LIB_TARGETS GTest::GTest)
    set(GTEST_EXE_TARGET GTest::Main)
  endif()

  add_subdirectory(unittests)
elseif(BUILD_TESTING)
  message(STATUS "gtest not found, unit tests will not be run")
else()
  message(STATUS "Unit tests are disabled")
endif()

# Catch-all target for running all tests
add_custom_target(
  check
  DEPENDS # test targets may not exist if BUILD_TESTING is off
          $<$<BOOL:BUILD_TESTING>:check-unit>
          $<$<BOOL:BUILD_TESTING>:check-lit>)

# Install

export(
  EXPORT ${LLZK_EXPORT_TARGETS}
  FILE ${CMAKE_CURRENT_BINARY_DIR}/${LLZK_EXPORT_TARGETS}.cmake
  NAMESPACE LLZK::)
install(
  EXPORT ${LLZK_EXPORT_TARGETS}
  NAMESPACE LLZK::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LLZK)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/LLZKConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LLZKConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/LLZKConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/LLZK")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/LLZKConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/LLZKConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/LLZK")
