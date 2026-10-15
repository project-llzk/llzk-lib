#
# Top level targets
#

# Set both MLIR and nanobind domain definitions consistently.
set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "llzk")

# Python runtime libraries are installed beside the extensions. In particular,
# Nix's global CMAKE_INSTALL_NAME_DIR points at lib/, not this package directory.
# MLIR adds @loader_path to the modules' rpaths so colocated libraries resolve.
if(APPLE)
  set(CMAKE_INSTALL_NAME_DIR "@rpath")
endif()

declare_mlir_python_sources(LLZKPythonDialectSources)
declare_mlir_python_sources(LLZKPythonDialectSources.Dialects ADD_TO_PARENT
                            LLZKPythonDialectSources)

# Include the tablegen files
include_directories("${LLZK_INCLUDE_DIR}" ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

#
# Dialect bindings
#

# These must go under the mlir module/namespace, as the generated sources use
# relative imports from mlir.

# Generate operation wrappers for every core dialect from the same ODS input.
# Custom types and attributes are available through MLIR's generic parse APIs.
set(LLZK_PYTHON_DIALECTS
    array bool cast constrain felt function global include llzk pod poly ram
    smt_info string struct verif)
# Track transitive ODS inputs in installed-MLIR builds as well as in-tree builds.
file(GLOB_RECURSE LLZK_PYTHON_ODS_DEPENDS CONFIGURE_DEPENDS
     "${LLZK_INCLUDE_DIR}/llzk/Dialect/*.td")
set(LLZK_PYTHON_DIALECT_WRAPPERS)
foreach(dialect IN LISTS LLZK_PYTHON_DIALECTS)
  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT LLZKPythonDialectSources.Dialects
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
    TD_FILE dialects/LLZKOps.td
    DIALECT_NAME ${dialect}
    DEPENDS ${LLZK_PYTHON_ODS_DEPENDS})

  # The module name matches the IR namespace for automatic OpView discovery.
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dialects/${dialect}.py"
       "# Generated LLZK dialect operation bindings.\nfrom ._${dialect}_ops_gen import *\n")
  list(APPEND LLZK_PYTHON_DIALECT_WRAPPERS dialects/${dialect}.py)
endforeach()
declare_mlir_python_sources(
  LLZKPythonDialectSources.Wrappers
  ADD_TO_PARENT LLZKPythonDialectSources
  ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  SOURCES ${LLZK_PYTHON_DIALECT_WRAPPERS})
declare_mlir_python_sources(
  LLZKPythonDialectSources.Compatibility
  ADD_TO_PARENT LLZKPythonDialectSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  SOURCES dialects/llzk_dialect.py)

#
# Python extensions
#

# All other LLZK Python bindings go under the llzk package.
declare_mlir_python_sources(LLZKPythonSources)

declare_mlir_python_sources(
  LLZKPythonSources.Core
  ADD_TO_PARENT
  LLZKPythonSources
  ROOT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/llzk"
  SOURCES
  __init__.py
  registration.py)

declare_mlir_python_extension(
  LLZKPythonExtension
  ADD_TO_PARENT
  LLZKPythonSources
  MODULE_NAME
  _llzkRegistration
  ROOT_DIR
  "${CMAKE_SOURCE_DIR}/lib/Bindings/Python"
  SOURCES
  LLZKRegistration.cpp
  PRIVATE_LINK_LIBS
  LLZKPythonBindings)

#
# Final modules
#

add_mlir_python_modules(
  LLZKPythonDialectModules
  MLIR_BINDINGS_PYTHON_NB_DOMAIN llzk
  ROOT_PREFIX
  "${CMAKE_CURRENT_BINARY_DIR}/mlir"
  INSTALL_PREFIX
  python/mlir
  DECLARED_SOURCES
  LLZKPythonDialectSources)

# Installed MLIR distributions do not consistently export this CMake target.
# Search the Python package as well as the normal library paths (including Nix's
# separate lib output), and reuse an exported target when one is available.
if(NOT TARGET MLIRPythonCAPI)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "from mlir._mlir_libs import get_lib_dirs; print(';'.join(get_lib_dirs()))"
    OUTPUT_VARIABLE MLIR_PYTHON_LIBRARY_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  find_library(MLIRPythonCAPI_LIBRARY NAMES MLIRPythonCAPI
               HINTS ${MLIR_PYTHON_LIBRARY_DIRS} REQUIRED)
  add_library(MLIRPythonCAPI SHARED IMPORTED)
  set_target_properties(MLIRPythonCAPI PROPERTIES
                        IMPORTED_LOCATION "${MLIRPythonCAPI_LIBRARY}")
endif()

add_mlir_python_modules(
  LLZKPythonExtensionModules
  MLIR_BINDINGS_PYTHON_NB_DOMAIN llzk
  # LLZK enables -Werror -Wshadow globally; nanobind's runtime has shadowed locals.
  MLIR_BINDINGS_PYTHON_EXTRA_NANOBIND_COMPILE_OPTIONS
  "$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang,GNU>:-Wno-shadow>"
  ROOT_PREFIX
  "${CMAKE_CURRENT_BINARY_DIR}/llzk"
  INSTALL_PREFIX
  python/llzk
  DECLARED_SOURCES
  LLZKPythonSources
  COMMON_CAPI_LINK_LIBS
  MLIRPythonCAPI)

# MLIRPythonCAPI already contains the upstream C++ runtime. Linking its static
# libraries again gives builtin types/attributes different TypeIDs on Darwin.
# LLZK libraries use MLIR's aggregate link filtering so only this consumer skips
# those archives; native tools retain their usual dependencies.
set(LLZK_PYTHON_EXCLUDE_LIBS)
foreach(library IN LISTS MLIR_EXPORTED_TARGETS)
  if(TARGET ${library})
    get_target_property(library_type ${library} TYPE)
    if(library_type STREQUAL "STATIC_LIBRARY")
      list(APPEND LLZK_PYTHON_EXCLUDE_LIBS ${library})
    endif()
  endif()
endforeach()
set_property(TARGET LLZKPythonExtensionModules.extension._llzkRegistration.dso
             PROPERTY MLIR_AGGREGATE_EXCLUDE_LIBS "${LLZK_PYTHON_EXCLUDE_LIBS}")
# The C API runtime depends on LLVM, which also supplies LLZK's direct LLVM calls.
target_link_libraries(LLZKPythonExtensionModules.extension._llzkRegistration.dso
                      PRIVATE LLVM
                      # Dense forward dataflow is not pulled into MLIRPythonCAPI.
                      # Link this archive without its transitive static runtime;
                      # the shared C API supplies the rest of its dependencies.
                      "$<TARGET_FILE:MLIRAnalysis>")
