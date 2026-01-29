add_library(LLZKDialectHeaders INTERFACE)
target_include_directories(LLZKDialectHeaders
  INTERFACE # headers in this folder's subdirectories
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_include_directories(LLZKDialectHeaders
  SYSTEM INTERFACE
            # TableGen'd files
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            # We assume LLZK is installed as-is to include top level
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS LLZKDialectHeaders EXPORT LLZKTargets)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" "${CMAKE_CURRENT_BINARY_DIR}/"
  TYPE INCLUDE
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.h.inc"
  PATTERN "*.td"
  PATTERN CMakeFiles EXCLUDE)

# subdirectories will add their IncGen targets as deps to the header folder
add_subdirectory(llzk/Analysis)
add_subdirectory(llzk/Config)
add_subdirectory(llzk/Dialect/Array/IR)
add_subdirectory(llzk/Dialect/Array/Transforms)
add_subdirectory(llzk/Dialect/Bool/IR)
add_subdirectory(llzk/Dialect/Cast/IR)
add_subdirectory(llzk/Dialect/Constrain/IR)
add_subdirectory(llzk/Dialect/Felt/IR)
add_subdirectory(llzk/Dialect/Function/IR)
add_subdirectory(llzk/Dialect/Global/IR)
add_subdirectory(llzk/Dialect/Include/IR)
add_subdirectory(llzk/Dialect/Include/Transforms)
add_subdirectory(llzk/Dialect/LLZK/IR)
add_subdirectory(llzk/Dialect/POD/IR)
add_subdirectory(llzk/Dialect/Polymorphic/IR)
add_subdirectory(llzk/Dialect/Polymorphic/Transforms)
add_subdirectory(llzk/Dialect/String/IR)
add_subdirectory(llzk/Dialect/Struct/IR)
add_subdirectory(llzk/Dialect/ZKLeanLean/IR)
add_subdirectory(llzk/Dialect/ZKBuilder/IR)
add_subdirectory(llzk/Dialect/ZKExpr/IR)
add_subdirectory(llzk/Conversions)

add_subdirectory(llzk/Transforms)
add_subdirectory(llzk/Util)
add_subdirectory(llzk/Validators)
