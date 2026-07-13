add_library(PCLDialectHeaders INTERFACE)
add_library(LLZK::PCLDialectHeaders ALIAS PCLDialectHeaders)
target_include_directories(PCLDialectHeaders
  INTERFACE # headers in this folder's subdirectories
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_include_directories(PCLDialectHeaders
  SYSTEM INTERFACE
            # TableGen'd files
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
target_include_directories(PCLDialectHeaders
  INTERFACE
            # Headers from core LLZK include directory
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/backends/pcl/include>)
install(TARGETS PCLDialectHeaders EXPORT LLZKTargets)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" "${CMAKE_CURRENT_BINARY_DIR}/"
  TYPE INCLUDE
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.h.inc"
  PATTERN "*.td"
  PATTERN CMakeFiles EXCLUDE)

add_subdirectory(pcl/Conversion/)
# add_subdirectory(pcl/Transforms/)
add_subdirectory(pcl/Dialect/IR/)
