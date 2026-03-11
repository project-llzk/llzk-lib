add_library(PCLConvDialectHeaders INTERFACE)
add_library(LLZK::PCLConvDialectHeaders ALIAS PCLConvDialectHeaders)
target_include_directories(PCLConvDialectHeaders
  INTERFACE # headers in this folder's subdirectories
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_include_directories(PCLConvDialectHeaders
  SYSTEM INTERFACE
            # TableGen'd files
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            # We assume LLZK is installed as-is to include top level
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(PCLConvDialectHeaders
  INTERFACE
            # Headers from core LLZK include directory
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/backends/pcl-conv/include>)
install(TARGETS PCLConvDialectHeaders EXPORT LLZKTargets)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" "${CMAKE_CURRENT_BINARY_DIR}/"
  TYPE INCLUDE
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.h.inc"
  PATTERN "*.td"
  PATTERN CMakeFiles EXCLUDE)

add_subdirectory(pcl-conv/Transforms/)
