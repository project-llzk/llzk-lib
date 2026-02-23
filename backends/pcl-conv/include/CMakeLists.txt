
add_library(PCLConvDialectHeaders INTERFACE)
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
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            # Headers from LLZK tools include directory
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tools/include>)
install(TARGETS PCLConvDialectHeaders EXPORT LLZKTargets)

add_subdirectory(pcl-conv/Transforms/)
