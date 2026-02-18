add_library(R1CSDialectHeaders INTERFACE)
target_include_directories(R1CSDialectHeaders
  INTERFACE # headers in this folder's subdirectories
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_include_directories(R1CSDialectHeaders
  SYSTEM INTERFACE
            # TableGen'd files
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            # We assume LLZK is installed as-is to include top level
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
install(TARGETS R1CSDialectHeaders EXPORT LLZKTargets)

add_subdirectory(r1cs/Dialect/IR/)
