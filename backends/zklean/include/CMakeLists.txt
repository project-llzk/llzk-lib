add_library(ZKLeanDialectHeaders INTERFACE)
target_include_directories(ZKLeanDialectHeaders
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_include_directories(ZKLeanDialectHeaders
  SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(ZKLeanDialectHeaders
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tools/include>)
install(TARGETS ZKLeanDialectHeaders EXPORT LLZKTargets)

add_subdirectory(zklean/Dialect/ZKExpr/IR)
add_subdirectory(zklean/Dialect/ZKBuilder/IR)
add_subdirectory(zklean/Dialect/ZKLeanLean/IR)
add_subdirectory(zklean/Transforms)
add_subdirectory(zklean/Conversions)
