add_library(SMTDialectHeaders INTERFACE)
target_include_directories(SMTDialectHeaders
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_include_directories(SMTDialectHeaders
  SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(SMTDialectHeaders
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tools/include>)
install(TARGETS SMTDialectHeaders EXPORT LLZKTargets)
add_subdirectory(smt/Conversions)
