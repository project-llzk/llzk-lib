add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Transforms)
add_subdirectory(Target)

# Keep dialect and pass registration separate from the dialect's consumers.
add_library(PCLDialectRegistration)
add_library(PCL::DialectRegistration ALIAS PCLDialectRegistration)
target_sources(PCLDialectRegistration PRIVATE DialectRegistration.cpp)
target_link_libraries(
  PCLDialectRegistration
  PUBLIC PCLDialect PCLConversion PCLTransforms PCLDialectHeaders MLIRFuncDialect)
llzk_target_add_mlir_link_settings(PCLDialectRegistration)

install(TARGETS PCLDialectRegistration EXPORT LLZKTargets)
