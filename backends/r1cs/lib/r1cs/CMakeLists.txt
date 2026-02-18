add_subdirectory(Dialect)
add_subdirectory(Transforms)

# Registration of dialects
add_library(R1CSDialectRegistration)
add_library(R1CS::DialectRegistration ALIAS R1CSDialectRegistration)
set_target_properties(R1CSDialectRegistration PROPERTIES EXPORT_NAME
                                                         "DialectRegistration")
target_sources(R1CSDialectRegistration PRIVATE DialectRegistration.cpp)
target_link_libraries(
  R1CSDialectRegistration PUBLIC LLZKAllDialects ${LLZK_DEP_DIALECT_LIBS}
                                 R1CSDialectHeaders
                                 R1CSTransforms)
llzk_target_add_mlir_link_settings(R1CSDialectRegistration)

install(TARGETS R1CSDialectRegistration EXPORT LLZKTargets)
