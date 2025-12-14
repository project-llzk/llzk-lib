set(LLZK_DEP_DIALECT_LIBS MLIRFuncDialect MLIRSCFDialect)
set(LLZK_DEP_DIALECT_LIBS
    ${LLZK_DEP_DIALECT_LIBS}
    PARENT_SCOPE)

# All dialects
add_subdirectory(Shared)
add_subdirectory(Array)
add_subdirectory(Bool)
add_subdirectory(Cast)
add_subdirectory(Constrain)
add_subdirectory(Felt)
add_subdirectory(Function)
add_subdirectory(Global)
add_subdirectory(Include)
add_subdirectory(POD)
add_subdirectory(LLZK)
add_subdirectory(Polymorphic)
add_subdirectory(String)
add_subdirectory(Struct)
add_subdirectory(ZKBuilder)
add_subdirectory(ZKExpr)

# Registration of dialects
add_library(LLZKDialectRegistration)
add_library(LLZK::DialectRegistration ALIAS LLZKDialectRegistration)
# Ensures that the target can be imported as "LLZK::DialectRegistration" and not
# "LLZK::LLZKDialectRegistration"
set_target_properties(LLZKDialectRegistration PROPERTIES EXPORT_NAME
                                                         "DialectRegistration")
target_sources(LLZKDialectRegistration PRIVATE InitDialects.cpp)
target_link_libraries(
  LLZKDialectRegistration PUBLIC LLZKAllDialects ${LLZK_DEP_DIALECT_LIBS}
                                 LLZKDialectHeaders
                                 LLZK::LLZKTransforms)
llzk_target_add_mlir_link_settings(LLZKDialectRegistration)

install(TARGETS LLZKDialectRegistration EXPORT LLZKTargets)
