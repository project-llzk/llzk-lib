add_library(LLZKArrayUtil)
add_library(LLZK::LLZKArrayUtil ALIAS LLZKArrayUtil)

target_sources(LLZKArrayUtil PRIVATE ArrayTypeHelper.cpp)
target_link_libraries(
  LLZKArrayUtil
  PUBLIC
    LLZKDialectHeaders
    # ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR
    # MLIRPass
    # MLIRParser
    # MLIRTransformUtils
    # MLIRSCFTransforms
    # LLVMHeaders
    # MLIRHeaders
    # LLZKDialect
    # LLZKDialectShared
)
llzk_target_add_mlir_link_settings(LLZKArrayUtil)

install(TARGETS LLZKArrayUtil EXPORT LLZKTargets)
