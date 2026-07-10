add_library(ZKLeanTarget)
add_library(ZKLean::Target ALIAS ZKLeanTarget)

file(GLOB ZKLeanTarget_SOURCES "*.cpp")
target_sources(ZKLeanTarget PRIVATE ${ZKLeanTarget_SOURCES})

target_link_libraries(
  ZKLeanTarget
  PUBLIC
    ZKLeanDialectHeaders
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    LLVMHeaders
    MLIRHeaders
  PRIVATE
    LLZKDialect
    ZKLeanZKExprDialect
    ZKLeanZKLeanLeanDialect
    LLZKBoolDialect
)

llzk_target_add_mlir_link_settings(ZKLeanTarget)
install(TARGETS ZKLeanTarget EXPORT LLZKTargets)
