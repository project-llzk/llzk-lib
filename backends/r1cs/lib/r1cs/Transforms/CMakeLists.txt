add_library(R1CSTransforms)
add_library(R1CS::Transforms ALIAS R1CSTransforms)

file(GLOB R1CSTransforms_SOURCES "*.cpp")


target_sources(R1CSTransforms PRIVATE ${R1CSTransforms_SOURCES})

target_link_libraries(
  R1CSTransforms
  PUBLIC
    LLZKDialectHeaders
    R1CSDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
    LLVMHeaders MLIRHeaders
  PRIVATE
    LLZKDialect
    R1CSDialect
    LLZKAnalysis
)

llzk_target_add_mlir_link_settings(R1CSTransforms)
install(TARGETS R1CSTransforms EXPORT LLZKTargets)
