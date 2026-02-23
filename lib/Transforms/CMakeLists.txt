add_library(LLZKTransforms)
add_library(LLZK::LLZKTransforms ALIAS LLZKTransforms)

file(GLOB LLZKTransforms_SOURCES "*.cpp")

target_sources(LLZKTransforms PRIVATE ${LLZKTransforms_SOURCES})

# Remove the PCL lowering .cpp only when disabled.
if(NOT LLZK_WITH_PCL_BOOL)
  list(REMOVE_ITEM LLZKTransforms_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/PCLLoweringPass.cpp")
endif()

target_link_libraries(
  LLZKTransforms
  PUBLIC
    $<$<BOOL:${LLZK_WITH_PCL_BOOL}>:PCL::PCLDialect>
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
    LLVMHeaders MLIRHeaders
  PRIVATE
    LLZKDialect
    LLZKAnalysis
)

llzk_target_add_mlir_link_settings(LLZKTransforms)
install(TARGETS LLZKTransforms EXPORT LLZKTargets)
