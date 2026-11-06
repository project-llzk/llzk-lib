include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "TransformationPasses.td")
mlir_tablegen(TransformationPasses.h.inc -gen-pass-decls -name=Transformation)
mlir_tablegen(TransformationPasses.capi.h.inc -gen-pass-capi-header --prefix LLZKGlobalTransformation)
mlir_tablegen(TransformationPasses.capi.cpp.inc -gen-pass-capi-impl --prefix LLZKGlobalTransformation)
llzk_add_mlir_doc(GlobalTransformationPassesDocGen passes/global/TransformationPasses.md -gen-pass-doc)

add_public_tablegen_target(GlobalTransformationIncGen)
add_dependencies(LLZKDialectHeaders GlobalTransformationIncGen)
