include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "TransformationPasses.td")
mlir_tablegen(TransformationPasses.h.inc -gen-pass-decls -name=StructTransformation)
mlir_tablegen(TransformationPasses.capi.h.inc -gen-pass-capi-header --prefix LLZKStructTransformation)
mlir_tablegen(TransformationPasses.capi.cpp.inc -gen-pass-capi-impl --prefix LLZKStructTransformation)
llzk_add_mlir_doc(StructTransformationPassesDocGen passes/struct/TransformationPasses.md -gen-pass-doc)

add_public_tablegen_target(StructTransformationIncGen)
add_dependencies(LLZKDialectHeaders StructTransformationIncGen)
