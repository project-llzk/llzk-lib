include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "TransformationPasses.td")
mlir_tablegen(TransformationPasses.h.inc -gen-pass-decls -name=Transformation)
mlir_tablegen(TransformationPasses.capi.h.inc -gen-pass-capi-header --prefix R1CSTransformation)
mlir_tablegen(TransformationPasses.capi.cpp.inc -gen-pass-capi-impl --prefix R1CSTransformation)
llzk_add_mlir_doc(TransformationPassesDocGen passes/TransformationPasses.md -gen-pass-doc)

add_public_tablegen_target(R1CSTransformationIncGen)
add_dependencies(mlir-headers R1CSTransformationIncGen)
add_dependencies(LLZKDialectHeaders R1CSTransformationIncGen)
