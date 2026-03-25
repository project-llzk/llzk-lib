include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "TransformationPasses.td")
mlir_tablegen(TransformationPasses.h.inc -gen-pass-decls -name=Transformation)
mlir_tablegen(TransformationPasses.capi.h.inc -gen-pass-capi-header --prefix LLZKPolymorphicTransformation)
mlir_tablegen(TransformationPasses.capi.cpp.inc -gen-pass-capi-impl --prefix LLZKPolymorphicTransformation)
mlir_tablegen(TransformationPassEnums.h.inc -gen-enum-decls)
mlir_tablegen(TransformationPassEnums.cpp.inc -gen-enum-defs)
llzk_add_mlir_doc(PolymorphicTransformationPassesDocGen passes/polymorphic/TransformationPasses.md -gen-pass-doc)

add_public_tablegen_target(PolymorphicTransformationIncGen)
add_dependencies(LLZKDialectHeaders PolymorphicTransformationIncGen)
