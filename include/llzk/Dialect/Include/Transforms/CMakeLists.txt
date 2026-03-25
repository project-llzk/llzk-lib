include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "InlineIncludesPass.td")
mlir_tablegen(InlineIncludesPass.h.inc -gen-pass-decls -name=Transformation)
mlir_tablegen(InlineIncludesPass.capi.h.inc -gen-pass-capi-header --prefix LLZKIncludeTransformation)
mlir_tablegen(InlineIncludesPass.capi.cpp.inc -gen-pass-capi-impl --prefix LLZKIncludeTransformation)
llzk_add_mlir_doc(InlineIncludesPassDocGen passes/InlineIncludesPass.md -gen-pass-doc)

add_public_tablegen_target(InlineIncludesPassIncGen)
add_dependencies(LLZKDialectHeaders InlineIncludesPassIncGen)
