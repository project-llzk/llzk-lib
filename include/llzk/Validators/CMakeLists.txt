include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "LLZKValidationPasses.td")
mlir_tablegen(LLZKValidationPasses.h.inc -gen-pass-decls -name=Validation)
mlir_tablegen(LLZKValidationPasses.capi.h.inc -gen-pass-capi-header --prefix LLZKValidation)
mlir_tablegen(LLZKValidationPasses.capi.cpp.inc -gen-pass-capi-impl --prefix LLZKValidation)
llzk_add_mlir_doc(LLZKValidationPassesDocGen passes/LLZKValidationPasses.md -gen-pass-doc)

add_public_tablegen_target(LLZKValidationIncGen)
add_dependencies(LLZKDialectHeaders LLZKValidationIncGen)
