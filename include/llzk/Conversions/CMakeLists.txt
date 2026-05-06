include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Passes.td")
mlir_tablegen(Passes.h.inc -gen-pass-decls -name=Conversion)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix LLZKConversions)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix LLZKConversions)
llzk_add_mlir_doc(LLZKTConversionsDocGen passes/LLZKConversionPasses.md -gen-pass-doc)

add_public_tablegen_target(LLZKTConversionsIncGen)
add_dependencies(LLZKDialectHeaders LLZKTConversionsIncGen)
