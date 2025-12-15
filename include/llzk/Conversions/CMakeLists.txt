include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Passes.td")
mlir_tablegen(LLZKConversionPasses.h.inc -gen-pass-decls -name LLZKConversion)
mlir_tablegen(LLZKConversionPasses.capi.h.inc -gen-pass-capi-header --prefix LLZKConversion)
mlir_tablegen(LLZKConversionPasses.capi.cpp.inc -gen-pass-capi-impl --prefix LLZKConversion)

add_public_tablegen_target(LLZKConversionsIncGen)
add_dependencies(mlir-headers LLZKConversionsIncGen)
add_dependencies(LLZKDialectHeaders LLZKConversionsIncGen)
