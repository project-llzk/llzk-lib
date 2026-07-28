include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "ConversionPasses.td")
mlir_tablegen(ConversionPasses.h.inc -gen-pass-decls -name Conversion)
mlir_tablegen(ConversionPasses.capi.h.inc -gen-pass-capi-header --prefix Conversion)
mlir_tablegen(ConversionPasses.capi.cpp.inc -gen-pass-capi-impl --prefix Conversion)

add_public_tablegen_target(SMTConversionIncGen)
add_dependencies(LLZKDialectHeaders SMTConversionIncGen)
