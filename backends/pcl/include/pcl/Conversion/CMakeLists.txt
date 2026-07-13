include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "ConversionPasses.td")
mlir_tablegen(ConversionPasses.h.inc -gen-pass-decls -name=PCLConversion)
mlir_tablegen(ConversionPasses.capi.h.inc -gen-pass-capi-header --prefix PCLConversion)
mlir_tablegen(ConversionPasses.capi.cpp.inc -gen-pass-capi-impl --prefix PCLConversion)
llzk_add_mlir_doc(PCLConversionPassesDocGen passes/PCLConversionPasses.md -gen-pass-doc)

add_public_tablegen_target(PCLConversionIncGen)
add_dependencies(LLZKDialectHeaders PCLConversionIncGen)
