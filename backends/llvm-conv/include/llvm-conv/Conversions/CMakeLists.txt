include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR} ${LLVM_CONV_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "ConversionPasses.td")
mlir_tablegen(ConversionPasses.h.inc -gen-pass-decls -name=LLVMConversion)
mlir_tablegen(ConversionPasses.capi.h.inc -gen-pass-capi-header --prefix LLVMConversion)
mlir_tablegen(ConversionPasses.capi.cpp.inc -gen-pass-capi-impl --prefix LLVMConversion)
llzk_add_mlir_doc(LLVMConversionPassesDocGen passes/LLVMConversionPasses.md -gen-pass-doc)

add_public_tablegen_target(LLVMConversionIncGen)
add_dependencies(LLZKDialectHeaders LLVMConversionIncGen)
