include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS SMTInfoDialect.td)
mlir_tablegen(SMTInfoDialect.h.inc -gen-dialect-decls -dialect=smt_info)
mlir_tablegen(SMTInfoDialect.cpp.inc -gen-dialect-defs -dialect=smt_info)

set(LLVM_TARGET_DEFINITIONS SMTInfoAttributes.td)
mlir_tablegen(SMTInfoAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SMTInfoAttributes.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS SMTInfoOps.td)
mlir_tablegen(SMTInfoOps.h.inc -gen-op-decls -dialect=smt_info)
mlir_tablegen(SMTInfoOps.cpp.inc -gen-op-defs -dialect=smt_info)

llzk_add_mlir_doc(SMTInfoOpsDocGen dialect/SMTInfoDialect.md -gen-dialect-doc -dialect=smt_info)

add_public_tablegen_target(SMTInfoDialectIncGen)
add_dependencies(LLZKDialectHeaders SMTInfoDialectIncGen)
