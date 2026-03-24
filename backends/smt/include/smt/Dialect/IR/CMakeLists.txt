include_directories(${MLIR_INCLUDE_DIRS} ${SMT_INCLUDE_DIR} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS SMTDialect.td)
mlir_tablegen(SMTDialect.h.inc -gen-dialect-decls -dialect=smt)
mlir_tablegen(SMTDialect.cpp.inc -gen-dialect-defs -dialect=smt)

set(LLVM_TARGET_DEFINITIONS SMTTypes.td)
mlir_tablegen(SMTTypes.h.inc -gen-typedef-decls -typedefs-dialect=smt)
mlir_tablegen(SMTTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=smt)

set(LLVM_TARGET_DEFINITIONS SMTAttributes.td)
mlir_tablegen(SMTAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SMTAttributes.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS SMT.td)
mlir_tablegen(SMT.h.inc -gen-op-decls -dialect=smt)
mlir_tablegen(SMT.cpp.inc -gen-op-defs -dialect=smt)

# Enums are emitted from the main op file.
mlir_tablegen(SMTEnums.h.inc -gen-enum-decls)
mlir_tablegen(SMTEnums.cpp.inc -gen-enum-defs)

llzk_add_mlir_doc(SMTOpsDocGen dialect/SMTDialect.md -gen-dialect-doc -dialect=smt)

add_public_tablegen_target(SMTDialectIncGen)
add_dependencies(LLZKDialectHeaders SMTDialectIncGen)
