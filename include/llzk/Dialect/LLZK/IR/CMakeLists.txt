include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=llzk)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=llzk)

set(LLVM_TARGET_DEFINITIONS "Attrs.td")
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=llzk)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=llzk)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=llzk)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=llzk)

llzk_add_mlir_doc(LLZKOpsDocGen dialect/LLZKDialect.md -gen-dialect-doc -dialect=llzk)

add_public_tablegen_target(LLZKDialectIncGen)
add_dependencies(mlir-headers LLZKDialectIncGen)
add_dependencies(LLZKDialectHeaders LLZKDialectIncGen)
