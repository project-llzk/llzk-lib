include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=pod)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=pod)

set(LLVM_TARGET_DEFINITIONS "Attrs.td")
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=pod)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=pod)

set(LLVM_TARGET_DEFINITIONS "Types.td")
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=pod)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=pod)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=pod)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=pod)

llzk_add_mlir_doc(PODOpsDocGen dialect/PODDialect.md -gen-dialect-doc -dialect=pod)

add_public_tablegen_target(PODDialectIncGen)
add_dependencies(mlir-headers PODDialectIncGen)
add_dependencies(LLZKDialectHeaders PODDialectIncGen)
