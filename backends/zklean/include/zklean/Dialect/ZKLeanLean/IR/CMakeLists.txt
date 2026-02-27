include_directories(${MLIR_INCLUDE_DIRS} ${ZKLEAN_INCLUDE_DIR} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "ZKLeanLeanDialect.td")
mlir_tablegen(ZKLeanLeanDialect.h.inc -gen-dialect-decls -dialect=ZKLeanLean)
mlir_tablegen(ZKLeanLeanDialect.cpp.inc -gen-dialect-defs -dialect=ZKLeanLean)

set(LLVM_TARGET_DEFINITIONS "ZKLeanLeanTypes.td")
mlir_tablegen(ZKLeanLeanTypes.h.inc -gen-typedef-decls -typedefs-dialect=ZKLeanLean)
mlir_tablegen(ZKLeanLeanTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ZKLeanLean)

set(LLVM_TARGET_DEFINITIONS "ZKLeanLeanOps.td")
mlir_tablegen(ZKLeanLeanOps.h.inc -gen-op-decls -dialect=ZKLeanLean)
mlir_tablegen(ZKLeanLeanOps.cpp.inc -gen-op-defs -dialect=ZKLeanLean)

llzk_add_mlir_doc(ZKLeanLeanDocGen dialect/ZKLeanLeanDialect.md -gen-dialect-doc -dialect=ZKLeanLean)

add_public_tablegen_target(ZKLeanLeanDialectIncGen)
add_dependencies(mlir-headers ZKLeanLeanDialectIncGen)
add_dependencies(ZKLeanDialectHeaders ZKLeanLeanDialectIncGen)
