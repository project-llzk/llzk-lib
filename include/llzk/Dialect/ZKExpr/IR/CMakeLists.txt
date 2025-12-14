include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "ZKExprDialect.td")
mlir_tablegen(ZKExprDialect.h.inc -gen-dialect-decls -dialect=ZKExpr)
mlir_tablegen(ZKExprDialect.cpp.inc -gen-dialect-defs -dialect=ZKExpr)

set(LLVM_TARGET_DEFINITIONS "ZKExprTypes.td")
mlir_tablegen(ZKExprTypes.h.inc -gen-typedef-decls -typedefs-dialect=ZKExpr)
mlir_tablegen(ZKExprTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ZKExpr)

set(LLVM_TARGET_DEFINITIONS "ZKExprOps.td")
mlir_tablegen(ZKExprOps.h.inc -gen-op-decls -dialect=ZKExpr)
mlir_tablegen(ZKExprOps.cpp.inc -gen-op-defs -dialect=ZKExpr)

add_public_tablegen_target(ZKExprDialectIncGen)
add_dependencies(mlir-headers ZKExprDialectIncGen)
add_dependencies(LLZKDialectHeaders ZKExprDialectIncGen)
