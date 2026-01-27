include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "ZKLeanStructDialect.td")
mlir_tablegen(ZKLeanStructDialect.h.inc -gen-dialect-decls -dialect=ZKLeanStruct)
mlir_tablegen(ZKLeanStructDialect.cpp.inc -gen-dialect-defs -dialect=ZKLeanStruct)

set(LLVM_TARGET_DEFINITIONS "ZKLeanStructTypes.td")
mlir_tablegen(ZKLeanStructTypes.h.inc -gen-typedef-decls -typedefs-dialect=ZKLeanStruct)
mlir_tablegen(ZKLeanStructTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ZKLeanStruct)

set(LLVM_TARGET_DEFINITIONS "ZKLeanStructOps.td")
mlir_tablegen(ZKLeanStructOps.h.inc -gen-op-decls -dialect=ZKLeanStruct)
mlir_tablegen(ZKLeanStructOps.cpp.inc -gen-op-defs -dialect=ZKLeanStruct)

add_public_tablegen_target(ZKLeanStructDialectIncGen)
add_dependencies(mlir-headers ZKLeanStructDialectIncGen)
add_dependencies(LLZKDialectHeaders ZKLeanStructDialectIncGen)
