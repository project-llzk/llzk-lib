include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=constrain)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=constrain)
llzk_tablegen(Dialect.capi.test.cpp.inc -gen-dialect-capi-tests -prefix=llzk -dialect=constrain)

set(LLVM_TARGET_DEFINITIONS "OpInterfaces.td")
mlir_tablegen(OpInterfaces.h.inc --gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc --gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=constrain)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=constrain)
llzk_tablegen(Ops.capi.h.inc -gen-op-capi-header -prefix=llzk -dialect=constrain)
llzk_tablegen(Ops.capi.cpp.inc -gen-op-capi-impl -prefix=llzk -dialect=constrain)
llzk_tablegen(Ops.capi.test.cpp.inc -gen-op-capi-tests -prefix=llzk -dialect=constrain)

llzk_add_mlir_doc(ConstrainOpsDocGen dialect/ConstrainDialect.md -gen-dialect-doc -dialect=constrain)

add_public_tablegen_target(ConstrainDialectIncGen)
add_dependencies(mlir-headers ConstrainDialectIncGen)
add_dependencies(LLZKDialectHeaders ConstrainDialectIncGen)
