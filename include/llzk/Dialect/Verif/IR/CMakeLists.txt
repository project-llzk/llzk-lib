include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=verif)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=verif)
llzk_tablegen(Dialect.capi.test.cpp.inc -gen-dialect-capi-tests -prefix=llzk -dialect=verif)

set(LLVM_TARGET_DEFINITIONS "OpInterfaces.td")
mlir_tablegen(OpInterfaces.h.inc --gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc --gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=verif)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=verif)
llzk_tablegen(Ops.capi.h.inc -gen-op-capi-header -prefix=llzk -dialect=verif)
llzk_tablegen(Ops.capi.cpp.inc -gen-op-capi-impl -prefix=llzk -dialect=verif)
llzk_tablegen(Ops.capi.test.cpp.inc -gen-op-capi-tests -prefix=llzk -dialect=verif)

llzk_add_mlir_doc(VerifOpsDocGen dialect/VerifDialect.md -gen-dialect-doc -dialect=verif)

add_public_tablegen_target(VerifDialectIncGen)
add_dependencies(LLZKDialectHeaders VerifDialectIncGen)
