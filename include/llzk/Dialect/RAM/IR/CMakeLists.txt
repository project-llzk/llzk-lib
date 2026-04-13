include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=ram)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=ram)
llzk_tablegen(Dialect.capi.test.cpp.inc -gen-dialect-capi-tests -prefix=llzk -dialect=ram)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=ram)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=ram)
llzk_tablegen(Ops.capi.h.inc -gen-op-capi-header -prefix=llzk -dialect=ram)
llzk_tablegen(Ops.capi.cpp.inc -gen-op-capi-impl -prefix=llzk -dialect=ram)
llzk_tablegen(Ops.capi.test.cpp.inc -gen-op-capi-tests -prefix=llzk -dialect=ram)

llzk_add_mlir_doc(RAMOpsDocGen dialect/RAMDialect.md -gen-dialect-doc -dialect=ram)

add_public_tablegen_target(RAMDialectIncGen)
add_dependencies(LLZKDialectHeaders RAMDialectIncGen)
