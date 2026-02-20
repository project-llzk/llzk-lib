include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=poly)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=poly)
llzk_tablegen(Dialect.capi.test.cpp.inc -gen-dialect-capi-tests -prefix=llzk -dialect=poly)

set(LLVM_TARGET_DEFINITIONS "Types.td")
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=poly)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=poly)
# Skip generating attribute getters for types because TypeVarType can be created from either a StringAttr
# or a FlatSymbolRefAttr, which are both MlirAttribute in the C API, so they must be handled together manually.
llzk_tablegen(Types.capi.h.inc -gen-type-capi-header -prefix=llzk -dialect=poly -gen-type-attr-get=false)
llzk_tablegen(Types.capi.cpp.inc -gen-type-capi-impl -prefix=llzk -dialect=poly -gen-type-attr-get=false)
llzk_tablegen(Types.capi.test.cpp.inc -gen-type-capi-tests -prefix=llzk -dialect=poly -gen-type-attr-get=false)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=poly)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=poly)
llzk_tablegen(Ops.capi.h.inc -gen-op-capi-header -prefix=llzk -dialect=poly)
llzk_tablegen(Ops.capi.cpp.inc -gen-op-capi-impl -prefix=llzk -dialect=poly)
llzk_tablegen(Ops.capi.test.cpp.inc -gen-op-capi-tests -prefix=llzk -dialect=poly)

llzk_add_mlir_doc(PolymorphicOpsDocGen dialect/PolymorphicDialect.md -gen-dialect-doc -dialect=poly)

add_public_tablegen_target(PolymorphicDialectIncGen)
add_dependencies(mlir-headers PolymorphicDialectIncGen)
add_dependencies(LLZKDialectHeaders PolymorphicDialectIncGen)
