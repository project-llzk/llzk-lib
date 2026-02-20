include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=string)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=string)
llzk_tablegen(Dialect.capi.test.cpp.inc -gen-dialect-capi-tests -prefix=llzk -dialect=string)

set(LLVM_TARGET_DEFINITIONS "Types.td")
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=string)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=string)
llzk_tablegen(Types.capi.h.inc -gen-type-capi-header -prefix=llzk -dialect=string)
llzk_tablegen(Types.capi.cpp.inc -gen-type-capi-impl -prefix=llzk -dialect=string)
llzk_tablegen(Types.capi.test.cpp.inc -gen-type-capi-tests -prefix=llzk -dialect=string)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=string)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=string)
llzk_tablegen(Ops.capi.h.inc -gen-op-capi-header -prefix=llzk -dialect=string)
llzk_tablegen(Ops.capi.cpp.inc -gen-op-capi-impl -prefix=llzk -dialect=string)
llzk_tablegen(Ops.capi.test.cpp.inc -gen-op-capi-tests -prefix=llzk -dialect=string)

llzk_add_mlir_doc(StringOpsDocGen dialect/StringDialect.md -gen-dialect-doc -dialect=string)

add_public_tablegen_target(StringDialectIncGen)
add_dependencies(mlir-headers StringDialectIncGen)
add_dependencies(LLZKDialectHeaders StringDialectIncGen)
