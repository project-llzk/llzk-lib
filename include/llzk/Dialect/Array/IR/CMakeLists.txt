include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=array)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=array)
llzk_tablegen(Dialect.capi.test.cpp.inc -gen-dialect-capi-tests -prefix=llzk -dialect=array)

set(LLVM_TARGET_DEFINITIONS "Types.td")
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=array)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=array)
llzk_tablegen(Types.capi.h.inc -gen-type-capi-header -prefix=llzk -dialect=array)
llzk_tablegen(Types.capi.cpp.inc -gen-type-capi-impl -prefix=llzk -dialect=array)
llzk_tablegen(Types.capi.test.cpp.inc -gen-type-capi-tests -prefix=llzk -dialect=array)

set(LLVM_TARGET_DEFINITIONS "OpInterfaces.td")
mlir_tablegen(OpInterfaces.h.inc --gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc --gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=array)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=array)
llzk_tablegen(Ops.capi.h.inc -gen-op-capi-header -prefix=llzk -dialect=array)
llzk_tablegen(Ops.capi.cpp.inc -gen-op-capi-impl -prefix=llzk -dialect=array)
llzk_tablegen(Ops.capi.test.cpp.inc -gen-op-capi-tests -prefix=llzk -dialect=array)

llzk_add_mlir_doc(ArrayOpsDocGen dialect/ArrayDialect.md -gen-dialect-doc -dialect=array)

add_public_tablegen_target(ArrayDialectIncGen)
add_dependencies(mlir-headers ArrayDialectIncGen)
add_dependencies(LLZKDialectHeaders ArrayDialectIncGen)
