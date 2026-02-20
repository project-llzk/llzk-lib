include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=struct)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=struct)
llzk_tablegen(Dialect.capi.test.cpp.inc -gen-dialect-capi-tests -prefix=llzk -dialect=struct)

set(LLVM_TARGET_DEFINITIONS "Types.td")
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=struct)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=struct)
llzk_tablegen(Types.capi.h.inc -gen-type-capi-header -prefix=llzk -dialect=struct)
llzk_tablegen(Types.capi.cpp.inc -gen-type-capi-impl -prefix=llzk -dialect=struct)
llzk_tablegen(Types.capi.test.cpp.inc -gen-type-capi-tests -prefix=llzk -dialect=struct)

set(LLVM_TARGET_DEFINITIONS "OpInterfaces.td")
mlir_tablegen(OpInterfaces.h.inc --gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc --gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=struct)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=struct)
llzk_tablegen(Ops.capi.h.inc -gen-op-capi-header -prefix=llzk -dialect=struct)
llzk_tablegen(Ops.capi.cpp.inc -gen-op-capi-impl -prefix=llzk -dialect=struct)
llzk_tablegen(Ops.capi.test.cpp.inc -gen-op-capi-tests -prefix=llzk -dialect=struct)

llzk_add_mlir_doc(StructOpsDocGen dialect/StructDialect.md -gen-dialect-doc -dialect=struct)

add_public_tablegen_target(StructDialectIncGen)
add_dependencies(mlir-headers StructDialectIncGen)
add_dependencies(LLZKDialectHeaders StructDialectIncGen)
