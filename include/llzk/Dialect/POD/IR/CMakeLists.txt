include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=pod)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=pod)
llzk_tablegen(Dialect.capi.test.cpp.inc -gen-dialect-capi-tests -prefix=llzk -dialect=pod)

set(LLVM_TARGET_DEFINITIONS "Attrs.td")
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=pod)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=pod)
llzk_tablegen(Attrs.capi.h.inc -gen-attr-capi-header -prefix=llzk -dialect=pod)
llzk_tablegen(Attrs.capi.cpp.inc -gen-attr-capi-impl -prefix=llzk -dialect=pod)
llzk_tablegen(Attrs.capi.test.cpp.inc -gen-attr-capi-tests -prefix=llzk -dialect=pod)

set(LLVM_TARGET_DEFINITIONS "Types.td")
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=pod)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=pod)
llzk_tablegen(Types.capi.h.inc -gen-type-capi-header -prefix=llzk -dialect=pod)
llzk_tablegen(Types.capi.cpp.inc -gen-type-capi-impl -prefix=llzk -dialect=pod)
llzk_tablegen(Types.capi.test.cpp.inc -gen-type-capi-tests -prefix=llzk -dialect=pod)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=pod)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=pod)
llzk_tablegen(Ops.capi.h.inc -gen-op-capi-header -prefix=llzk -dialect=pod)
llzk_tablegen(Ops.capi.cpp.inc -gen-op-capi-impl -prefix=llzk -dialect=pod)
llzk_tablegen(Ops.capi.test.cpp.inc -gen-op-capi-tests -prefix=llzk -dialect=pod)

llzk_add_mlir_doc(PODOpsDocGen dialect/PODDialect.md -gen-dialect-doc -dialect=pod)

add_public_tablegen_target(PODDialectIncGen)
add_dependencies(mlir-headers PODDialectIncGen)
add_dependencies(LLZKDialectHeaders PODDialectIncGen)
