include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=llzk)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=llzk)
llzk_tablegen(Dialect.capi.test.cpp.inc -gen-dialect-capi-tests -prefix=llzk -dialect=llzk)

set(LLVM_TARGET_DEFINITIONS "Attrs.td")
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=llzk)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=llzk)
llzk_tablegen(Attrs.capi.h.inc -gen-attr-capi-header -prefix=llzk -dialect=llzk)
llzk_tablegen(Attrs.capi.cpp.inc -gen-attr-capi-impl -prefix=llzk -dialect=llzk)
llzk_tablegen(Attrs.capi.test.cpp.inc -gen-attr-capi-tests -prefix=llzk -dialect=llzk)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=llzk)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=llzk)
llzk_tablegen(Ops.capi.h.inc -gen-op-capi-header -prefix=llzk -dialect=llzk)
llzk_tablegen(Ops.capi.cpp.inc -gen-op-capi-impl -prefix=llzk -dialect=llzk)
llzk_tablegen(Ops.capi.test.cpp.inc -gen-op-capi-tests -prefix=llzk -dialect=llzk)

llzk_add_mlir_doc(LLZKOpsDocGen dialect/LLZKDialect.md -gen-dialect-doc -dialect=llzk)

add_public_tablegen_target(LLZKDialectIncGen)
add_dependencies(mlir-headers LLZKDialectIncGen)
add_dependencies(LLZKDialectHeaders LLZKDialectIncGen)
