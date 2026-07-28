include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=cast)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=cast)
llzk_tablegen(Dialect.capi.test.cpp.inc -gen-dialect-capi-tests -prefix=llzk -dialect=cast)

set(LLVM_TARGET_DEFINITIONS "Attrs.td")
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=cast)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=cast)
llzk_tablegen(Attrs.capi.h.inc -gen-attr-capi-header -prefix=llzk -dialect=cast)
llzk_tablegen(Attrs.capi.cpp.inc -gen-attr-capi-impl -prefix=llzk -dialect=cast)
llzk_tablegen(Attrs.capi.test.cpp.inc -gen-attr-capi-tests -prefix=llzk -dialect=cast)

set(LLVM_TARGET_DEFINITIONS "Enums.td")
mlir_tablegen(Enums.h.inc -gen-enum-decls)
mlir_tablegen(Enums.cpp.inc -gen-enum-defs)
llzk_tablegen(Enums.capi.h.inc -gen-enum-capi-header -prefix=llzk -dialect=cast)
llzk_tablegen(Enums.capi.cpp.inc -gen-enum-capi-impl -prefix=llzk -dialect=cast)
llzk_tablegen(Enums.capi.test.cpp.inc -gen-enum-capi-tests -prefix=llzk -dialect=cast)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=cast)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=cast)
llzk_tablegen(Ops.capi.h.inc -gen-op-capi-header -prefix=llzk -dialect=cast)
llzk_tablegen(Ops.capi.cpp.inc -gen-op-capi-impl -prefix=llzk -dialect=cast)
llzk_tablegen(Ops.capi.test.cpp.inc -gen-op-capi-tests -prefix=llzk -dialect=cast)

llzk_add_mlir_doc(CastOpsDocGen dialect/CastDialect.md -gen-dialect-doc -dialect=cast)

add_public_tablegen_target(CastDialectIncGen)
add_dependencies(LLZKDialectHeaders CastDialectIncGen)
