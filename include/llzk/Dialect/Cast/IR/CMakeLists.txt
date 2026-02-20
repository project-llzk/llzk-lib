include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=cast)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=cast)
llzk_tablegen(Dialect.capi.test.cpp.inc -gen-dialect-capi-tests -prefix=llzk -dialect=cast)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=cast)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=cast)
llzk_tablegen(Ops.capi.h.inc -gen-op-capi-header -prefix=llzk -dialect=cast)
llzk_tablegen(Ops.capi.cpp.inc -gen-op-capi-impl -prefix=llzk -dialect=cast)
llzk_tablegen(Ops.capi.test.cpp.inc -gen-op-capi-tests -prefix=llzk -dialect=cast)

llzk_add_mlir_doc(CastOpsDocGen dialect/CastDialect.md -gen-dialect-doc -dialect=cast)

add_public_tablegen_target(CastDialectIncGen)
add_dependencies(mlir-headers CastDialectIncGen)
add_dependencies(LLZKDialectHeaders CastDialectIncGen)
