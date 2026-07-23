include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=channel)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=channel)
llzk_tablegen(Dialect.capi.test.cpp.inc -gen-dialect-capi-tests -prefix=llzk -dialect=channel)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=channel)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=channel)
llzk_tablegen(Ops.capi.h.inc -gen-op-capi-header -prefix=llzk -dialect=channel)
llzk_tablegen(Ops.capi.cpp.inc -gen-op-capi-impl -prefix=llzk -dialect=channel)
llzk_tablegen(Ops.capi.test.cpp.inc -gen-op-capi-tests -prefix=llzk -dialect=channel)

llzk_add_mlir_doc(ChannelOpsDocGen dialect/ChannelDialect.md -gen-dialect-doc -dialect=channel)

add_public_tablegen_target(ChannelDialectIncGen)
add_dependencies(LLZKDialectHeaders ChannelDialectIncGen)
