include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../ZKExpr/IR)

set(LLVM_TARGET_DEFINITIONS "ZKBuilderDialect.td")
mlir_tablegen(ZKBuilderDialect.h.inc -gen-dialect-decls -dialect=ZKBuilder)
mlir_tablegen(ZKBuilderDialect.cpp.inc -gen-dialect-defs -dialect=ZKBuilder)

set(LLVM_TARGET_DEFINITIONS "ZKBuilderTypes.td")
mlir_tablegen(ZKBuilderTypes.h.inc -gen-typedef-decls -typedefs-dialect=ZKBuilder)
mlir_tablegen(ZKBuilderTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ZKBuilder)

set(LLVM_TARGET_DEFINITIONS "ZKBuilderOps.td")
mlir_tablegen(ZKBuilderOps.h.inc -gen-op-decls -dialect=ZKBuilder)
mlir_tablegen(ZKBuilderOps.cpp.inc -gen-op-defs -dialect=ZKBuilder)

llzk_add_mlir_doc(ZKBuilderDocGen dialect/ZKBuilderDialect.md -gen-dialect-doc -dialect=ZKBuilder)

add_public_tablegen_target(ZKBuilderDialectIncGen)
add_dependencies(mlir-headers ZKBuilderDialectIncGen)
add_dependencies(LLZKDialectHeaders ZKBuilderDialectIncGen)
