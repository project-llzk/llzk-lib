include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "DiscardableAllocationOpInterfaces.td")
mlir_tablegen(DiscardableAllocationOpInterfaces.h.inc --gen-op-interface-decls)
mlir_tablegen(DiscardableAllocationOpInterfaces.cpp.inc --gen-op-interface-defs)

add_public_tablegen_target(SharedDialectIncGen)
add_dependencies(LLZKDialectHeaders SharedDialectIncGen)
