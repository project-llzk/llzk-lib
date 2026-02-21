include_directories(${MLIR_INCLUDE_DIRS} ${ZKLEAN_INCLUDE_DIR} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "ZKLeanPasses.td")
mlir_tablegen(ZKLeanPasses.h.inc -gen-pass-decls -name=ZKLean)

add_public_tablegen_target(ZKLeanPassesIncGen)
add_dependencies(mlir-headers ZKLeanPassesIncGen)
add_dependencies(ZKLeanDialectHeaders ZKLeanPassesIncGen)
