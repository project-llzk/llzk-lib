include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "SMTPasses.td")
mlir_tablegen(SMTPasses.h.inc -gen-pass-decls -name=SMT)

add_public_tablegen_target(SMTPassesIncGen)
add_dependencies(SMTDialectHeaders SMTPassesIncGen)
