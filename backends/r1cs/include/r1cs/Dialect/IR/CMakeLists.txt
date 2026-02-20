include_directories(${MLIR_INCLUDE_DIRS} ${R1CS_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=r1cs)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=r1cs)

set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=r1cs)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=r1cs)

set(LLVM_TARGET_DEFINITIONS Attrs.td)
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=r1cs)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=r1cs)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=r1cs)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=r1cs)

# Optional: if you later define interfaces
# set(LLVM_TARGET_DEFINITIONS "OpInterfaces.td")
# mlir_tablegen(OpInterfaces.h.inc --gen-op-interface-decls)
# mlir_tablegen(OpInterfaces.cpp.inc --gen-op-interface-defs)

llzk_add_mlir_doc(R1CSOpsDocGen dialect/R1CSDialect.md -gen-dialect-doc -dialect=r1cs)

add_public_tablegen_target(R1CSDialectIncGen)
add_dependencies(mlir-headers R1CSDialectIncGen)
add_dependencies(LLZKDialectHeaders R1CSDialectIncGen)
