include_directories(${MLIR_INCLUDE_DIRS} ${PCL_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=pcl -I backends/pcl/include/)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=pcl -I backends/pcl/include/)

set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=pcl -I backends/pcl/include/)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=pcl -I backends/pcl/include/)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=pcl -I backends/pcl/include/)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=pcl -I backends/pcl/include/)

set(LLVM_TARGET_DEFINITIONS Attrs.td)
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=pcl -I backends/pcl/include/)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=pcl -I backends/pcl/include/)

add_public_tablegen_target(PCLDialectIncGen)
add_dependencies(mlir-headers PCLDialectIncGen)
add_dependencies(PCLDialectHeaders PCLDialectIncGen)
