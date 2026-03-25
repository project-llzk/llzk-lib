include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "AnalysisPasses.td")
mlir_tablegen(AnalysisPasses.h.inc -gen-pass-decls -name=Analysis)
mlir_tablegen(AnalysisPassEnums.h.inc -gen-enum-decls)
mlir_tablegen(AnalysisPassEnums.cpp.inc -gen-enum-defs)
llzk_add_mlir_doc(AnalysisPassesDocGen passes/AnalysisPasses.md -gen-pass-doc)

add_public_tablegen_target(LLZKAnalysisIncGen)
add_dependencies(LLZKDialectHeaders LLZKAnalysisIncGen)
