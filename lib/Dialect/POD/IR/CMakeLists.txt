add_library(LLZKPODDialect)
add_library(LLZK::LLZKPODDialect ALIAS LLZKPODDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKPODDialect)

target_sources(LLZKPODDialect PRIVATE
  Dialect.cpp
  Ops.cpp
  Types.cpp
  Attrs.cpp
)
target_link_libraries(
  LLZKPODDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser  
                     LLVMHeaders MLIRHeaders)
llzk_target_add_mlir_link_settings(LLZKPODDialect)

install(TARGETS LLZKPODDialect EXPORT LLZKTargets)
