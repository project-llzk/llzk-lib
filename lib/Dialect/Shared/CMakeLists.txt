add_library(LLZKDialectShared)
# add_library(LLZK::LLZKDialectShared ALIAS LLZKDialectShared)
target_link_libraries(LLZKAllDialects INTERFACE LLZKDialectShared)

target_sources(LLZKDialectShared PRIVATE Builders.cpp)
target_link_libraries(
  LLZKDialectShared
  PUBLIC LLZKDialectHeaders
         ${LLZK_DEP_DIALECT_LIBS}
         MLIRIR
         MLIRParser
         LLVMHeaders
         MLIRHeaders
         LLZKUtil)
llzk_target_add_mlir_link_settings(LLZKDialectShared)

install(TARGETS LLZKDialectShared EXPORT LLZKTargets)

add_library(LLZKSharedOpInterfaces)
target_sources(LLZKSharedOpInterfaces
               PRIVATE DiscardableAllocationOpInterfaces.cpp)
target_link_libraries(LLZKSharedOpInterfaces PUBLIC LLZKDialectHeaders MLIRHeaders MLIRIR)
llzk_target_add_mlir_link_settings(LLZKSharedOpInterfaces)
install(TARGETS LLZKSharedOpInterfaces EXPORT LLZKTargets)
