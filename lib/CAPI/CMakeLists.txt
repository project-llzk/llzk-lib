add_library(LLZKCAPI)
add_library(LLZK::LLZKCAPI ALIAS LLZKCAPI)

target_sources(LLZKCAPI PRIVATE 
  Transforms.cpp 
  Validators.cpp 
  InitDialects.cpp 
  Typing.cpp
  Builder.cpp 
  Support.cpp
  Constants.cpp
  Support.cpp
  Dialect/Array.cpp 
  Dialect/Bool.cpp 
  Dialect/Cast.cpp 
  Dialect/Constrain.cpp 
  Dialect/Felt.cpp 
  Dialect/Function.cpp 
  Dialect/Global.cpp 
  Dialect/Include.cpp 
  Dialect/LLZK.cpp 
  Dialect/POD.cpp
  Dialect/Poly.cpp 
  Dialect/String.cpp 
  Dialect/Struct.cpp 
  Dialect/Undef.cpp)
target_link_libraries(
  LLZKCAPI PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} 
  MLIRIR
  MLIRParser 
  LLVMHeaders 
  MLIRHeaders 
  MLIRCAPIIR 
  MLIRCAPIRegisterEverything 
  LLZK::DialectRegistration
  LLZK::LLZKTransforms 
  LLZK::LLZKValidators
  LLZK::LLZKAnalysis 
)
llzk_target_add_mlir_link_settings(LLZKCAPI)

install(TARGETS LLZKCAPI EXPORT LLZKTargets)
