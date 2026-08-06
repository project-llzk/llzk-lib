add_library(R1CSTarget)
add_library(R1CS::Target ALIAS R1CSTarget)

target_sources(R1CSTarget PRIVATE R1CSBinary.cpp TranslateRegistration.cpp)
target_link_libraries(
  R1CSTarget
  PUBLIC
    R1CSDialectHeaders
    MLIRIR
    MLIRTranslateLib
    LLVMHeaders
    MLIRHeaders
  PRIVATE
    LLZKUtil
    R1CSDialect
)

llzk_target_add_mlir_link_settings(R1CSTarget)
install(TARGETS R1CSTarget EXPORT LLZKTargets)
