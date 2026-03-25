add_library(ZKLeanTransforms)
add_library(ZKLean::Transforms ALIAS ZKLeanTransforms)

file(GLOB ZKLeanTransforms_SOURCES "*.cpp")
target_sources(ZKLeanTransforms PRIVATE ${ZKLeanTransforms_SOURCES})

target_link_libraries(
  ZKLeanTransforms
  PUBLIC
    ZKLeanDialectHeaders
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
    LLVMHeaders MLIRHeaders
  PRIVATE
    LLZKDialect
    LLZKAnalysis
    ZKLeanZKBuilderDialect
    ZKLeanZKExprDialect
    ZKLeanZKLeanLeanDialect)

llzk_target_add_mlir_link_settings(ZKLeanTransforms)
install(TARGETS ZKLeanTransforms EXPORT LLZKTargets)
