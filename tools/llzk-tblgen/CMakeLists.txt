add_tablegen(
  llzk-tblgen
  LLZK
  DESTINATION
  "${CMAKE_BINARY_DIR}/bin"
  EXPORT
  LLZKTargets
  CommonCAPIGen.cpp
  CommonAttrOrTypeCAPITestGen.cpp
  AttrCAPIGen.cpp
  AttrCAPITestGen.cpp
  TypeCAPIGen.cpp
  TypeCAPITestGen.cpp
  EnumCAPIGen.cpp
  EnumCAPITestGen.cpp
  OpCAPIParamHelper.cpp
  OpCAPIGen.cpp
  OpCAPITestGen.cpp
  DialectCAPITestGen.cpp
  llzk-tblgen.cpp)

target_link_libraries(
  llzk-tblgen
  PRIVATE MLIRTblgenLib
          LLVMSupport
          $<$<BOOL:${LLZK_TBLGEN_USE_LIBCLANGCPP_BOOL}>:clang-cpp>
          $<$<NOT:$<BOOL:${LLZK_TBLGEN_USE_LIBCLANGCPP_BOOL}>>:clangBasic>
          $<$<NOT:$<BOOL:${LLZK_TBLGEN_USE_LIBCLANGCPP_BOOL}>>:clangLex>
          LLVMHeaders
          MLIRHeaders)

mlir_check_all_link_libraries(llzk-tblgen)
