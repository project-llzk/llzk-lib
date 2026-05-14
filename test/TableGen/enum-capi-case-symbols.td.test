// RUN: llzk-tblgen %tablegen_includes -gen-enum-capi-header -prefix=llzk -dialect=test %s | FileCheck %s --check-prefix=HEADER
// RUN: llzk-tblgen %tablegen_includes -gen-enum-capi-tests -prefix=llzk -dialect=test %s | FileCheck %s --check-prefix=TEST

include "mlir/IR/EnumAttr.td"

def LowerCase : I32EnumAttrCase<"lowerCase", 0, "lower-case">;
def MixedCase : I32EnumAttrCase<"MixedCase", 1, "mixed-case">;
def UpperCase : I32EnumAttrCase<"UPPER_CASE", 2, "upper-case">;

let cppNamespace = "::llzk::test" in
def CaseSymbolEnum
    : I32EnumAttr<"CaseSymbolEnum", "enum with case-sensitive symbols",
                  [LowerCase, MixedCase, UpperCase]>;

// HEADER: LlzkTestCaseSymbolEnum_lowerCase = 0
// HEADER: LlzkTestCaseSymbolEnum_MixedCase = 1
// HEADER: LlzkTestCaseSymbolEnum_UPPER_CASE = 2
// HEADER-NOT: LlzkTestCaseSymbolEnum_LOWERCASE

// TEST: LlzkTestCaseSymbolEnum enumValue = LlzkTestCaseSymbolEnum_lowerCase;
// TEST: EXPECT_EQ(enumValue, LlzkTestCaseSymbolEnum_lowerCase);
// TEST: LlzkTestCaseSymbolEnum cValue = LlzkTestCaseSymbolEnum_lowerCase;
// TEST-NOT: LlzkTestCaseSymbolEnum_LOWERCASE
