//===- CommonAttrOrTypeCAPITestGen.h - Common test generation utilities ---===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides common utilities for generating C API link tests for
/// attributes and types. The test generation strategy is:
///
/// **Test Philosophy:**
/// These are link-time verification tests, not functional tests. They ensure
/// that all generated C API functions compile correctly and link properly,
/// catching issues like:
/// - Missing function definitions (link errors)
/// - Signature mismatches between header and implementation
/// - Missing symbols in the build system
/// - ABI compatibility problems
/// - Breaking changes from refactoring
///
/// **Test Pattern:**
/// Each test creates a dummy object (IndexType or IntegerAttr from MLIR builtins)
/// and wraps the C API function call inside a conditional that checks if the
/// dummy object is of the target dialect type. Since the dummy is from a
/// different dialect, the condition is always false at runtime, but the compiler
/// still verifies type correctness and the linker ensures symbol resolution.
///
/// **Limitations:**
/// These tests do NOT verify:
/// - Runtime correctness of the generated code
/// - Semantic behavior of the operations
/// - Error handling paths
/// - Generator logic bugs (if generator is wrong, tests will be wrong too)
///
/// For functional testing, separate integration tests are needed.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/TableGen/AttrOrTypeDef.h>
#include <mlir/TableGen/Dialect.h>

#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

#include "CommonCAPIGen.h"

/// @brief Generate dummy parameters for Get builder (used by both Attr and Type)
/// @param def The attribute or type definition
/// @param isType true if generating for a type, false for an attribute
/// @return String containing dummy parameter declarations
///
/// This function generates C code that declares dummy variables for all parameters
/// of an attribute or type Get builder. For ArrayRef parameters, it generates both
/// a count variable and an array variable. For MlirType/MlirAttribute parameters,
/// it calls helper functions to create test instances.
std::string
generateDummyParamsForAttrOrTypeGet(const mlir::tblgen::AttrOrTypeDef &def, bool isType);

/// @brief Generate parameter list for Get builder call (used by both Attr and Type)
/// @param def The attribute or type definition
/// @return String containing the parameter list
///
/// This function generates a comma-separated list of parameter names to pass to
/// a Get builder function. For ArrayRef parameters, it includes both the count
/// and array pointer. For regular parameters, it includes just the parameter name.
std::string generateParamListForAttrOrTypeGet(const mlir::tblgen::AttrOrTypeDef &def);

/// @brief Base class for attribute and type test generators
///
/// This class provides common functionality for generating unit tests
/// for attributes and types. It extends the base TestGenerator class.
struct AttrOrTypeTestGenerator : public TestGenerator {
  using TestGenerator::TestGenerator;

  virtual ~AttrOrTypeTestGenerator() = default;

  /// @brief Set the parameter name for code generation
  /// @param name The parameter name from the TableGen definition
  void setParamName(mlir::StringRef name) {
    this->paramName = name;
    this->paramNameCapitalized = toPascalCase(name);
  }

  /// @brief Generate Get builder test for a definition
  /// @param dummyParams Dummy parameter declarations
  /// @param paramList Parameter list for the call
  virtual void
  genGetBuilderTest(const std::string &dummyParams, const std::string &paramList) const {
    static constexpr char fmt[] = R"(
// This test ensures {0}{2}_{3}Get links properly.
TEST_F({2}{1}LinkTests, Get_{3}) {{
  auto test{1} = createIndex{1}();

  // We only verify the function compiles and links, wrapped in an unreachable condition
  if ({0}{1}IsA_{2}_{3}(test{1})) {{
{4}
    (void){0}{2}_{3}Get(context{5});
  }
}
)";
    assert(!className.empty() && "className must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,         // {0}
        kind,                   // {1}
        dialectNameCapitalized, // {2}
        className,              // {3}
        dummyParams,            // {4}
        paramList               // {5}
    );
  }

  /// @brief Generate parameter getter test
  virtual void genParamGetterTest() const {
    static constexpr char fmt[] = R"(
// This test ensures {0}{2}_{3}Get{5} links properly.
TEST_F({2}{1}LinkTests, Get_{3}_{4}) {{
  auto test{1} = createIndex{1}();

  if ({0}{1}IsA_{2}_{3}(test{1})) {{
    (void){0}{2}_{3}Get{5}(test{1});
  }
}
)";
    assert(!className.empty() && "className must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialectNameCapitalized, className, paramName,
        paramNameCapitalized
    );
  }

  /// @brief Generate ArrayRef parameter count getter test
  virtual void genArrayRefParamCountTest() const {
    static constexpr char fmt[] = R"(
// This test ensures {0}{2}_{3}Get{5}Count links properly.
TEST_F({2}{1}LinkTests, Get_{3}_{4}Count) {{
  auto test{1} = createIndex{1}();

  if ({0}{1}IsA_{2}_{3}(test{1})) {{
    (void){0}{2}_{3}Get{5}Count(test{1});
  }
}
)";
    assert(!className.empty() && "className must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialectNameCapitalized, className, paramName,
        paramNameCapitalized
    );
  }

  /// @brief Generate ArrayRef parameter element getter test
  virtual void genArrayRefParamAtTest() const {
    static constexpr char fmt[] = R"(
// This test ensures {0}{2}_{3}Get{5}At links properly.
TEST_F({2}{1}LinkTests, Get_{3}_{4}At) {{
  auto test{1} = createIndex{1}();

  if ({0}{1}IsA_{2}_{3}(test{1})) {{
    (void){0}{2}_{3}Get{5}At(test{1}, 0);
  }
}
)";
    assert(!className.empty() && "className must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialectNameCapitalized, className, paramName,
        paramNameCapitalized
    );
  }

  void genCompleteRecord(const mlir::tblgen::AttrOrTypeDef def, bool isType) {
    const mlir::tblgen::Dialect &defDialect = def.getDialect();

    // Generate for the selected dialect only
    if (defDialect.getName() != DialectName) {
      return;
    }

    this->setDialectAndClassName(&defDialect, def.getCppClassName());

    // Generate IsA test
    if (GenIsA) {
      this->genIsATest();
    }

    // Generate Get builder test
    if (GenTypeOrAttrGet && !def.skipDefaultBuilders()) {
      std::string dummyParams = generateDummyParamsForAttrOrTypeGet(def, isType);
      std::string paramList = generateParamListForAttrOrTypeGet(def);
      this->genGetBuilderTest(dummyParams, paramList);
    }

    // Generate parameter getter tests
    if (GenTypeOrAttrParamGetters) {
      for (const auto &param : def.getParameters()) {
        this->setParamName(param.getName());
        mlir::StringRef cppType = param.getCppType();
        if (isArrayRefType(cppType)) {
          this->genArrayRefParamCountTest();
          this->genArrayRefParamAtTest();
        } else {
          this->genParamGetterTest();
        }
      }
    }

    // Generate extra class method tests
    if (GenExtraClassMethods) {
      std::optional<mlir::StringRef> extraDecls = def.getExtraDecls();
      if (extraDecls.has_value()) {
        this->genExtraMethods(extraDecls.value());
      }
    }
  }

protected:
  mlir::StringRef paramName;
  std::string paramNameCapitalized;
};
