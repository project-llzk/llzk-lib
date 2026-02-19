//===- CommonCAPIGen.h - Common utilities for C API generation ------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Common utilities shared between all CAPI generators (ops, attrs, types)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/TableGen/Dialect.h>

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>

#include <memory>
#include <string>

constexpr bool WARN_SKIPPED_METHODS = false;

/// @brief Print warning about skipping a function.
template <typename S> inline void warnSkipped(const S &methodName, const std::string &message) {
  if (WARN_SKIPPED_METHODS) {
    llvm::errs() << "Warning: Skipping method '" << methodName << "' - " << message << '\n';
  }
}

/// @brief Print warning about skipping a function due to no conversion of C++ type to C API type.
template <typename S>
inline void warnSkippedNoConversion(const S &methodName, const std::string &cppType) {
  if (WARN_SKIPPED_METHODS) {
    warnSkipped(methodName, "no conversion to C API type for '" + cppType + '\'');
  }
}

// Forward declarations for Clang classes
namespace clang {
class Lexer;
class SourceManager;
} // namespace clang

// Shared command-line options used by all CAPI generators
extern llvm::cl::OptionCategory OpGenCat;
extern llvm::cl::opt<std::string> DialectName;
extern llvm::cl::opt<std::string> FunctionPrefix;

// Shared flags for controlling code generation
extern llvm::cl::opt<bool> GenIsA;
extern llvm::cl::opt<bool> GenOpBuild;
extern llvm::cl::opt<bool> GenOpOperandGetters;
extern llvm::cl::opt<bool> GenOpOperandSetters;
extern llvm::cl::opt<bool> GenOpAttributeGetters;
extern llvm::cl::opt<bool> GenOpAttributeSetters;
extern llvm::cl::opt<bool> GenOpRegionGetters;
extern llvm::cl::opt<bool> GenOpResultGetters;
extern llvm::cl::opt<bool> GenTypeOrAttrGet;
extern llvm::cl::opt<bool> GenTypeOrAttrParamGetters;
extern llvm::cl::opt<bool> GenExtraClassMethods;

/// @brief Convert names separated by underscore or colon to PascalCase.
/// @param str The input string to convert (may contain underscores or colons)
/// @return The converted PascalCase string
///
/// Examples:
///   "no_inline" -> "NoInline"
///   "::llzk::boolean::Type" -> "LlzkBooleanType"
inline std::string toPascalCase(mlir::StringRef str) {
  if (str.empty()) {
    return "";
  }

  std::string result;
  result.reserve(str.size());
  llvm::raw_string_ostream resultStream(result);
  bool capitalizeNext = true;

  for (char c : str) {
    if (c == '_' || c == ':') {
      capitalizeNext = true;
    } else {
      resultStream << (capitalizeNext ? llvm::toUpper(c) : c);
      capitalizeNext = false;
    }
  }

  return result;
}

/// @brief Check if a C++ type is a known integer type
/// @param type The type string to check
/// @return true if the type is an integer type (size_t, unsigned, int*, uint*, etc.)
inline bool isIntegerType(mlir::StringRef type) {
  // Consume optional root namespace token
  type.consume_front("::");
  // Handle special names first
  if (type == "signed" || type == "unsigned" || type == "size_t" || type == "char32_t" ||
      type == "char16_t" || type == "char8_t" || type == "wchar_t") {
    return true;
  }
  // Handle standard integer types with optional signed/unsigned prefix
  type.consume_front("signed ") || type.consume_front("unsigned ");
  if (type == "char" || type == "int" || type == "short" || type == "short int" || type == "long" ||
      type == "long int" || type == "long long" || type == "long long int") {
    return true;
  }
  // Handle fixed-width integer types (https://cppreference.com/w/cpp/types/integer.html)
  type.consume_front("std::"); // optional
  if (type.consume_back("_t") && (type.consume_front("int") || type.consume_front("uint"))) {
    // intmax_t, intptr_t, uintmax_t, uintptr_t
    if (type == "max" || type == "ptr") {
      return true;
    }
    // Optional "_fast" or "_least" and finally bit width to cover the rest
    type.consume_back("_fast") || type.consume_back("_least");
    if (type == "8" || type == "16" || type == "32" || type == "64") {
      return true;
    }
  }
  return false;
}

/// @brief Check if a C++ type is a known primitive type
/// @param cppType The C++ type string to check
/// @return true if the type is a primitive (bool, void, int, etc.)
///
/// @note This function must be called on the CPP type because after converting to CAPI type, some
/// things like APInt become primitive which can lead to missing wrap/unwrap functions.
inline bool isPrimitiveType(mlir::StringRef cppType) {
  cppType.consume_front("::");
  return cppType == "void" || cppType == "bool" || cppType == "float" || cppType == "double" ||
         cppType == "long double" || isIntegerType(cppType);
}

/// @brief Check if a token text represents a C++ modifier/specifier keyword
/// @param tokenText The token to check
/// @return true if the token is a C++ modifier (inline, static, virtual, etc.)
inline bool isCppModifierKeyword(mlir::StringRef tokenText) {
  return llvm::StringSwitch<bool>(tokenText)
      .Case("inline", true)
      .Case("static", true)
      .Case("virtual", true)
      .Case("explicit", true)
      .Case("constexpr", true)
      .Case("consteval", true)
      .Case("extern", true)
      .Case("mutable", true)
      .Case("friend", true)
      .Default(false);
}

/// @brief Check if a method name represents a C++ control flow keyword or language construct
/// @param methodName The method name to check
/// @return true if the name is a C++ language construct (if, for, while, etc.)
inline bool isCppLanguageConstruct(mlir::StringRef methodName) {
  return llvm::StringSwitch<bool>(methodName)
      .Case("if", true)
      .Case("for", true)
      .Case("while", true)
      .Case("switch", true)
      .Case("return", true)
      .Case("sizeof", true)
      .Case("decltype", true)
      .Case("alignof", true)
      .Case("typeid", true)
      .Case("static_assert", true)
      .Case("noexcept", true)
      .Default(false);
}

/// @brief Check if a C++ type is APInt
/// @param cppType The C++ type string to check
/// @return true if the type is APInt, llvm::APInt, or ::llvm::APInt
inline bool isAPIntType(mlir::StringRef cppType) {
  cppType.consume_front("::");
  cppType.consume_front("llvm::") || cppType.consume_front("mlir::");
  return cppType == "APInt";
}

/// @brief Check if a C++ type is an ArrayRef type
/// @param cppType The C++ type string to check
/// @return true if the type is ArrayRef, llvm::ArrayRef, or ::llvm::ArrayRef
inline bool isArrayRefType(mlir::StringRef cppType) {
  cppType.consume_front("::");
  cppType.consume_front("llvm::") || cppType.consume_front("mlir::");
  return cppType.starts_with("ArrayRef<");
}

/// Extract element type from ArrayRef<...>
inline mlir::StringRef extractArrayRefElementType(mlir::StringRef cppType) {
  assert(isArrayRefType(cppType) && "must check `isArrayRefType()` outside");

  // Remove "ArrayRef<" prefix and ">" suffix
  cppType.consume_front("::");
  cppType.consume_front("llvm::") || cppType.consume_front("mlir::");
  cppType.consume_front("ArrayRef<") && cppType.consume_back(">");
  return cppType;
}

/// @brief RAII wrapper for Clang lexer infrastructure
///
/// This class simplifies setting up Clang's lexer for parsing C++ code snippets.
/// It manages the lifetime of all required Clang objects (FileManager, SourceManager,
/// DiagnosticsEngine, etc.) and provides easy access to the lexer.
///
/// The Lexer is used instead of the Parser so that comments preceding method declarations
/// can be captured for documentation generation.
class ClangLexerContext {
public:
  /// @brief Construct a lexer context for the given source code
  /// @param source The C++ source code to lex
  /// @param bufferName Optional name for the memory buffer (for diagnostics)
  explicit ClangLexerContext(mlir::StringRef source, mlir::StringRef bufferName = "input");

  /// @brief Get the lexer instance
  /// @return Reference to the Clang lexer
  clang::Lexer &getLexer() const;

  /// @brief Get the source manager instance
  /// @return Reference to the Clang source manager
  clang::SourceManager &getSourceManager() const;

  /// @brief Check if the lexer was successfully created
  /// @return true if the lexer is valid and ready to use
  bool isValid() const { return lexer != nullptr; }

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
  clang::Lexer *lexer = nullptr;
};

/// @brief Structure to represent a parameter in a parsed method signature from
/// an `extraClassDeclaration`
struct MethodParameter {
  /// The C++ type of the parameter
  std::string type;
  /// The name of the parameter
  std::string name;

  /// @brief Construct a new Method Parameter object
  /// @param paramType
  /// @param paramName
  MethodParameter(const std::string &paramType, const std::string &paramName)
      : type(mlir::StringRef(paramType).trim().str()),
        name(mlir::StringRef(paramName).trim().str()) {}
};

/// @brief Structure to represent a parsed method signature from an `extraClassDeclaration`
///
/// This structure holds information extracted from parsing C++ method declarations.
/// It is used to generate corresponding C API wrapper functions.
struct ExtraMethod {
  /// The C++ return type of the method
  std::string returnType;
  /// The name of the method
  std::string methodName;
  /// Documentation comment (if any)
  std::string documentation;
  /// Whether the method is const-qualified
  bool isConst = false;
  /// Whether the method has parameters (unsupported for now)
  bool hasParameters = false;
  /// The parameters of the method
  std::vector<MethodParameter> parameters;
};

/// @brief Parse method declarations from an `extraClassDeclaration` using Clang's Lexer
/// @param extraDecl The C++ code from an `extraClassDeclaration`
/// @return Vector of parsed method signatures
///
/// This function parses C++ method declarations to extract signatures that can be
/// wrapped in C API functions. It identifies methods by looking for the pattern:
/// [modifiers] <return_type> <identifier> '(' [params] ')' [const] ';'
///
/// Example input:
/// @code
///   /// Get the width of this type
///   unsigned getWidth() const;
///   bool isSignless() const;
/// @endcode
///
/// Example output:
/// - ExtraMethod {
///     returnType="unsigned", methodName="getWidth",
///     documentation="Get the width of this type",
///     isConst=true, hasParameters=false, parameters={}
///   }
/// - ExtraMethod {
///     returnType="bool", methodName="isSignless",
///     documentation="",
///     isConst=true, hasParameters=false, parameters={}
///   }
llvm::SmallVector<ExtraMethod> parseExtraMethods(mlir::StringRef extraDecl);

/// @brief Check if a C++ type matches an MLIR type pattern
/// @param cppType The C++ type to check
/// @param typeName The MLIR type name to match against
/// @return true if the C++ type matches the MLIR type
bool matchesMLIRClass(mlir::StringRef cppType, mlir::StringRef typeName);

/// @brief Convert C++ type to MLIR C API type
/// @param cppType The C++ type to convert
/// @return The corresponding MLIR C API type if convertible, std::nullopt otherwise
std::optional<std::string> tryCppTypeToCapiType(mlir::StringRef cppType);

/// @brief Map C++ type to corresponding C API type
/// @param cppType The C++ type to map
/// @return The corresponding C API type string
///
/// @note This function should not be called directly for ArrayRef types.
/// Use extractArrayRefElementType() first and then use this on the result.
std::string mapCppTypeToCapiType(mlir::StringRef cppType);

/// @brief Base class for C API generators
struct Generator {
  Generator(std::string_view recordKind, llvm::raw_ostream &outputStream)
      : kind(recordKind), os(outputStream), dialectNameCapitalized(toPascalCase(DialectName)) {}
  virtual ~Generator() = default;

  /// @brief Set the dialect and class name for code generation
  /// @param d Pointer to the dialect definition
  /// @param cppClassName The C++ class name of the entity being generated
  virtual void
  setDialectAndClassName(const mlir::tblgen::Dialect *d, mlir::StringRef cppClassName) {
    this->dialect = d;
    this->className = cppClassName;
  }

  /// @brief Generate code for extra methods from an `extraClassDeclaration`
  /// @param extraDecl The extra class declaration string
  virtual void genExtraMethods(mlir::StringRef extraDecl) const {
    if (extraDecl.empty()) {
      return;
    }
    for (const ExtraMethod &method : parseExtraMethods(extraDecl)) {
      genExtraMethod(method);
    }
  }

  /// @brief Generate code for an extra method
  /// @param method The extra method to generate code for
  virtual void genExtraMethod(const ExtraMethod &method) const = 0;

protected:
  std::string kind;
  llvm::raw_ostream &os;
  std::string dialectNameCapitalized;
  const mlir::tblgen::Dialect *dialect;
  mlir::StringRef className;
};

/// @brief Generator for common C header file elements
struct HeaderGenerator : public Generator {
  using Generator::Generator;
  virtual ~HeaderGenerator() = default;

  virtual void genPrologue() const {
    os << R"(
#include "llzk-c/Builder.h"
#include <mlir-c/IR.h>

#ifdef __cplusplus
extern "C" {
#endif
)";
  }

  virtual void genEpilogue() const {
    os << R"(
#ifdef __cplusplus
}
#endif
)";
  }

  virtual void genIsADecl() const {
    static constexpr char fmt[] = R"(
/* Returns true if the {1} is a {4}::{3}. */
MLIR_CAPI_EXPORTED bool {0}{1}IsA_{2}_{3}(Mlir{1});
)";
    assert(dialect && "Dialect must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,            // {0}
        kind,                      // {1}
        dialectNameCapitalized,    // {2}
        className,                 // {3}
        dialect->getCppNamespace() // {4}
    );
  }

  /// @brief Generate declaration for an extra method from an `extraClassDeclaration`
  virtual void genExtraMethod(const ExtraMethod &method) const override {
    // Convert return type to C API type, skip if it can't be converted
    std::optional<std::string> capiReturnTypeOpt = tryCppTypeToCapiType(method.returnType);
    if (!capiReturnTypeOpt.has_value()) {
      warnSkippedNoConversion(method.methodName, method.returnType);
      return;
    }
    std::string capiReturnType = capiReturnTypeOpt.value();

    // Build parameter list
    std::string paramList;
    llvm::raw_string_ostream paramListStream(paramList);
    paramListStream << llvm::formatv("Mlir{0} inp", kind);
    for (const auto &param : method.parameters) {
      // Convert C++ type to C API type for parameter, skip if it can't be converted
      std::optional<std::string> capiParamTypeOpt = tryCppTypeToCapiType(param.type);
      if (!capiParamTypeOpt.has_value()) {
        warnSkippedNoConversion(method.methodName, param.type);
        return;
      }
      std::string capiParamType = capiParamTypeOpt.value();
      paramListStream << ", " << capiParamType << ' ' << param.name;
    }

    // Generate declaration
    std::string docComment =
        method.documentation.empty() ? method.methodName : method.documentation;

    os << llvm::formatv("\n/* {0} */\n", docComment);
    os << llvm::formatv(
        "MLIR_CAPI_EXPORTED {0} {1}{2}_{3}{4}({5});\n",
        capiReturnType,                  // {0}
        FunctionPrefix,                  // {1}
        dialectNameCapitalized,          // {2}
        className,                       // {3}
        toPascalCase(method.methodName), // {4}
        paramList                        // {5}
    );
  }
};

/// @brief Generator for common C implementation file elements
struct ImplementationGenerator : public Generator {
  using Generator::Generator;
  virtual ~ImplementationGenerator() = default;

  virtual void genIsAImpl() const {
    static constexpr char fmt[] = R"(
bool {0}{1}IsA_{2}_{3}(Mlir{1} inp) {{
  return llvm::isa<{3}>(unwrap(inp));
}
)";
    assert(!className.empty() && "className must be set");
    os << llvm::formatv(fmt, FunctionPrefix, kind, dialectNameCapitalized, className);
  }

  /// @brief Generate implementation for an extra method from an `extraClassDeclaration`
  virtual void genExtraMethod(const ExtraMethod &method) const override {
    // Convert return type to C API type, skip if it can't be converted
    std::optional<std::string> capiReturnTypeOpt = tryCppTypeToCapiType(method.returnType);
    if (!capiReturnTypeOpt.has_value()) {
      warnSkippedNoConversion(method.methodName, method.returnType);
      return;
    }
    std::string capiReturnType = capiReturnTypeOpt.value();

    // Build the return statement prefix and suffix
    std::string returnPrefix;
    std::string returnSuffix;
    mlir::StringRef cppReturnType = method.returnType;

    if (cppReturnType == "void") {
      // "void" type doesn't even need "return"
      returnPrefix = "";
      returnSuffix = "";
    } else {
      // Check if return needs wrapping
      if (isPrimitiveType(cppReturnType)) {
        // Primitive types don't need wrapping
        returnPrefix = "return ";
        returnSuffix = "";
      } else if (capiReturnType.starts_with("Mlir") || isAPIntType(cppReturnType)) {
        // MLIR C API types and APInt type need wrapping
        returnPrefix = "return wrap(";
        returnSuffix = ")";
      } else {
        return;
      }
    }

    // Build parameter list for C API function signature
    std::string paramList;
    llvm::raw_string_ostream paramListStream(paramList);
    paramListStream << llvm::formatv("Mlir{0} inp", kind);
    for (const auto &param : method.parameters) {
      // Convert C++ type to C API type for parameter, skip if it can't be converted
      std::optional<std::string> capiParamTypeOpt = tryCppTypeToCapiType(param.type);
      if (!capiParamTypeOpt.has_value()) {
        warnSkippedNoConversion(method.methodName, param.type);
        return;
      }
      std::string capiParamType = capiParamTypeOpt.value();
      paramListStream << ", " << capiParamType << ' ' << param.name;
    }

    // Build argument list for C++ method call
    std::string argList;
    llvm::raw_string_ostream argListStream(argList);
    for (size_t i = 0; i < method.parameters.size(); ++i) {
      if (i > 0) {
        argListStream << ", ";
      }
      const auto &param = method.parameters[i];

      // Check if parameter needs unwrapping
      mlir::StringRef cppParamType = param.type;
      if (isPrimitiveType(cppParamType)) {
        // Primitive types don't need unwrapping
        argListStream << param.name;
      } else if (isAPIntType(cppParamType)) {
        // APInt needs unwrapping
        argListStream << "unwrap(" << param.name << ')';
      } else {
        // Convert C++ type to C API type for parameter, skip if it can't be converted
        std::optional<std::string> capiParamTypeOpt = tryCppTypeToCapiType(cppParamType);
        if (capiParamTypeOpt.has_value() && capiParamTypeOpt->starts_with("Mlir")) {
          // MLIR C API types need unwrapping
          argListStream << "unwrap(" << param.name << ')';
        } else {
          warnSkippedNoConversion(method.methodName, cppParamType.str());
          return;
        }
      }
    }

    // Generate implementation
    os << '\n';
    os << llvm::formatv(
        "{0} {1}{2}_{3}{4}({5}) {{\n",
        capiReturnType,                  // {0}
        FunctionPrefix,                  // {1}
        dialectNameCapitalized,          // {2}
        className,                       // {3}
        toPascalCase(method.methodName), // {4}
        paramList                        // {5}
    );
    os << llvm::formatv(
        "  {0}llvm::cast<{1}>(unwrap(inp)).{2}({3}){4};\n",
        returnPrefix,      // {0}
        className,         // {1}
        method.methodName, // {2}
        argList,           // {3}
        returnSuffix       // {4}
    );
    os << "}\n";
  }
};

/// @brief Generator for common test implementation file elements
struct TestGenerator : public Generator {
  using Generator::Generator;
  virtual ~TestGenerator() = default;

  /// @brief Generate the test class prologue
  virtual void genTestClassPrologue() const {
    static constexpr char fmt[] = "class {0}{1}LinkTests : public CAPITest {{};\n";
    os << llvm::formatv(fmt, dialectNameCapitalized, kind);
  }

  /// @brief Generate IsA test for a class
  virtual void genIsATest() const {
    static constexpr char fmt[] = R"(
// This test ensures {0}{1}IsA_{2}_{3} links properly.
TEST_F({2}{1}LinkTests, IsA_{2}_{3}) {{
  auto test{1} = createIndex{1}();

  // This will always return false since `createIndex*` returns an MLIR builtin
  EXPECT_FALSE({0}{1}IsA_{2}_{3}(test{1}));

  {4}(test{1});
}
)";
    assert(!className.empty() && "className must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,         // {0}
        kind,                   // {1}
        dialectNameCapitalized, // {2}
        className,              // {3}
        genCleanup()            // {4}
    );
  }

  /// @brief Generate test for an extra method from extraClassDeclaration
  virtual void genExtraMethod(const ExtraMethod &method) const override {
    // Convert return type to C API type, skip if it can't be converted
    std::optional<std::string> capiReturnTypeOpt = tryCppTypeToCapiType(method.returnType);
    if (!capiReturnTypeOpt.has_value()) {
      warnSkippedNoConversion(method.methodName, method.returnType);
      return;
    }

    // Build parameter list for dummy values
    std::string dummyParams;
    llvm::raw_string_ostream dummyParamsStream(dummyParams);
    std::string paramList;
    llvm::raw_string_ostream paramListStream(paramList);

    for (const auto &param : method.parameters) {
      // Convert C++ type to C API type for parameter, skip if it can't be converted
      std::optional<std::string> capiParamTypeOpt = tryCppTypeToCapiType(param.type);
      if (!capiParamTypeOpt.has_value()) {
        warnSkippedNoConversion(method.methodName, param.type);
        return;
      }
      std::string capiParamType = capiParamTypeOpt.value();
      std::string name = param.name;

      // Generate dummy value creation for each parameter
      if (capiParamType == "bool") {
        dummyParamsStream << "    bool " << name << " = false;\n";
      } else if (capiParamType == "MlirValue") {
        dummyParamsStream << "    auto " << name << " = mlirOperationGetResult(testOp, 0);\n";
      } else if (capiParamType == "MlirType") {
        dummyParamsStream << "    auto " << name << " = createIndexType();\n";
      } else if (capiParamType == "MlirAttribute") {
        dummyParamsStream << "    auto " << name << " = createIndexAttribute();\n";
      } else if (capiParamType == "MlirStringRef") {
        dummyParamsStream << "    auto " << name << " = mlirStringRefCreateFromCString(\"\");\n";
      } else if (isIntegerType(capiParamType)) {
        dummyParamsStream << "    " << capiParamType << ' ' << name << " = 0;\n";
      } else {
        // For unknown types, create a default-initialized variable
        dummyParamsStream << "    " << capiParamType << ' ' << name << " = {};\n";
      }

      paramListStream << ", " << name;
    }

    static constexpr char fmt[] = R"(
// This test ensures {0}{2}_{3}{4} links properly.
TEST_F({2}{1}LinkTests, {0}_{3}_{4}) {{
  auto test{1} = createIndex{1}();

  if ({0}{1}IsA_{2}_{3}(test{1})) {{
{5}
    (void){0}{2}_{3}{4}(test{1}{6});
  }

  {7}(test{1});
}
)";
    assert(!className.empty() && "className must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,                  // {0}
        kind,                            // {1}
        dialectNameCapitalized,          // {2}
        className,                       // {3}
        toPascalCase(method.methodName), // {4}
        dummyParams,                     // {5}
        paramList,                       // {6}
        genCleanup()                     // {7}
    );
  }

  /// @brief Generate cleanup code for test methods
  /// @return String containing the cleanup function name or comment marker
  ///
  /// This method generates the cleanup code that should be called at the end
  /// of each test to properly release resources. The default implementation
  /// returns "//" which comments out the cleanup call (for types/attributes
  /// that don't need explicit cleanup). Derived classes should override this
  /// to return the appropriate cleanup function name (e.g., "mlirOperationDestroy"
  /// for operations).
  virtual std::string genCleanup() const {
    // The default case is to just comment out the rest of the cleanup line
    return "//";
  }
};
