//===- CommonAttrOrTypeCAPIGen.h ------------------------------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Common utilities shared between Attr and Type CAPI generators
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/TableGen/AttrOrTypeDef.h>

#include "CommonCAPIGen.h"

/// @brief Generator for attribute/type C header files
///
/// This class extends HeaderGenerator to provide attribute and type-specific
/// header generation capabilities, including parameter getters and builders.
struct AttrOrTypeHeaderGenerator : public HeaderGenerator {
  using HeaderGenerator::HeaderGenerator;
  virtual ~AttrOrTypeHeaderGenerator() = default;

  /// @brief Set the parameter name for code generation
  /// @param name The parameter name from the TableGen definition
  void setParamName(mlir::StringRef name) {
    this->paramName = name;
    this->paramNameCapitalized = toPascalCase(name);
  }

  /// @brief Generate regular getter for non-ArrayRef type parameter
  virtual void genParameterGetterDecl(mlir::StringRef cppType) const {
    static constexpr char fmt[] = R"(
/* Get '{5}' parameter from a {6}::{3} {1}. */
MLIR_CAPI_EXPORTED {7} {0}{2}_{3}Get{4}(Mlir{1});
)";
    assert(dialect && "Dialect must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,               // {0}
        kind,                         // {1}
        dialectNameCapitalized,       // {2}
        className,                    // {3}
        paramNameCapitalized,         // {4}
        paramName,                    // {5}
        dialect->getCppNamespace(),   // {6}
        mapCppTypeToCapiType(cppType) // {7}
    );
  }

  /// @brief Generate accessor function for ArrayRef parameter elements
  virtual void genArrayRefParameterGetterDecls(mlir::StringRef cppType) const {
    static constexpr char fmt[] = R"(
/* Get count of '{5}' parameter from a {6}::{3} {1}. */
MLIR_CAPI_EXPORTED intptr_t {0}{2}_{3}Get{4}Count(Mlir{1});

/* Get element at index from '{5}' parameter from a {6}::{3} {1}. */
MLIR_CAPI_EXPORTED {7} {0}{2}_{3}Get{4}At(Mlir{1}, intptr_t pos);
)";
    assert(dialect && "Dialect must be set");
    assert(!paramName.empty() && "paramName must be set");
    mlir::StringRef cppElemType = extractArrayRefElementType(cppType);
    os << llvm::formatv(
        fmt,
        FunctionPrefix,                   // {0}
        kind,                             // {1}
        dialectNameCapitalized,           // {2}
        className,                        // {3}
        paramNameCapitalized,             // {4}
        paramName,                        // {5}
        dialect->getCppNamespace(),       // {6}
        mapCppTypeToCapiType(cppElemType) // {7}
    );
  }

  /// @brief Generate default Get builder declaration
  virtual void genDefaultGetBuilderDecl(const mlir::tblgen::AttrOrTypeDef &def) const {
    static constexpr char fmt[] = R"(
/* Create a {5}::{3} {1} with the given parameters. */
MLIR_CAPI_EXPORTED Mlir{1} {0}{2}_{3}Get(MlirContext ctx{4});
)";
    assert(dialect && "Dialect must be set");

    // Use raw_string_ostream for efficient string building of parameter list
    std::string paramListBuffer;
    llvm::raw_string_ostream paramListStream(paramListBuffer);
    for (const auto &param : def.getParameters()) {
      mlir::StringRef cppType = param.getCppType();
      if (isArrayRefType(cppType)) {
        // For ArrayRef parameters, use intptr_t for count and pointer to element type
        mlir::StringRef cppElemType = extractArrayRefElementType(cppType);
        paramListStream << ", intptr_t " << param.getName() << "Count, "
                        << mapCppTypeToCapiType(cppElemType) << " *" << param.getName();
      } else {
        paramListStream << ", " << mapCppTypeToCapiType(cppType) << ' ' << param.getName();
      }
    }

    os << llvm::formatv(
        fmt,
        FunctionPrefix,            // {0}
        kind,                      // {1}
        dialectNameCapitalized,    // {2}
        className,                 // {3}
        paramListBuffer,           // {4}
        dialect->getCppNamespace() // {5}
    );
  }

  void genCompleteRecord(const mlir::tblgen::AttrOrTypeDef def) {
    const mlir::tblgen::Dialect &defDialect = def.getDialect();

    // Generate for the selected dialect only
    if (defDialect.getName() != DialectName) {
      return;
    }

    this->setDialectAndClassName(&defDialect, def.getCppClassName());

    // Generate IsA check
    if (GenIsA) {
      this->genIsADecl();
    }

    // Generate default Get builder if not skipped
    if (GenTypeOrAttrGet && !def.skipDefaultBuilders()) {
      this->genDefaultGetBuilderDecl(def);
    }

    // Generate parameter getters
    if (GenTypeOrAttrParamGetters) {
      for (const auto &param : def.getParameters()) {
        this->setParamName(param.getName());
        mlir::StringRef cppType = param.getCppType();
        if (isArrayRefType(cppType)) {
          this->genArrayRefParameterGetterDecls(cppType);
        } else {
          this->genParameterGetterDecl(cppType);
        }
      }
    }

    // Generate extra class method declarations
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

/// @brief Generator for attribute/type C implementation files
///
/// This class extends ImplementationGenerator to provide attribute and type-specific
/// implementation generation capabilities, including parameter getters and builders.
struct AttrOrTypeImplementationGenerator : public ImplementationGenerator {
  using ImplementationGenerator::ImplementationGenerator;
  virtual ~AttrOrTypeImplementationGenerator() = default;

  /// @brief Set the parameter name for code generation
  /// @param name The parameter name from the TableGen definition
  void setParamName(mlir::StringRef name) {
    this->paramName = name;
    this->paramNameCapitalized = toPascalCase(name);
  }

  virtual void genPrologue() const {
    os << R"(
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;
using namespace llvm;
)";
  }

  virtual void genArrayRefParameterImpls(mlir::StringRef cppType) const {
    static constexpr char fmt[] = R"(
intptr_t {0}{2}_{3}Get{4}Count(Mlir{1} inp) {{
  return static_cast<intptr_t>(llvm::cast<{3}>(unwrap(inp)).get{4}().size());
}

{5} {0}{2}_{3}Get{4}At(Mlir{1} inp, intptr_t pos) {{
  return {6}(llvm::cast<{3}>(unwrap(inp)).get{4}()[pos]);
}
 )";
    assert(!className.empty() && "className must be set");
    assert(!paramName.empty() && "paramName must be set");
    mlir::StringRef cppElemType = extractArrayRefElementType(cppType);
    os << llvm::formatv(
        fmt,
        FunctionPrefix,                            // {0}
        kind,                                      // {1}
        dialectNameCapitalized,                    // {2}
        className,                                 // {3}
        paramNameCapitalized,                      // {4}
        mapCppTypeToCapiType(cppElemType),         // {5}
        isPrimitiveType(cppElemType) ? "" : "wrap" // {6}
    );
  }

  virtual void genParameterGetterImpl(mlir::StringRef cppType) const {
    static constexpr char fmt[] = R"(
{5} {0}{2}_{3}Get{4}(Mlir{1} inp) {{
  return {6}(llvm::cast<{3}>(unwrap(inp)).get{4}());
}
 )";
    assert(!className.empty() && "className must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,                        // {0}
        kind,                                  // {1}
        dialectNameCapitalized,                // {2}
        className,                             // {3}
        paramNameCapitalized,                  // {4}
        mapCppTypeToCapiType(cppType),         // {5}
        isPrimitiveType(cppType) ? "" : "wrap" // {6}
    );
  }

  /// @brief Generate default Get builder implementation
  virtual void genDefaultGetBuilderImpl(const mlir::tblgen::AttrOrTypeDef &def) const {
    static constexpr char fmt[] = R"(
Mlir{1} {0}{2}_{3}Get(MlirContext ctx{4}) {{
  {6}
  return wrap({3}::get(unwrap(ctx){5}));
}
 )";
    assert(!className.empty() && "className must be set");

    // Use raw_string_ostream for efficient string building
    std::string paramListBuffer;
    std::string argListBuffer;
    std::string prefixBuffer;
    llvm::raw_string_ostream paramListStream(paramListBuffer);
    llvm::raw_string_ostream argListStream(argListBuffer);
    llvm::raw_string_ostream prefixStream(prefixBuffer);

    for (const auto &param : def.getParameters()) {
      mlir::StringRef pName = param.getName();
      mlir::StringRef cppType = param.getCppType();
      if (isArrayRefType(cppType)) {
        // For ArrayRef parameters, convert from pointer + count to ArrayRef
        mlir::StringRef cppElemType = extractArrayRefElementType(cppType);
        std::string capiElemType = mapCppTypeToCapiType(cppElemType);
        paramListStream << ", intptr_t " << pName << "Count, " << capiElemType << " *" << pName;

        // In the call, we need to convert back to ArrayRef. Check if elements need unwrapping.
        if (isPrimitiveType(cppElemType)) {
          argListStream << ", ::llvm::ArrayRef<" << cppElemType << ">(" << pName << ", " << pName
                        << "Count)";
        } else {
          prefixStream << "SmallVector<Attribute> storage;";
          argListStream << ", llvm::map_to_vector(unwrapList(" << pName << "Count, " << pName
                        << ", storage), [](auto a) {return llvm::cast<RecordAttr>(a);})";
        }
      } else {
        std::string capiType = mapCppTypeToCapiType(cppType);
        paramListStream << ", " << capiType << ' ' << pName;

        // Add unwrapping if needed
        argListStream << ", ";
        if (isPrimitiveType(cppType)) {
          argListStream << pName;
        } else if (capiType == "MlirAttribute" || capiType == "MlirType") {
          // Needs additional cast to the specific attribute/type class
          argListStream << "::llvm::cast<" << cppType << ">(unwrap(" << pName << "))";
        } else {
          // Any other cases, just use an "unwrap" function
          argListStream << "unwrap(" << pName << ")";
        }
      }
    }

    os << llvm::formatv(
        fmt,
        FunctionPrefix,         // {0}
        kind,                   // {1}
        dialectNameCapitalized, // {2}
        className,              // {3}
        paramListBuffer,        // {4}
        argListBuffer,          // {5}
        prefixBuffer            // {6}
    );
  }

  void genCompleteRecord(const mlir::tblgen::AttrOrTypeDef def) {
    const mlir::tblgen::Dialect &defDialect = def.getDialect();

    // Generate for the selected dialect only
    if (defDialect.getName() != DialectName) {
      return;
    }

    this->setDialectAndClassName(&defDialect, def.getCppClassName());

    // Generate IsA check implementation
    if (GenIsA) {
      this->genIsAImpl();
    }

    // Generate default Get builder implementation if not skipped
    if (GenTypeOrAttrGet && !def.skipDefaultBuilders()) {
      this->genDefaultGetBuilderImpl(def);
    }

    // Generate parameter getter implementations
    if (GenTypeOrAttrParamGetters) {
      for (const auto &param : def.getParameters()) {
        this->setParamName(param.getName());
        mlir::StringRef cppType = param.getCppType();
        if (isArrayRefType(cppType)) {
          this->genArrayRefParameterImpls(cppType);
        } else {
          this->genParameterGetterImpl(cppType);
        }
      }
    }

    // Generate extra class method implementations
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
