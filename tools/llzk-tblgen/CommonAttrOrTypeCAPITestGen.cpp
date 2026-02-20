//===- CommonAttrOrTypeCAPITestGen.cpp - Common test generation utilities -===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Implementation of CAPI test generation utilities for Attribute and Type.
//
//===----------------------------------------------------------------------===//

#include "CommonAttrOrTypeCAPITestGen.h"

#include <llvm/Support/FormatVariadic.h>

using namespace mlir;
using namespace mlir::tblgen;

/// Generate dummy parameters for Get builder
std::string generateDummyParamsForAttrOrTypeGet(const AttrOrTypeDef &def, bool isType) {
  // Use raw_string_ostream for efficient string building
  std::string paramsBuffer;
  // Reserve approximate space: ~80 chars per parameter
  paramsBuffer.reserve(80 * def.getParameters().size());
  llvm::raw_string_ostream paramsStream(paramsBuffer);

  for (const auto &param : def.getParameters()) {
    // Cache the string conversions to avoid repeated calls
    const StringRef cppType = param.getCppType();
    const StringRef pName = param.getName();

    if (isArrayRefType(cppType)) {
      paramsStream << llvm::formatv("    intptr_t {0}Count = 0;\n", pName);
      const StringRef cppElemType = extractArrayRefElementType(cppType);
      const std::string elemType = mapCppTypeToCapiType(cppElemType);
      if (isPrimitiveType(cppElemType)) {
        paramsStream << llvm::formatv("    {0} {1}Array = 0;\n", elemType, pName);
        paramsStream << llvm::formatv("    {0} *{1} = &{1}Array;\n", elemType, pName);
      } else if (isType && elemType == "MlirType") {
        paramsStream << llvm::formatv("    auto {0}Elem = createIndexType();\n", pName);
        paramsStream << llvm::formatv("    {0} *{1} = &{0}Elem;\n", elemType, pName);
      } else if (!isType && elemType == "MlirAttribute") {
        paramsStream << llvm::formatv("    auto {0}Elem = createIndexAttribute();\n", pName);
        paramsStream << llvm::formatv("    {0} *{1} = &{0}Elem;\n", elemType, pName);
      } else {
        paramsStream << llvm::formatv("    {0} {1}Elem = {{};\n", elemType, pName);
        paramsStream << llvm::formatv("    {0} *{1} = &{1}Elem;\n", elemType, pName);
      }
    } else {
      const std::string capiType = mapCppTypeToCapiType(cppType);
      if (isPrimitiveType(cppType)) {
        paramsStream << llvm::formatv("    {0} {1} = 0;\n", capiType, pName);
      } else if (isType && capiType == "MlirType") {
        paramsStream << llvm::formatv("    auto {0} = createIndexType();\n", pName);
      } else if (!isType && capiType == "MlirAttribute") {
        paramsStream << llvm::formatv("    auto {0} = createIndexAttribute();\n", pName);
      } else {
        // For enum or other types, use static_cast to initialize with 0
        paramsStream << llvm::formatv("    {0} {1} = static_cast<{0}>(0);\n", capiType, pName);
      }
    }
  }

  return paramsBuffer;
}

/// Generate parameter list for Get builder call
std::string generateParamListForAttrOrTypeGet(const AttrOrTypeDef &def) {
  // Use raw_string_ostream for efficient string building
  std::string paramsBuffer;
  // Reserve approximate space: ~30 chars per parameter
  paramsBuffer.reserve(30 * def.getParameters().size());
  llvm::raw_string_ostream paramsStream(paramsBuffer);

  for (const auto &param : def.getParameters()) {
    // Cache the string conversion to avoid repeated calls
    const StringRef pName = param.getName();
    if (isArrayRefType(param.getCppType())) {
      paramsStream << llvm::formatv(", {0}Count, {0}", pName);
    } else {
      paramsStream << llvm::formatv(", {0}", pName);
    }
  }

  return paramsBuffer;
}
