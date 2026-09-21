//===- OpCAPIParamHelper.cpp ----------------------------------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "OpCAPIParamHelper.h"

#include "CommonCAPIGen.h"

#include <mlir/TableGen/Interfaces.h>
#include <mlir/TableGen/Operator.h>

#include <llvm/ADT/StringSet.h>
#include <llvm/Support/FormatVariadic.h>

llvm::SmallVector<ExtraMethod> getCAPIExposedOpMethods(const mlir::tblgen::Operator &op) {
  llvm::SmallVector<ExtraMethod> methods = parseExtraMethods(op.getExtraClassDeclaration());
  llvm::StringSet<> methodNames;
  for (const ExtraMethod &method : methods) {
    methodNames.insert(method.methodName);
  }

  for (const mlir::tblgen::Trait &traitDef : op.getTraits()) {
    const auto *trait = llvm::dyn_cast<mlir::tblgen::InterfaceTrait>(&traitDef);
    if (!trait || !trait->shouldDeclareMethods()) {
      continue;
    }

    llvm::StringSet<> requestedMethods;
    for (llvm::StringRef methodName : trait->getAlwaysDeclaredMethods()) {
      requestedMethods.insert(methodName);
    }
    mlir::tblgen::Interface interface = trait->getInterface();
    for (const mlir::tblgen::InterfaceMethod &interfaceMethod : interface.getMethods()) {
      // `alwaysOverriddenMethods` is the explicit C API opt-in. Keep the
      // remaining declaration checks in sync with MLIR's
      // OpEmitter::genOpInterfaceMethods().
      if (!requestedMethods.contains(interfaceMethod.getName()) || interfaceMethod.isStatic() ||
          interfaceMethod.getBody()) {
        continue;
      }

      if (!methodNames.insert(interfaceMethod.getName()).second) {
        warnSkipped(interfaceMethod.getName(), "C API does not support method overloading");
        continue;
      }

      ExtraMethod method;
      method.returnType = interfaceMethod.getReturnType().str();
      method.methodName = interfaceMethod.getName().str();
      for (const mlir::tblgen::InterfaceMethod::Argument &argument :
           interfaceMethod.getArguments()) {
        method.parameters.emplace_back(argument.type.str(), argument.name.str());
      }
      methods.push_back(std::move(method));
    }
  }

  return methods;
}

std::string GenStringFromOpPieces::gen(const mlir::tblgen::Operator &op) {
  std::string params;
  llvm::raw_string_ostream oss(params);
  genHeader(oss);
  if (!op.allResultTypesKnown()) {
    // If result types are not inferred, call handler for each result
    for (auto [i, result] : llvm::enumerate(op.getResults())) {
      llvm::StringRef name = result.name;
      genResult(oss, result, name.empty() ? llvm::formatv("result{0}", i).str() : name.str());
    }
  } else {
    // Otherewise, call inferred result handler
    genResultInferred(oss);
  }
  for (const mlir::tblgen::NamedTypeConstraint &operand : op.getOperands()) {
    genOperand(oss, operand);
  }
  {
    auto attrs = op.getAttributes();
    if (!attrs.empty()) {
      genAttributesPrefix(oss, op);
      for (const mlir::tblgen::NamedAttribute &namedAttr : attrs) {
        genAttribute(oss, namedAttr);
      }
      genAttributesSuffix(oss, op);
    }
  }
  {
    auto regions = op.getRegions();
    if (!regions.empty()) {
      genRegionsPrefix(oss, op);
      for (const mlir::tblgen::NamedRegion &region : regions) {
        genRegion(oss, region);
      }
      genRegionsSuffix(oss, op);
    }
  }
  return params;
}
