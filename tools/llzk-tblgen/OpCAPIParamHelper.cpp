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

#include <mlir/TableGen/Operator.h>

#include <llvm/Support/FormatVariadic.h>

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
