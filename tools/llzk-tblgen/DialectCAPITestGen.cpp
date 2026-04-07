//===- DialectCAPITestGen.cpp - C API test generator for dialects ---------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// DialectCAPITestGen generates unit tests for dialect-level C API functions.
// These are link-time tests that ensure dialect handle functions compile and
// link properly.
//
// Test Strategy:
// - Tests verify that mlirGetDialectHandle__<namespace>__() functions can be called
// - This ensures the dialect handle is properly exported and linkable
// - The test simply calls the function without asserting anything about the result
//
//===----------------------------------------------------------------------===//

#include "CommonCAPIGen.h"

#include <mlir/TableGen/Dialect.h>
#include <mlir/TableGen/GenInfo.h>

#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/TableGen/Record.h>
#include <llvm/TableGen/TableGenBackend.h>

#include <algorithm>

using namespace mlir;
using namespace mlir::tblgen;

static llvm::cl::OptionCategory dialectTestGenCat("Options for -gen-dialect-capi-tests");

namespace test_templates {

static constexpr char DialectTestTemplate[] = R"(
#include <mlir-c/IR.h>

class {0}DialectLinkTests : public CAPITest {{};

TEST_F({0}DialectLinkTests, get_dialect_handle_{2}) {{
  (void)mlirGetDialectHandle{1}__();
}
)";

} // namespace test_templates

/// Emit dialect C API tests
static bool emitDialectCAPITests(const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
  // Find the dialect definition
  const auto &defs = records.getAllDerivedDefinitions("Dialect");

  if (defs.empty()) {
    llvm::errs() << "Error: No Dialect definition found in the input file\n";
    return true;
  }

  if (defs.size() > 1) {
    llvm::errs() << "Warning: Multiple Dialect definitions found, using the first one\n";
  }

  Dialect dialect(defs[0]);

  // Use command-line dialect name if provided, otherwise use from definition
  std::string effectiveDialectName =
      DialectName.empty() ? dialect.getName().str() : DialectName.getValue();

  // Get the C++ namespace from the dialect definition. It's like "::llzk::boolean"
  // so replace all ':' with '_' to form the handle suffix.
  std::string cppNamespaceStr = dialect.getCppNamespace().str();
  std::replace(cppNamespaceStr.begin(), cppNamespaceStr.end(), ':', '_');

  // Generate the test file
  emitSourceFileHeader("Dialect C API Tests", os, records);
  os << llvm::formatv(
      test_templates::DialectTestTemplate,
      toPascalCase(effectiveDialectName), // {0} - Capitalized dialect name for class/file
      cppNamespaceStr,                    // {1} - Dialect handle suffix (e.g., "__llzk__boolean")
      effectiveDialectName                // {2} - Dialect name for test name
  );

  return false;
}

static mlir::GenRegistration genDialectCAPITests(
    "gen-dialect-capi-tests", "Generate dialect-level C API unit tests", &emitDialectCAPITests
);
