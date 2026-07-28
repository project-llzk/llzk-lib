//===-- llzk-lsp-server.cpp - LLZK LSP server -------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file runs the MLIR LSP server configured with LLZK dialects.
//
//===----------------------------------------------------------------------===//

#include "tools/config.h"

#include "llzk/Dialect/InitDialects.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-lsp-server/MlirLspServerMain.h>

#include <llvm/Support/PrettyStackTrace.h>

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  llvm::setBugReportMsg(
      "PLEASE submit a bug report to " BUG_REPORT_URL
      " and include the crash backtrace and inciting LLZK files.\n"
  );
  llzk::registerAllDialects(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
