//===-- llzk-tblgen.cpp - LLZK tblgen tool ----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the main entry point for the llzk-tblgen tool.
/// The tool extends mlir-tblgen with additional C API generators that are
/// registered by the other source files in this directory (OpCAPIGen.cpp,
/// AttrCAPIGen.cpp, TypeCAPIGen.cpp, etc.). These generators provide more
/// comprehensive C API coverage than the default MLIR tablegen tool.
///
//===----------------------------------------------------------------------===//

#include "CAPIGenRegistration.h"

#include <mlir/Tools/mlir-tblgen/MlirTblgenMain.h>

#include <llvm/Support/raw_ostream.h>

#include <cstdlib>
#include <exception>

int main(int argc, char **argv) noexcept {
  try {
    llzk::registerCAPIOptions();
    llzk::registerAttrCAPIGenerators();
    llzk::registerAttrCAPITestGenerator();
    llzk::registerTypeCAPIGenerators();
    llzk::registerTypeCAPITestGenerator();
    llzk::registerEnumCAPIGenerators();
    llzk::registerEnumCAPITestGenerator();
    llzk::registerOpCAPIGenerators();
    llzk::registerOpCAPITestGenerator();
    llzk::registerDialectCAPITestGenerator();
    return mlir::MlirTblgenMain(argc, argv);
  } catch (const std::exception &ex) {
    llvm::errs() << "llzk-tblgen: unhandled exception: " << ex.what() << '\n';
  } catch (...) {
    llvm::errs() << "llzk-tblgen: unhandled non-standard exception\n";
  }
  return EXIT_FAILURE;
}
