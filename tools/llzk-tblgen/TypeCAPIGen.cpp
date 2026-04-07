//===- TypeCAPIGen.cpp - C API generator for types ------------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// TypeCAPIGen uses the description of types to generate C API for the types.
//
//===----------------------------------------------------------------------===//

#include "CommonAttrOrTypeCAPIGen.h"
#include "CommonCAPIGen.h"

#include <mlir/TableGen/AttrOrTypeDef.h>
#include <mlir/TableGen/GenInfo.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/TableGen/Record.h>
#include <llvm/TableGen/TableGenBackend.h>

using namespace mlir;
using namespace mlir::tblgen;

/// Emit C API header for types
static bool emitTypeCAPIHeader(const llvm::RecordKeeper &records, raw_ostream &os) {
  emitSourceFileHeader("Type C API Declarations", os, records);

  AttrOrTypeHeaderGenerator generator("Type", os);
  generator.genPrologue();

  for (const auto *def : records.getAllDerivedDefinitions("TypeDef")) {
    const AttrOrTypeDef type(def);
    generator.genCompleteRecord(type);
  }

  generator.genEpilogue();
  return false;
}

/// Emit C API implementation for types
static bool emitTypeCAPIImpl(const llvm::RecordKeeper &records, raw_ostream &os) {
  emitSourceFileHeader("Type C API Implementations", os, records);

  AttrOrTypeImplementationGenerator generator("Type", os);
  generator.genPrologue();

  for (const auto *def : records.getAllDerivedDefinitions("TypeDef")) {
    const AttrOrTypeDef type(def);
    generator.genCompleteRecord(type);
  }

  return false;
}

static mlir::GenRegistration genTypeCAPIHeader(
    "gen-type-capi-header", "Generate type C API header declarations", &emitTypeCAPIHeader
);

static mlir::GenRegistration
    genTypeCAPIImpl("gen-type-capi-impl", "Generate type C API implementations", &emitTypeCAPIImpl);
