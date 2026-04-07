//===- AttrCAPIGen.cpp - C API generator for attributes -------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// AttrCAPIGen uses the description of attributes to generate C API for the attrs.
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

/// Emit C API header for attributes
static bool emitAttrCAPIHeader(const llvm::RecordKeeper &records, raw_ostream &os) {
  emitSourceFileHeader("Attr C API Declarations", os, records);

  AttrOrTypeHeaderGenerator generator("Attribute", os);
  generator.genPrologue();

  for (const auto *def : records.getAllDerivedDefinitions("AttrDef")) {
    const AttrOrTypeDef attr(def);
    generator.genCompleteRecord(attr);
  }

  generator.genEpilogue();
  return false;
}

/// Emit C API implementation for attributes
static bool emitAttrCAPIImpl(const llvm::RecordKeeper &records, raw_ostream &os) {
  emitSourceFileHeader("Attr C API Implementations", os, records);

  AttrOrTypeImplementationGenerator generator("Attribute", os);
  generator.genPrologue();

  for (const auto *def : records.getAllDerivedDefinitions("AttrDef")) {
    const AttrOrTypeDef attr(def);
    generator.genCompleteRecord(attr);
  }

  return false;
}

static mlir::GenRegistration genAttrCAPIHeader(
    "gen-attr-capi-header", "Generate attribute C API header declarations", &emitAttrCAPIHeader
);

static mlir::GenRegistration genAttrCAPIImpl(
    "gen-attr-capi-impl", "Generate attribute C API implementations", &emitAttrCAPIImpl
);
