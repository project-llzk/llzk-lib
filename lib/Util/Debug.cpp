//===-- Debug.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Util/Debug.h"
#include "llzk/Util/StreamHelper.h"

using namespace mlir;

namespace llzk {
namespace debug {

void dumpSymbolTableWalk(Operation *symbolTableOp) {
  std::string output; // buffer to avoid multi-threaded mess
  llvm::raw_string_ostream oss(output);
  oss << "Dumping symbol walk (self = [" << symbolTableOp << "]): \n";
  auto walkFn = [&](Operation *op, bool allUsesVisible) {
    oss << "  found op [" << op << "] " << op->getName() << " named "
        << op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()) << '\n';
  };
  SymbolTable::walkSymbolTables(symbolTableOp, /*allSymUsesVisible=*/true, walkFn);
  llvm::dbgs() << output;
}

void dumpSymbolTable(llvm::raw_ostream &stream, SymbolTable &symTab, unsigned indent) {
  indent *= 2;
  stream.indent(indent) << "Dumping SymbolTable [" << &symTab << "]: \n";
  auto *rawSymbolTablePtr = reinterpret_cast<char *>(&symTab);
  auto *privateMemberPtr =
      reinterpret_cast<llvm::DenseMap<Attribute, Operation *> *>(rawSymbolTablePtr + 8);
  indent += 2;
  for (llvm::detail::DenseMapPair<Attribute, Operation *> &p : *privateMemberPtr) {
    Operation *op = p.second;
    stream.indent(indent) << p.first << " -> [" << op << "] " << op->getName() << '\n';
  }
}

void dumpSymbolTable(SymbolTable &symTab) {
  // Buffer to a string then print to avoid multi-threaded mess
  llvm::dbgs() << buildStringViaCallback([&symTab](llvm::raw_ostream &stream) {
    dumpSymbolTable(stream, symTab);
  });
}

void dumpSymbolTables(llvm::raw_ostream &stream, SymbolTableCollection &tables) {
  stream << "Dumping SymbolTableCollection [" << &tables << "]: \n";
  auto *rawObjectPtr = reinterpret_cast<char *>(&tables);
  auto *privateMemberPtr =
      reinterpret_cast<llvm::DenseMap<Operation *, std::unique_ptr<SymbolTable>> *>(
          rawObjectPtr + 0
      );
  for (llvm::detail::DenseMapPair<Operation *, std::unique_ptr<SymbolTable>> &p :
       *privateMemberPtr) {
    stream << "  [" << p.first << "] " << p.first->getName() << " -> " << '\n';
    dumpSymbolTable(stream, *p.second.get(), 2);
  }
}

void dumpSymbolTables(SymbolTableCollection &tables) {
  // Buffer to a string then print to avoid multi-threaded mess
  llvm::dbgs() << buildStringViaCallback([&tables](llvm::raw_ostream &stream) {
    dumpSymbolTables(stream, tables);
  });
}

void dumpToFile(Operation *op, llvm::StringRef filename) {
  std::error_code err;
  llvm::raw_fd_stream stream(filename, err);
  if (!err) {
    auto options = OpPrintingFlags().assumeVerified().useLocalScope();
    op->print(stream, options);
    stream << '\n';
  }
}

} // namespace debug
} // namespace llzk
