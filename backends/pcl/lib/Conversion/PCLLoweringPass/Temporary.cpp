//===-- Temporary.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "Temporary.h"

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;
using namespace pcl::lowering;
using namespace llzk;

using Prefix = const char *;
using Names = DenseMap<mlir::Operation *, mlir::StringAttr>;

#define CASE(T, prefix) Case<T>([](auto) { return prefix; })

namespace {

class TemporariesBuilder {

  Names &names;
  /// Maps the prefix to the number of times it's been used.
  llvm::StringMap<unsigned> prefixes;
  /// Names already in use (i.e. struct members)
  StringSet<> usedNames;
  Builder builder;
  ModuleOp root;

  /// Gets the prefix associated with the operation type.
  Prefix getPrefix(Operation *op) {
    return TypeSwitch<Operation *, Prefix>(op)
        .CASE(llzk::NonDetOp, "__nondet")
        .CASE(arith::SelectOp, "__arith_select")
        .Default([](auto) { return nullptr; });
  }

  /// Returns a fresh name using the given prefix.
  StringAttr next(StringRef prefix) {
    assert(!prefix.empty() && "empty prefixes not allowed");
    auto &count = prefixes[prefix];
    StringRef name;
    SmallString<64> sto;

    do {
      sto.clear();
      name = (prefix + Twine(count++)).toStringRef(sto);
    } while (usedNames.contains(name));

    return builder.getStringAttr(name);
  }

  /// If the op is one that defines a name in the environment,
  /// add that name to the set of used names to avoid collisions.
  void fillUsedNames(Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<component::MemberDefOp>([this](auto member) {
      usedNames.insert(member.getSymName());
    }).Default([](auto) {});
  }

public:
  TemporariesBuilder(ModuleOp R, Names &N) : names(N), builder(R), root(R) {}

  void fill() {
    root->walk([this](Operation *op) { fillUsedNames(op); });
    root->walk([this](Operation *op) {
      if (Prefix prefix = getPrefix(op)) {
        names.insert({op, next(prefix)});
      }
    });
  }
};
} // namespace

Temporaries::Temporaries(ModuleOp root) {
  TemporariesBuilder builder(root, names);
  builder.fill();
}
