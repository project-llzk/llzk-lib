//===-- LLZKUnusedDeclarationEliminationPass.cpp ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-unused-declaration-elim` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_UNUSEDDECLARATIONELIMINATIONPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
using namespace llzk::component;

#define DEBUG_TYPE "llzk-unused-declaration-elim"

namespace {

class PassImpl : public llzk::impl::UnusedDeclarationEliminationPassBase<PassImpl> {
  using Base = UnusedDeclarationEliminationPassBase<PassImpl>;
  using Base::Base;

  /// @brief Shared context between the operations in this pass (member removal, struct removal)
  /// that doesn't need to be persisted after the pass completes.
  struct PassContext {
    DenseMap<SymbolRefAttr, StructDefOp> symbolToStruct;
    DenseMap<StructDefOp, SymbolRefAttr> structToSymbol;

    const SymbolRefAttr &getSymbol(StructDefOp s) const { return structToSymbol.at(s); }
    StructDefOp getStruct(const SymbolRefAttr &sym) const { return symbolToStruct.at(sym); }

    static PassContext populate(ModuleOp modOp) {
      PassContext ctx;

      modOp.walk<WalkOrder::PreOrder>([&ctx](StructDefOp structDef) {
        auto structSymbolRes = getPathFromTopRoot(structDef);
        ensure(succeeded(structSymbolRes), "failed to lookup struct symbol");
        SymbolRefAttr structSym = *structSymbolRes;
        ctx.symbolToStruct[structSym] = structDef;
        ctx.structToSymbol[structDef] = structSym;
      });
      return ctx;
    }
  };

  void runOnOperation() override {
    PassContext ctx = PassContext::populate(getOperation());
    // First, remove unused members. This may allow more structs to be removed,
    // if their final remaining uses are as types for unused members.
    if (failed(removeUnusedMembers(ctx))) {
      signalPassFailure();
      return;
    }

    // Last, remove unused structs if configured
    if (removeStructs) {
      removeUnusedStructs(ctx);
      removeEmptyModules();
    }
  }

  /// @brief Removes unused members.
  /// A member is unused if it is never read from (only written to).
  /// @param structDef
  LogicalResult removeUnusedMembers(PassContext &ctx) {
    ModuleOp modOp = getOperation();

    SymbolTableCollection symbolTables;

    // Candidate members are tracked by resolved definitions instead of by the
    // symbol text used at each read/write site, which may be relative.
    DenseSet<MemberDefOp> members;
    for (const auto &entry : ctx.structToSymbol) {
      StructDefOp structDef = entry.first;
      bool notMain = !structDef.isMainComponent();
      structDef.walk([notMain, &members](MemberDefOp member) {
        // We don't consider public members in the Main component for removal, as these are output
        // values and removing them would result in modifying the overall circuit interface.
        if (notMain || !member.hasPublicAttr()) {
          members.insert(member);
        }
      });
    }

    // Remove all members that are read.
    WalkResult readWalk = modOp.walk([&](MemberReadOp readm) {
      FailureOr<SymbolLookupResult<MemberDefOp>> memberDef = readm.getMemberDefOp(symbolTables);
      if (failed(memberDef)) {
        return WalkResult::interrupt();
      }

      members.erase(memberDef->get());
      return WalkResult::advance();
    });
    if (readWalk.wasInterrupted()) {
      return failure();
    }

    // Remove all writes that reference the remaining members, as these writes
    // are now known to only update write-only members.
    WalkResult writeWalk = modOp.walk([&](MemberWriteOp writem) {
      FailureOr<SymbolLookupResult<MemberDefOp>> memberDef = writem.getMemberDefOp(symbolTables);
      if (failed(memberDef)) {
        return WalkResult::interrupt();
      }

      MemberDefOp writtenMember = memberDef->get();
      if (members.contains(writtenMember)) {
        // We need not check the users of a writem, since it produces no results.
        LLVM_DEBUG(
            llvm::dbgs() << "Removing write " << writem << " to write-only member " << writtenMember
                         << '\n'
        );
        writem.erase();
      }

      return WalkResult::advance();
    });
    if (writeWalk.wasInterrupted()) {
      return failure();
    }

    // Finally, erase the remaining members.
    for (MemberDefOp memberDef : members) {
      LLVM_DEBUG(llvm::dbgs() << "Removing member " << memberDef << '\n');
      memberDef->erase();
    }

    return success();
  }

  /// @brief Remove unused structs by looking for any uses of the struct's fully-qualified
  /// symbol. This catches any uses, such as member declarations of the struct's type
  /// or calls to any of the struct's methods.
  /// @param ctx
  void removeUnusedStructs(PassContext &ctx) {
    DenseMap<StructDefOp, DenseSet<StructDefOp>> uses;
    DenseMap<StructDefOp, DenseSet<StructDefOp>> usedBy;

    // initialize both maps with empty sets so we can identify unused structs
    for (auto &[structDef, _] : ctx.structToSymbol) {
      uses[structDef] = {};
      usedBy[structDef] = {};
    }

    getOperation().walk([&](Operation *op) {
      auto structParent = op->getParentOfType<StructDefOp>();
      if (structParent == nullptr) {
        return WalkResult::advance();
      }

      auto tryAddUse = [&](Type ty) {
        if (auto structTy = dyn_cast<StructType>(ty)) {
          // This name ref is required to be fully qualified
          SymbolRefAttr sym = structTy.getNameRef();
          StructDefOp refStruct = ctx.getStruct(sym);
          if (refStruct != structParent) {
            uses[structParent].insert(refStruct);
            usedBy[refStruct].insert(structParent);
          }
        }
      };

      // LLZK requires fully-qualified references to struct symbols. So, we
      // simply need to look for the struct symbol within this op's symbol uses.

      // Check operands
      for (Value operand : op->getOperands()) {
        tryAddUse(operand.getType());
      }

      // Check results
      for (Value result : op->getResults()) {
        tryAddUse(result.getType());
      }

      // Check block arguments
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments()) {
            tryAddUse(arg.getType());
          }
        }
      }

      // Check attributes
      for (const auto &namedAttr : op->getAttrs()) {
        namedAttr.getValue().walk([&tryAddUse](TypeAttr typeAttr) {
          tryAddUse(typeAttr.getValue());
        });
      }

      return WalkResult::advance();
    });

    SmallVector<StructDefOp> unusedStructs;

    auto updateUnusedStructs = [&usedBy, &unusedStructs]() {
      for (auto &[structDef, users] : usedBy) {
        if (users.empty() && !structDef.isMainComponent()) {
          unusedStructs.push_back(structDef);
        }
      }
    };

    updateUnusedStructs();

    while (!unusedStructs.empty()) {
      StructDefOp unusedStruct = unusedStructs.back();
      unusedStructs.pop_back();

      // See what structs are being used by this unused struct
      for (auto usedStruct : uses[unusedStruct]) {
        // The usedStruct is no longer used by the unusedStruct
        usedBy[usedStruct].erase(unusedStruct);
      }

      // Remove the unused struct from both maps and the IR
      usedBy.erase(unusedStruct);
      uses.erase(unusedStruct);
      unusedStruct->erase();

      // Check to see if we've created any more unused structs after we process
      // all existing known unused structs (to avoid double processing).
      if (unusedStructs.empty()) {
        updateUnusedStructs();
      }
    }
  }

  /// @brief Remove nested `module` ops with empty body.
  void removeEmptyModules() {
    SmallVector<ModuleOp> emptyModules;

    ModuleOp rootModOp = getOperation();
    rootModOp.walk<WalkOrder::PostOrder>([&](ModuleOp modOp) {
      if (modOp == rootModOp) {
        return;
      }
      Region &region = modOp.getBodyRegion();
      if (region.empty() || region.front().empty()) { // module has `SingleBlock` trait
        emptyModules.push_back(modOp);
      }
    });

    for (ModuleOp modOp : emptyModules) {
      LLVM_DEBUG(llvm::dbgs() << "Removing empty module " << modOp.getName() << '\n');
      modOp->erase();
    }
  }
};

} // namespace
