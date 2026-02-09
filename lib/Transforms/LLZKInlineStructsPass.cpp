//===-- LLZKInlineStructsPass.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-inline-structs` pass.
///
/// This pass should be run after `llzk-flatten` to ensure structs do not have template parameters
/// (this restriction may be removed in the future).
///
/// This pass also assumes that all subcomponents that are created by calling a struct "@compute"
/// function are ultimately written to exactly one member within the current struct.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/GraphUtil.h"
#include "llzk/Analysis/SymbolUseGraph.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKConversionUtils.h"
#include "llzk/Transforms/LLZKInlineStructsPass.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/InliningUtils.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>

#include <concepts>

// Include the generated base pass class definitions.
namespace llzk {
// the *DECL* macro is required when a pass has options to declare the option struct
#define GEN_PASS_DECL_INLINESTRUCTSPASS
#define GEN_PASS_DEF_INLINESTRUCTSPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
using namespace llzk::component;
using namespace llzk::function;

#define DEBUG_TYPE "llzk-inline-structs"

namespace {

using DestMemberWithSrcStructType = MemberDefOp;
using DestCloneOfSrcStructMember = MemberDefOp;
/// Mapping of the name of each member in the inlining source struct to the new cloned version of
/// the source member in the destination struct. Uses `std::map` for consistent ordering between
/// multiple compilations of the same LLZK IR input.
using SrcStructMemberToCloneInDest = std::map<StringRef, DestCloneOfSrcStructMember>;
/// Mapping of `MemberDefOp` in the inlining destination struct to each `MemberDefOp` from the
/// inlining source struct to the new cloned version of the source member in the destination struct.
using DestToSrcToClonedSrcInDest =
    DenseMap<DestMemberWithSrcStructType, SrcStructMemberToCloneInDest>;

/// Get the "self" value for the given `FuncDefOp`, which must be either a "compute" or
/// "constrain" function.
static inline Value getSelfValue(FuncDefOp f) {
  if (f.nameIsCompute()) {
    return f.getSelfValueFromCompute();
  } else if (f.nameIsConstrain()) {
    return f.getSelfValueFromConstrain();
  } else {
    llvm_unreachable("expected \"@compute\" or \"@constrain\" function");
  }
}

/// Get the `MemberDefOp` that defines the member referenced by the given `MemberRefOpInterface`
/// with an assertion failure if it is not found.
static inline MemberDefOp getDef(SymbolTableCollection &tables, MemberRefOpInterface fRef) {
  auto r = fRef.getMemberDefOp(tables);
  assert(succeeded(r));
  return r->get();
}

/// Find the `MemberWriteOp` that writes the given subcomponent struct `Value`. Produce an error
/// (using the given callback) if there is not exactly once such `MemberWriteOp`.
static FailureOr<MemberWriteOp>
findOpThatStoresSubcmp(Value writtenValue, function_ref<InFlightDiagnostic()> emitError) {
  MemberWriteOp foundWrite = nullptr;
  for (Operation *user : writtenValue.getUsers()) {
    if (MemberWriteOp writeOp = llvm::dyn_cast<MemberWriteOp>(user)) {
      // Find the write op that stores the created value
      if (writeOp.getVal() == writtenValue) {
        if (foundWrite) {
          // Note: There is no reason for a subcomponent to be stored to more than one member.
          auto diag = emitError().append("result should not be written to more than one member.");
          diag.attachNote(foundWrite.getLoc()).append("written here");
          diag.attachNote(writeOp.getLoc()).append("written here");
          return diag;
        } else {
          foundWrite = writeOp;
        }
      }
    }
  }
  if (!foundWrite) {
    // Note: There is no reason to construct a subcomponent and not store it to a member.
    return emitError().append("result should be written to a member.");
  }
  return foundWrite;
}

/// If there exists a member ref chain in `destToSrcToClone` for the given `MemberReadOp` (as
/// described in `combineReadChain()` or `combineNewThenReadChain()`), replace it with a
/// new `MemberReadOp` that directly reads from the given cloned member and delete it.
static bool combineHelper(
    MemberReadOp readOp, SymbolTableCollection &tables,
    const DestToSrcToClonedSrcInDest &destToSrcToClone, MemberRefOpInterface destMemberRefOp
) {
  LLVM_DEBUG({
    llvm::dbgs() << "[combineHelper] " << readOp << " => " << destMemberRefOp << '\n';
  });

  auto srcToClone = destToSrcToClone.find(getDef(tables, destMemberRefOp));
  if (srcToClone == destToSrcToClone.end()) {
    return false;
  }
  SrcStructMemberToCloneInDest oldToNewMembers = srcToClone->second;
  auto resNewMember = oldToNewMembers.find(readOp.getMemberName());
  if (resNewMember == oldToNewMembers.end()) {
    return false;
  }

  // Replace this MemberReadOp with a new one that targets the cloned member.
  OpBuilder builder(readOp);
  MemberReadOp newRead = builder.create<MemberReadOp>(
      readOp.getLoc(), readOp.getType(), destMemberRefOp.getComponent(),
      resNewMember->second.getNameAttr()
  );
  readOp.replaceAllUsesWith(newRead.getOperation());
  readOp.erase(); // delete the original MemberReadOp
  return true;
}

/// If the base component Value of the given MemberReadOp is the result of reading from a member in
/// `destToSrcToClone` and the member referenced by this MemberReadOp has a cloned member mapping in
/// `destToSrcToClone`, replace this read with a new MemberReadOp referencing the cloned member.
///
/// Example:
///   Given the mapping (@fa, !struct.type<@Component10A>) -> @f -> \@"fa:!s<@Component10A>+f"
///   And the input:
///     %0 = struct.readm %arg0[@fa] : !struct.type<@Main>, !struct.type<@Component10A>
///     %3 = struct.readm %0[@f] : !struct.type<@Component10A>, !felt.type
///   Replace the final read with:
///     %3 = struct.readm %arg0[@"fa:!s<@Component10A>+f"] : !struct.type<@Main>, !felt.type
///
/// Return true if replaced, false if not.
static bool combineReadChain(
    MemberReadOp readOp, SymbolTableCollection &tables,
    const DestToSrcToClonedSrcInDest &destToSrcToClone
) {
  LLVM_DEBUG({ llvm::dbgs() << "[combineReadChain] " << readOp << '\n'; });

  MemberReadOp readThatDefinesBaseComponent =
      llvm::dyn_cast_if_present<MemberReadOp>(readOp.getComponent().getDefiningOp());
  if (!readThatDefinesBaseComponent) {
    return false;
  }
  return combineHelper(readOp, tables, destToSrcToClone, readThatDefinesBaseComponent);
}

/// If the base component Value of the given MemberReadOp is the result of `struct.new` which is
/// written to a member in `destToSrcToClone` and the member referenced by the given MemberReadOp
/// has a cloned member mapping in `destToSrcToClone`, replace the given MemberReadOp with a new
/// MemberReadOp referencing the cloned member.
///
/// Example:
///   Given the mapping (@fa, !struct.type<@Component10A>) -> @f -> \@"fa:!s<@Component10A>+f"
///   And the input:
///     %0 = struct.new : !struct.type<@Main>
///     %2 = struct.new : !struct.type<@Component10A>
///     struct.writem %0[@fa] = %2 : !struct.type<@Main>, !struct.type<@Component10A>
///     %4 = struct.readm %2[@f] : !struct.type<@Component10A>, !felt.type
///   Replace the final read with:
///     %4 = struct.readm %0[@"fa:!s<@Component10A>+f"] : !struct.type<@Main>, !felt.type
///
/// Return true if replaced, false if not.
static LogicalResult combineNewThenReadChain(
    MemberReadOp readOp, SymbolTableCollection &tables,
    const DestToSrcToClonedSrcInDest &destToSrcToClone
) {
  LLVM_DEBUG({ llvm::dbgs() << "[combineNewThenReadChain] " << readOp << '\n'; });

  CreateStructOp createThatDefinesBaseComponent =
      llvm::dyn_cast_if_present<CreateStructOp>(readOp.getComponent().getDefiningOp());
  if (!createThatDefinesBaseComponent) {
    return success(); // No error. The pattern simply doesn't match.
  }
  FailureOr<MemberWriteOp> foundWrite =
      findOpThatStoresSubcmp(createThatDefinesBaseComponent, [&createThatDefinesBaseComponent]() {
    return createThatDefinesBaseComponent.emitOpError();
  });
  if (failed(foundWrite)) {
    return failure(); // error already printed within findOpThatStoresSubcmp()
  }
  return success(combineHelper(readOp, tables, destToSrcToClone, foundWrite.value()));
}

static inline MemberReadOp getMemberReadThatDefinesSelfValuePassedToConstrain(CallOp callOp) {
  Value selfArgFromCall = callOp.getSelfValueFromConstrain();
  return llvm::dyn_cast_if_present<MemberReadOp>(selfArgFromCall.getDefiningOp());
}

/// Cache various ops from the caller struct that should be erased but only after all callees are
/// fully handled (to avoid "still has uses" errors).
struct PendingErasure {
  SmallPtrSet<Operation *, 8> memberReadOps;
  SmallPtrSet<Operation *, 8> memberWriteOps;
  SmallVector<CreateStructOp> newStructOps;
  SmallVector<DestMemberWithSrcStructType> memberDefs;
};

/// Handles the bulk of inlining one struct into another.
class StructInliner {
  SymbolTableCollection &tables;
  PendingErasure &toDelete;
  /// The struct that will be inlined (and maybe removed).
  StructDefOp srcStruct;
  /// The struct whose body will be augmented with the inlined content.
  StructDefOp destStruct;

  inline MemberDefOp getDef(MemberRefOpInterface fRef) const { return ::getDef(tables, fRef); }

  // Update member read/write ops that target the "self" value of the FuncDefOp plus some key in
  // `oldToNewMemberDef` to instead target the new base Value provided to the constructor plus the
  // mapped Value from `oldToNewMemberDef`.
  // Example:
  //  old:  %1 = struct.readm %0[@f1] : <@Component1A>, !felt.type
  //  new:  %1 = struct.readm %self[@"f2:!s<@Component1A>+f1"] : <@Component1B>, !felt.type
  class MemberRefRewriter final : public OpInterfaceRewritePattern<MemberRefOpInterface> {
    /// This is initially the `originalFunc` parameter from the constructor but after the clone is
    /// created within `cloneWithMemberRefUpdate()`, it is reassigned to the cloned function.
    FuncDefOp funcRef;
    /// The "self" value in the cloned function.
    Value oldBaseVal;
    /// The new base value for updated member references.
    Value newBaseVal;
    const SrcStructMemberToCloneInDest &oldToNewMembers;

  public:
    MemberRefRewriter(
        FuncDefOp originalFunc, Value newRefBase,
        const SrcStructMemberToCloneInDest &oldToNewMemberDef
    )
        : OpInterfaceRewritePattern(originalFunc.getContext()), funcRef(originalFunc),
          oldBaseVal(nullptr), newBaseVal(newRefBase), oldToNewMembers(oldToNewMemberDef) {}

    LogicalResult match(MemberRefOpInterface op) const final {
      assert(oldBaseVal); // ensure it's used via `cloneWithMemberRefUpdate()` only
      // Check if the MemberRef accesses a member of "self" within the `oldToNewMembers` map.
      // Per `cloneWithMemberRefUpdate()`, `oldBaseVal` is the "self" value of `funcRef` so
      // check for a match there and then check that the referenced member name is in the map.
      return success(
          op.getComponent() == oldBaseVal && oldToNewMembers.contains(op.getMemberName())
      );
    }

    void rewrite(MemberRefOpInterface op, PatternRewriter &rewriter) const final {
      rewriter.modifyOpInPlace(op, [this, &op]() {
        DestCloneOfSrcStructMember newF = oldToNewMembers.at(op.getMemberName());
        op.setMemberName(newF.getSymName());
        op.getComponentMutable().set(this->newBaseVal);
      });
    }

    /// Create a clone of the `FuncDefOp` and update member references according to the
    /// `SrcStructMemberToCloneInDest` map (both are within the given `MemberRefRewriter`).
    static FuncDefOp cloneWithMemberRefUpdate(std::unique_ptr<MemberRefRewriter> thisPat) {
      IRMapping mapper;
      FuncDefOp srcFuncClone = thisPat->funcRef.clone(mapper);
      // Update some data in the `MemberRefRewriter` instance before moving it.
      thisPat->funcRef = srcFuncClone;
      thisPat->oldBaseVal = getSelfValue(srcFuncClone);
      // Run the rewriter to replace read/write ops
      MLIRContext *ctx = thisPat->getContext();
      RewritePatternSet patterns(ctx, std::move(thisPat));
      walkAndApplyPatterns(srcFuncClone, std::move(patterns));

      return srcFuncClone;
    }
  };

  /// Common implementation for inlining both "constrain" and "compute" functions.
  class ImplBase {
  protected:
    const StructInliner &data;
    const DestToSrcToClonedSrcInDest &destToSrcToClone;

    /// Get the "self" struct parameter from the CallOp and determine which member that struct was
    /// stored in within the caller.
    virtual MemberRefOpInterface getSelfRefMember(CallOp callOp) = 0;
    virtual void processCloneBeforeInlining(FuncDefOp func) {}
    virtual ~ImplBase() = default;

  public:
    ImplBase(const StructInliner &inliner, const DestToSrcToClonedSrcInDest &destToSrcToCloneRef)
        : data(inliner), destToSrcToClone(destToSrcToCloneRef) {}

    LogicalResult doInlining(FuncDefOp srcFunc, FuncDefOp destFunc) {
      LLVM_DEBUG({
        llvm::dbgs() << "[doInlining] SOURCE FUNCTION:\n";
        srcFunc.dump();
        llvm::dbgs() << "[doInlining] DESTINATION FUNCTION:\n";
        destFunc.dump();
      });

      InlinerInterface inliner(destFunc.getContext());

      /// Replaces CallOp that target `srcFunc` with an inlined version of `srcFunc`.
      auto callHandler = [this, &inliner, &srcFunc](CallOp callOp) {
        // Ensure the CallOp targets `srcFunc`
        auto callOpTarget = callOp.getCalleeTarget(this->data.tables);
        assert(succeeded(callOpTarget));
        if (callOpTarget->get() != srcFunc) {
          return WalkResult::advance();
        }

        // Get the "self" struct parameter from the CallOp and determine which member that struct
        // was stored in within the caller (i.e. `destFunc`).
        MemberRefOpInterface selfMemberRefOp = this->getSelfRefMember(callOp);
        if (!selfMemberRefOp) {
          // Note: error message was already printed within `getSelfRefMember()`
          return WalkResult::interrupt(); // use interrupt to signal failure
        }

        // Create a clone of the source function (must do the whole function not just the body
        // region because `inlineCall()` expects the Region to have a parent op) and update member
        // references to the old struct members to instead use the new struct members.
        FuncDefOp srcFuncClone = MemberRefRewriter::cloneWithMemberRefUpdate(
            std::make_unique<MemberRefRewriter>(
                srcFunc, selfMemberRefOp.getComponent(),
                this->destToSrcToClone.at(this->data.getDef(selfMemberRefOp))
            )
        );
        this->processCloneBeforeInlining(srcFuncClone);

        // Inline the cloned function in place of `callOp`
        LogicalResult inlineCallRes =
            inlineCall(inliner, callOp, srcFuncClone, &srcFuncClone.getBody(), false);
        if (failed(inlineCallRes)) {
          callOp.emitError().append("Failed to inline ", srcFunc.getFullyQualifiedName()).report();
          return WalkResult::interrupt(); // use interrupt to signal failure
        }
        srcFuncClone.erase();      // delete what's left after transferring the body elsewhere
        callOp.erase();            // delete the original CallOp
        return WalkResult::skip(); // Must skip because the CallOp was erased.
      };

      auto memberWriteHandler = [this](MemberWriteOp writeOp) {
        // Check if the member ref op should be deleted in the end
        if (this->destToSrcToClone.contains(this->data.getDef(writeOp))) {
          this->data.toDelete.memberWriteOps.insert(writeOp);
        }
        return WalkResult::advance();
      };

      /// Combine chained MemberReadOp according to replacements in `destToSrcToClone`.
      /// See `combineReadChain()`
      auto memberReadHandler = [this](MemberReadOp readOp) {
        // Check if the member ref op should be deleted in the end
        if (this->destToSrcToClone.contains(this->data.getDef(readOp))) {
          this->data.toDelete.memberReadOps.insert(readOp);
        }
        // If the MemberReadOp was replaced/erased, must skip.
        return combineReadChain(readOp, this->data.tables, destToSrcToClone)
                   ? WalkResult::skip()
                   : WalkResult::advance();
      };

      WalkResult walkRes = destFunc.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
        return TypeSwitch<Operation *, WalkResult>(op)
            .Case<CallOp>(callHandler)
            .Case<MemberWriteOp>(memberWriteHandler)
            .Case<MemberReadOp>(memberReadHandler)
            .Default([](Operation *) { return WalkResult::advance(); });
      });

      return failure(walkRes.wasInterrupted());
    }
  };

  class ConstrainImpl : public ImplBase {
    using ImplBase::ImplBase;

    MemberRefOpInterface getSelfRefMember(CallOp callOp) override {
      LLVM_DEBUG({ llvm::dbgs() << "[ConstrainImpl::getSelfRefMember] " << callOp << '\n'; });

      // The typical pattern is to read a struct instance from a member and then call "constrain()"
      // on it. Get the Value passed as the "self" struct to the CallOp and determine which member
      // it was read from in the current struct (i.e., `destStruct`).
      MemberRefOpInterface selfMemberRef =
          getMemberReadThatDefinesSelfValuePassedToConstrain(callOp);
      if (selfMemberRef &&
          selfMemberRef.getComponent().getType() == this->data.destStruct.getType()) {
        return selfMemberRef;
      }
      callOp.emitError()
          .append(
              "expected \"self\" parameter to \"@", FUNC_NAME_CONSTRAIN,
              "\" to be passed a value read from a member in the current stuct."
          )
          .report();
      return nullptr;
    }
  };

  class ComputeImpl : public ImplBase {
    using ImplBase::ImplBase;

    MemberRefOpInterface getSelfRefMember(CallOp callOp) override {
      LLVM_DEBUG({ llvm::dbgs() << "[ComputeImpl::getSelfRefMember] " << callOp << '\n'; });

      // The typical pattern is to write the return value of "compute()" to a member in
      // the current struct (i.e., `destStruct`).
      // It doesn't really make sense (although there is no semantic restriction against it) to just
      // pass the "compute()" result into another function and never write it to a member since that
      // leaves no way for the "constrain()" function to call "constrain()" on that result struct.
      FailureOr<MemberWriteOp> foundWrite =
          findOpThatStoresSubcmp(callOp.getSelfValueFromCompute(), [&callOp]() {
        return callOp.emitOpError().append("\"@", FUNC_NAME_COMPUTE, "\" ");
      });
      return static_cast<MemberRefOpInterface>(foundWrite.value_or(nullptr));
    }

    void processCloneBeforeInlining(FuncDefOp func) override {
      // Within the compute function, find `CreateStructOp` with `srcStruct` type and mark them
      // for later deletion. The deletion must occur later because these values may still have
      // uses until ALL callees of a function have been inlined.
      func.getBody().walk([this](CreateStructOp newStructOp) {
        if (newStructOp.getType() == this->data.srcStruct.getType()) {
          this->data.toDelete.newStructOps.push_back(newStructOp);
        }
      });
    }
  };

  // Find any member(s) in `destStruct` whose type matches `srcStruct` (allowing any parameters, if
  // applicable). For each such member, clone all members from `srcStruct` into `destStruct` and
  // cache the mapping of `destStruct` to `srcStruct` to cloned members in the return value.
  DestToSrcToClonedSrcInDest cloneMembers() {
    DestToSrcToClonedSrcInDest destToSrcToClone;

    SymbolTable &destStructSymTable = tables.getSymbolTable(destStruct);
    StructType srcStructType = srcStruct.getType();
    for (MemberDefOp destMember : destStruct.getMemberDefs()) {
      if (StructType destMemberType = llvm::dyn_cast<StructType>(destMember.getType())) {
        UnificationMap unifications;
        if (!structTypesUnify(srcStructType, destMemberType, {}, &unifications)) {
          continue;
        }
        assert(unifications.empty()); // `makePlan()` reports failure earlier
        // Mark the original `destMember` for deletion
        toDelete.memberDefs.push_back(destMember);
        // Clone each member from 'srcStruct' into 'destStruct'. Add an entry to `destToSrcToClone`
        // even if there are no members in `srcStruct` so its presence can be used as a marker.
        SrcStructMemberToCloneInDest &srcToClone = destToSrcToClone[destMember];
        std::vector<MemberDefOp> srcMembers = srcStruct.getMemberDefs();
        if (srcMembers.empty()) {
          continue;
        }
        OpBuilder builder(destMember);
        std::string newNameBase =
            destMember.getName().str() + ':' + BuildShortTypeString::from(destMemberType);
        for (MemberDefOp srcMember : srcMembers) {
          DestCloneOfSrcStructMember newF = llvm::cast<MemberDefOp>(builder.clone(*srcMember));
          newF.setName(builder.getStringAttr(newNameBase + '+' + newF.getName()));
          srcToClone[srcMember.getSymNameAttr()] = newF;
          // Also update the cached SymbolTable
          destStructSymTable.insert(newF);
        }
      }
    }
    return destToSrcToClone;
  }

  /// Inline the "constrain" function from `srcStruct` into `destStruct`.
  inline LogicalResult inlineConstrainCall(const DestToSrcToClonedSrcInDest &destToSrcToClone) {
    return ConstrainImpl(*this, destToSrcToClone)
        .doInlining(srcStruct.getConstrainFuncOp(), destStruct.getConstrainFuncOp());
  }

  /// Inline the "compute" function from `srcStruct` into `destStruct`.
  inline LogicalResult inlineComputeCall(const DestToSrcToClonedSrcInDest &destToSrcToClone) {
    return ComputeImpl(*this, destToSrcToClone)
        .doInlining(srcStruct.getComputeFuncOp(), destStruct.getComputeFuncOp());
  }

public:
  StructInliner(
      SymbolTableCollection &tbls, PendingErasure &opsToDelete, StructDefOp from, StructDefOp into
  )
      : tables(tbls), toDelete(opsToDelete), srcStruct(from), destStruct(into) {}

  FailureOr<DestToSrcToClonedSrcInDest> doInline() {
    LLVM_DEBUG(
        llvm::dbgs() << "[StructInliner] merge " << srcStruct.getSymNameAttr() << " into "
                     << destStruct.getSymNameAttr() << '\n'
    );

    DestToSrcToClonedSrcInDest destToSrcToClone = cloneMembers();
    if (failed(inlineConstrainCall(destToSrcToClone)) ||
        failed(inlineComputeCall(destToSrcToClone))) {
      return failure(); // error already printed within doInlining()
    }
    return destToSrcToClone;
  }
};

template <typename T>
concept HasContainsOp = requires(const T &t, Operation *p) {
  { t.contains(p) } -> std::convertible_to<bool>;
};

/// Handles remaining uses of an Operation's result Value before erasing the Operation.
template <typename... PendingDeletionSets>
  requires(HasContainsOp<PendingDeletionSets> && ...)
class DanglingUseHandler {
  SymbolTableCollection &tables;
  const DestToSrcToClonedSrcInDest &destToSrcToClone;
  std::tuple<const PendingDeletionSets &...> otherRefsToBeDeleted;

public:
  DanglingUseHandler(
      SymbolTableCollection &symTables, const DestToSrcToClonedSrcInDest &destToSrcToCloneRef,
      const PendingDeletionSets &...otherRefsPendingDeletion
  )
      : tables(symTables), destToSrcToClone(destToSrcToCloneRef),
        otherRefsToBeDeleted(otherRefsPendingDeletion...) {}

  /// Call before erasing an Operation to ensure that any remaining uses of the Operation's result
  /// are removed if possible, else report an error (the subsequent call to erase() would fail
  /// anyway if the result Value still has uses). Handles the following cases:
  /// - If the op is used as argument to a function with a body, convert to take members separately.
  /// - If the op is used as argument to a function without a body, report an error.
  LogicalResult handle(Operation *op) const {
    if (op->use_empty()) {
      return success(); // safe to erase
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[DanglingUseHandler::handle] op: " << *op << '\n';
      llvm::dbgs() << "[DanglingUseHandler::handle]   in function: "
                   << op->getParentOfType<FuncDefOp>() << '\n';
    });
    for (OpOperand &use : llvm::make_early_inc_range(op->getUses())) {
      if (CallOp c = llvm::dyn_cast<CallOp>(use.getOwner())) {
        if (failed(handleUseInCallOp(use, c, op))) {
          return failure();
        }
      } else {
        Operation *user = use.getOwner();
        // Report an error for any user other than some member ref that will be deleted anyway.
        if (!opWillBeDeleted(user)) {
          return op->emitOpError()
              .append(
                  "with use in '", user->getName().getStringRef(),
                  "' is not (currently) supported by this pass."
              )
              .attachNote(user->getLoc())
              .append("used by this operation");
        }
      }
    }
    // Ensure that all users of the 'op' were deleted above, or will be per 'otherRefsToBeDeleted'.
    if (!op->use_empty()) {
      for (Operation *user : op->getUsers()) {
        if (!opWillBeDeleted(user)) {
          llvm::errs() << "Op has remaining use(s) that could not be removed: " << *op << '\n';
          llvm_unreachable("Expected all uses to be removed");
        }
      }
    }
    return success();
  }

private:
  inline LogicalResult handleUseInCallOp(OpOperand &use, CallOp inCall, Operation *origin) const {
    LLVM_DEBUG(
        llvm::dbgs() << "[DanglingUseHandler::handleUseInCallOp]   use in call: " << inCall << '\n'
    );
    unsigned argIdx = use.getOperandNumber() - inCall.getArgOperands().getBeginOperandIndex();
    LLVM_DEBUG(
        llvm::dbgs() << "[DanglingUseHandler::handleUseInCallOp]     at index: " << argIdx << '\n'
    );

    auto tgtFuncRes = inCall.getCalleeTarget(tables);
    if (failed(tgtFuncRes)) {
      return origin
          ->emitOpError("as argument to an unknown function is not supported by this pass.")
          .attachNote(inCall.getLoc())
          .append("used by this call");
    }
    FuncDefOp tgtFunc = tgtFuncRes->get();
    LLVM_DEBUG(
        llvm::dbgs() << "[DanglingUseHandler::handleUseInCallOp]   call target: " << tgtFunc << '\n'
    );
    if (tgtFunc.isExternal()) {
      // Those without a body (i.e. external implementation) present a problem because LLZK does
      // not define a memory layout for the external implementation to interpret the struct.
      return origin
          ->emitOpError("as argument to a no-body free function is not supported by this pass.")
          .attachNote(inCall.getLoc())
          .append("used by this call");
    }

    MemberRefOpInterface paramFromMember =
        TypeSwitch<Operation *, MemberRefOpInterface>(origin)
            .template Case<MemberReadOp>([](auto p) { return p; })
            .template Case<CreateStructOp>([](auto p) {
      return findOpThatStoresSubcmp(p, [&p]() { return p.emitOpError(); }).value_or(nullptr);
    }).Default([](Operation *p) {
      llvm::errs() << "Encountered unexpected op: "
                   << (p ? p->getName().getStringRef() : "<<null>>") << '\n';
      llvm_unreachable("Unexpected op kind");
      return nullptr;
    });
    LLVM_DEBUG({
      llvm::dbgs() << "[DanglingUseHandler::handleUseInCallOp]   member ref op for param: "
                   << (paramFromMember ? debug::toStringOne(paramFromMember) : "<<null>>") << '\n';
    });
    if (!paramFromMember) {
      return failure(); // error already printed within findOpThatStoresSubcmp()
    }
    const SrcStructMemberToCloneInDest &newMembers =
        destToSrcToClone.at(getDef(tables, paramFromMember));
    LLVM_DEBUG({
      llvm::dbgs() << "[DanglingUseHandler::handleUseInCallOp]   members to split: "
                   << debug::toStringList(newMembers) << '\n';
    });

    // Convert the FuncDefOp side first (to use the easier builder for the new CallOp).
    splitFunctionParam(tgtFunc, argIdx, newMembers);
    LLVM_DEBUG({
      llvm::dbgs() << "[DanglingUseHandler::handleUseInCallOp]   UPDATED call target: " << tgtFunc
                   << '\n';
      llvm::dbgs() << "[DanglingUseHandler::handleUseInCallOp]   UPDATED call target type: "
                   << tgtFunc.getFunctionType() << '\n';
    });

    // Convert the CallOp side. Add a MemberReadOp for each value from the struct and pass them
    // individually in place of the struct parameter.
    OpBuilder builder(inCall);
    SmallVector<Value> splitArgs;
    // Before the CallOp, insert a read from every new member. These Values will replace the
    // original argument in the CallOp.
    Value originalBaseVal = paramFromMember.getComponent();
    for (auto [origName, newMemberRef] : newMembers) {
      splitArgs.push_back(builder.create<MemberReadOp>(
          inCall.getLoc(), newMemberRef.getType(), originalBaseVal, newMemberRef.getNameAttr()
      ));
    }
    // Generate the new argument list from the original but replace 'argIdx'
    SmallVector<Value> newOpArgs(inCall.getArgOperands());
    newOpArgs.insert(
        newOpArgs.erase(newOpArgs.begin() + argIdx), splitArgs.begin(), splitArgs.end()
    );
    // Create the new CallOp, replace uses of the old with the new, delete the old
    inCall.replaceAllUsesWith(builder.create<CallOp>(
        inCall.getLoc(), tgtFunc, CallOp::toVectorOfValueRange(inCall.getMapOperands()),
        inCall.getNumDimsPerMapAttr(), newOpArgs
    ));
    inCall.erase();
    LLVM_DEBUG({
      llvm::dbgs() << "[DanglingUseHandler::handleUseInCallOp]   UPDATED function: "
                   << origin->getParentOfType<FuncDefOp>() << '\n';
    });
    return success();
  }

  /// Helper function to determine if an Operation is contained in 'otherRefsToBeDeleted'
  inline bool opWillBeDeleted(Operation *otherOp) const {
    return std::apply([&](const auto &...sets) {
      return ((sets.contains(otherOp)) || ...);
    }, otherRefsToBeDeleted);
  }

  /// Replace the function parameter at `paramIdx` with multiple parameters according to the types
  /// of the values in the given `SrcStructMemberToCloneInDest` map. Within the body, replace reads
  /// from the original parameter with direct uses of the new block argument Values per the member
  /// name keys in the map.
  static void splitFunctionParam(
      FuncDefOp func, unsigned paramIdx, const SrcStructMemberToCloneInDest &nameToNewMember
  ) {
    class Impl : public FunctionTypeConverter {
      unsigned inputIdx;
      const SrcStructMemberToCloneInDest &newMembers;

    public:
      Impl(unsigned paramIdx, const SrcStructMemberToCloneInDest &nameToNewMember)
          : inputIdx(paramIdx), newMembers(nameToNewMember) {}

    protected:
      SmallVector<Type> convertInputs(ArrayRef<Type> origTypes) override {
        SmallVector<Type> newTypes(origTypes);
        auto it = newTypes.erase(newTypes.begin() + inputIdx);
        for (auto [_, newMember] : newMembers) {
          newTypes.insert(it, newMember.getType());
          ++it;
        }
        return newTypes;
      }
      SmallVector<Type> convertResults(ArrayRef<Type> origTypes) override {
        return SmallVector<Type>(origTypes);
      }
      ArrayAttr convertInputAttrs(ArrayAttr origAttrs, SmallVector<Type>) override {
        if (origAttrs) {
          // Replicate the value at `origAttrs[inputIdx]` to have `newMembers.size()`
          SmallVector<Attribute> newAttrs(origAttrs.getValue());
          newAttrs.insert(newAttrs.begin() + inputIdx, newMembers.size() - 1, origAttrs[inputIdx]);
          return ArrayAttr::get(origAttrs.getContext(), newAttrs);
        }
        return nullptr;
      }
      ArrayAttr convertResultAttrs(ArrayAttr origAttrs, SmallVector<Type>) override {
        return origAttrs;
      }

      void processBlockArgs(Block &entryBlock, RewriterBase &rewriter) override {
        Value oldStructRef = entryBlock.getArgument(inputIdx);

        // Insert new Block arguments, one per member, following the original one. Keep a map
        // of member name to the associated block argument for replacing MemberReadOp.
        llvm::StringMap<BlockArgument> memberNameToNewArg;
        Location loc = oldStructRef.getLoc();
        unsigned idx = inputIdx;
        for (auto [memberName, newMember] : newMembers) {
          // note: pre-increment so the original to be erased is still at `inputIdx`
          BlockArgument newArg = entryBlock.insertArgument(++idx, newMember.getType(), loc);
          memberNameToNewArg[memberName] = newArg;
        }

        // Find all member reads from the original Block argument and replace uses of those
        // reads with the appropriate new Block argument.
        for (OpOperand &oldBlockArgUse : llvm::make_early_inc_range(oldStructRef.getUses())) {
          if (MemberReadOp readOp = llvm::dyn_cast<MemberReadOp>(oldBlockArgUse.getOwner())) {
            if (readOp.getComponent() == oldStructRef) {
              BlockArgument newArg = memberNameToNewArg.at(readOp.getMemberName());
              rewriter.replaceAllUsesWith(readOp, newArg);
              rewriter.eraseOp(readOp);
              continue;
            }
          }
          // Currently, there's no other way in which a StructType parameter can be used.
          llvm::errs() << "Unexpected use of " << oldBlockArgUse.get() << " in "
                       << *oldBlockArgUse.getOwner() << '\n';
          llvm_unreachable("Not yet implemented");
        }

        // Delete the original Block argument
        entryBlock.eraseArgument(inputIdx);
      }
    };
    IRRewriter rewriter(func.getContext());
    Impl(paramIdx, nameToNewMember).convert(func, rewriter);
  }
};

static LogicalResult finalizeStruct(
    SymbolTableCollection &tables, StructDefOp caller, PendingErasure &&toDelete,
    DestToSrcToClonedSrcInDest &&destToSrcToClone
) {
  LLVM_DEBUG({
    llvm::dbgs() << "[finalizeStruct] dumping 'caller' struct before compressing chains:\n";
    caller.print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
    llvm::dbgs() << '\n';
  });

  // Compress chains of reads that result after inlining multiple callees.
  caller.getConstrainFuncOp().walk([&tables, &destToSrcToClone](MemberReadOp readOp) {
    combineReadChain(readOp, tables, destToSrcToClone);
  });
  FuncDefOp computeFn = caller.getComputeFuncOp();
  Value computeSelfVal = computeFn.getSelfValueFromCompute();
  auto res = computeFn.walk([&tables, &destToSrcToClone, &computeSelfVal](MemberReadOp readOp) {
    combineReadChain(readOp, tables, destToSrcToClone);
    // Reads targeting the "self" value from "compute()" are not eligible for the compression
    // provided in `combineNewThenReadChain()` and will actually cause an error within.
    if (readOp.getComponent() == computeSelfVal) {
      return WalkResult::advance();
    }
    LogicalResult innerRes = combineNewThenReadChain(readOp, tables, destToSrcToClone);
    return failed(innerRes) ? WalkResult::interrupt() : WalkResult::advance();
  });
  if (res.wasInterrupted()) {
    return failure(); // error already printed within combineNewThenReadChain()
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[finalizeStruct] dumping 'caller' struct before deleting ops:\n";
    caller.print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
    llvm::dbgs() << '\n';
    llvm::dbgs() << "[finalizeStruct] ops marked for deletion:\n";
    for (Operation *op : toDelete.memberReadOps) {
      llvm::dbgs().indent(2) << *op << '\n';
    }
    for (Operation *op : toDelete.memberWriteOps) {
      llvm::dbgs().indent(2) << *op << '\n';
    }
    for (CreateStructOp op : toDelete.newStructOps) {
      llvm::dbgs().indent(2) << op << '\n';
    }
    for (DestMemberWithSrcStructType op : toDelete.memberDefs) {
      llvm::dbgs().indent(2) << op << '\n';
    }
  });

  // Handle remaining uses of CreateStructOp before deleting anything because this process
  // needs to be able to find the MemberWriteOp instances that store the result of these ops.
  DanglingUseHandler<SmallPtrSet<Operation *, 8>, SmallPtrSet<Operation *, 8>> useHandler(
      tables, destToSrcToClone, toDelete.memberWriteOps, toDelete.memberReadOps
  );
  for (CreateStructOp op : toDelete.newStructOps) {
    if (failed(useHandler.handle(op))) {
      return failure(); // error already printed within handle()
    }
  }
  // Next, to avoid "still has uses" errors, must erase MemberWriteOp first, then MemberReadOp,
  // before erasing the CreateStructOp or MemberDefOp.
  for (Operation *op : toDelete.memberWriteOps) {
    if (failed(useHandler.handle(op))) {
      return failure(); // error already printed within handle()
    }
    op->erase();
  }
  for (Operation *op : toDelete.memberReadOps) {
    if (failed(useHandler.handle(op))) {
      return failure(); // error already printed within handle()
    }
    op->erase();
  }
  for (CreateStructOp op : toDelete.newStructOps) {
    op.erase();
  }
  // Finally, erase MemberDefOp via SymbolTable so table itself is updated too.
  SymbolTable &callerSymTab = tables.getSymbolTable(caller);
  for (DestMemberWithSrcStructType op : toDelete.memberDefs) {
    assert(op.getParentOp() == caller); // using correct SymbolTable
    callerSymTab.erase(op);
  }

  return success();
}

} // namespace

LogicalResult performInlining(SymbolTableCollection &tables, InliningPlan &plan) {
  for (auto &[caller, callees] : plan) {
    // Cache operations that should be deleted but must wait until all callees are processed
    // to ensure that all uses of the values defined by these operations are replaced.
    PendingErasure toDelete;
    // Cache old-to-new member mappings across all callees inlined for the current struct.
    DestToSrcToClonedSrcInDest aggregateReplacements;
    // Inline callees/subcomponents of the current struct
    for (StructDefOp toInline : callees) {
      FailureOr<DestToSrcToClonedSrcInDest> res =
          StructInliner(tables, toDelete, toInline, caller).doInline();
      if (failed(res)) {
        return failure();
      }
      // Add current member replacements to the aggregate
      for (auto &[k, v] : res.value()) {
        assert(!aggregateReplacements.contains(k) && "duplicate not possible");
        aggregateReplacements[k] = std::move(v);
      }
    }
    // Complete steps to finalize/cleanup the caller
    LogicalResult finalizeResult =
        finalizeStruct(tables, caller, std::move(toDelete), std::move(aggregateReplacements));
    if (failed(finalizeResult)) {
      return failure();
    }
  }
  return success();
}

namespace {

class InlineStructsPass : public llzk::impl::InlineStructsPassBase<InlineStructsPass> {
  static uint64_t complexity(FuncDefOp f) {
    uint64_t complexity = 0;
    f.getBody().walk([&complexity](Operation *op) {
      if (llvm::isa<felt::MulFeltOp>(op)) {
        ++complexity;
      } else if (auto ee = llvm::dyn_cast<constrain::EmitEqualityOp>(op)) {
        complexity += computeEmitEqCardinality(ee.getLhs().getType());
      } else if (auto ec = llvm::dyn_cast<constrain::EmitContainmentOp>(op)) {
        // TODO: increment based on dimension sizes in the operands
        // Pending update to implementation/semantics of EmitContainmentOp.
        ++complexity;
      }
    });
    return complexity;
  }

  static FailureOr<FuncDefOp>
  getIfStructConstrain(const SymbolUseGraphNode *node, SymbolTableCollection &tables) {
    auto lookupRes = node->lookupSymbol(tables, false);
    assert(succeeded(lookupRes) && "graph contains node with invalid path");
    if (FuncDefOp f = llvm::dyn_cast<FuncDefOp>(lookupRes->get())) {
      if (f.isStructConstrain()) {
        return f;
      }
    }
    return failure();
  }

  /// Return the parent StructDefOp for the given Function (which is known to be a struct
  /// "constrain" function so it must have a StructDefOp parent).
  static inline StructDefOp getParentStruct(FuncDefOp func) {
    assert(func.isStructConstrain()); // pre-condition
    FailureOr<StructDefOp> currentNodeParentStruct = getParentOfType<StructDefOp>(func);
    assert(succeeded(currentNodeParentStruct)); // follows from ODS definition
    return currentNodeParentStruct.value();
  }

  /// Return 'true' iff the `maxComplexity` option is set and the given value exceeds it.
  inline bool exceedsMaxComplexity(uint64_t check) {
    return maxComplexity > 0 && check > maxComplexity;
  }

  /// Check for additional conditions that make inlining impossible (at least in the current
  /// implementation).
  static inline bool canInline(FuncDefOp currentFunc, FuncDefOp successorFunc) {
    // Find CallOp for `successorFunc` within `currentFunc` and check the condition used by
    // `ConstrainImpl::getSelfRefMember()`.
    //
    // Implementation Note: There is a possibility that the "self" value is not from a member read.
    // It could be a parameter to the current/destination function or a global read. Inlining a
    // struct stored to a global would probably require splitting up the global into multiple, one
    // for each member in the successor/source struct. That may not be a good idea. The parameter
    // case could be handled but it will not have a mapping in `destToSrcToClone` in
    // `getSelfRefMember()` and new members will still need to be added. They can be prefixed with
    // parameter index since there is no current member name to use as the unique prefix. Handling
    // that would require refactoring the inlining process a bit.
    WalkResult res = currentFunc.walk([](CallOp c) {
      return getMemberReadThatDefinesSelfValuePassedToConstrain(c)
                 ? WalkResult::interrupt() // use interrupt to indicate success
                 : WalkResult::advance();
    });
    LLVM_DEBUG({
      llvm::dbgs() << "[canInline] " << successorFunc.getFullyQualifiedName() << " into "
                   << currentFunc.getFullyQualifiedName() << "? " << res.wasInterrupted() << '\n';
    });
    return res.wasInterrupted();
  }

  /// Perform a bottom-up traversal of the "constrain" function nodes in the SymbolUseGraph to
  /// determine which ones can be inlined to their callers while respecting the `maxComplexity`
  /// option. Using a bottom-up traversal may give a better result than top-down because the latter
  /// could result in a chain of structs being inlined differently from different use sites.
  inline FailureOr<InliningPlan>
  makePlan(const SymbolUseGraph &useGraph, SymbolTableCollection &tables) {
    LLVM_DEBUG({
      llvm::dbgs() << "Running InlineStructsPass with max complexity ";
      if (maxComplexity == 0) {
        llvm::dbgs() << "unlimited";
      } else {
        llvm::dbgs() << maxComplexity;
      }
      llvm::dbgs() << '\n';
    });
    InliningPlan retVal;
    DenseMap<const SymbolUseGraphNode *, uint64_t> complexityMemo;

    // NOTE: The assumption that the use graph has no cycles allows `complexityMemo` to only
    // store the result for relevant nodes and assume nodes without a mapped value are `0`. This
    // must be true of the "compute"/"constrain" function uses and member defs because circuits
    // must be acyclic. This is likely true to for the symbol use graph is general but if a
    // counterexample is ever found, the algorithm below must be re-evaluated.
    assert(!hasCycle(&useGraph));

    // Traverse "constrain" function nodes to compute their complexity and an inlining plan. Use
    // post-order traversal so the complexity of all successor nodes is computed before computing
    // the current node's complexity.
    for (const SymbolUseGraphNode *currentNode : llvm::post_order(&useGraph)) {
      LLVM_DEBUG(llvm::dbgs() << "\ncurrentNode = " << currentNode->toString());
      if (!currentNode->isRealNode()) {
        continue;
      }
      if (currentNode->isStructParam()) {
        // Try to get the location of the StructDefOp to report an error.
        Operation *lookupFrom = currentNode->getSymbolPathRoot().getOperation();
        SymbolRefAttr prefix = getPrefixAsSymbolRefAttr(currentNode->getSymbolPath());
        auto res = lookupSymbolIn<StructDefOp>(tables, prefix, lookupFrom, lookupFrom, false);
        // If that lookup didn't work for some reason, report at the path root location.
        Operation *reportLoc = succeeded(res) ? res->get() : lookupFrom;
        return reportLoc->emitError("Cannot inline structs with parameters.");
      }
      FailureOr<FuncDefOp> currentFuncOpt = getIfStructConstrain(currentNode, tables);
      if (failed(currentFuncOpt)) {
        continue;
      }
      FuncDefOp currentFunc = currentFuncOpt.value();
      uint64_t currentComplexity = complexity(currentFunc);
      // If the current complexity is already too high, store it and continue.
      if (exceedsMaxComplexity(currentComplexity)) {
        complexityMemo[currentNode] = currentComplexity;
        continue;
      }
      // Otherwise, make a plan that adds successor "constrain" functions unless the
      // complexity becomes too high by adding that successor.
      SmallVector<StructDefOp> successorsToMerge;
      for (const SymbolUseGraphNode *successor : currentNode->successorIter()) {
        LLVM_DEBUG(llvm::dbgs().indent(2) << "successor: " << successor->toString() << '\n');
        // Note: all "constrain" function nodes will have a value, and all other nodes will not.
        auto memoResult = complexityMemo.find(successor);
        if (memoResult == complexityMemo.end()) {
          continue; // inner loop
        }
        uint64_t sComplexity = memoResult->second;
        assert(
            sComplexity <= (std::numeric_limits<uint64_t>::max() - currentComplexity) &&
            "addition will overflow"
        );
        uint64_t potentialComplexity = currentComplexity + sComplexity;
        if (!exceedsMaxComplexity(potentialComplexity)) {
          currentComplexity = potentialComplexity;
          FailureOr<FuncDefOp> successorFuncOpt = getIfStructConstrain(successor, tables);
          assert(succeeded(successorFuncOpt)); // follows from the Note above
          FuncDefOp successorFunc = successorFuncOpt.value();
          if (canInline(currentFunc, successorFunc)) {
            successorsToMerge.push_back(getParentStruct(successorFunc));
          }
        }
      }
      complexityMemo[currentNode] = currentComplexity;
      if (!successorsToMerge.empty()) {
        retVal.emplace_back(getParentStruct(currentFunc), std::move(successorsToMerge));
      }
    }
    LLVM_DEBUG({
      llvm::dbgs() << "-----------------------------------------------------------------\n";
      llvm::dbgs() << "InlineStructsPass plan:\n";
      for (auto &[caller, callees] : retVal) {
        llvm::dbgs().indent(2) << "inlining the following into \"" << caller.getSymName() << "\"\n";
        for (StructDefOp c : callees) {
          llvm::dbgs().indent(4) << "\"" << c.getSymName() << "\"\n";
        }
      }
      llvm::dbgs() << "-----------------------------------------------------------------\n";
    });
    return retVal;
  }

public:
  void runOnOperation() override {
    const SymbolUseGraph &useGraph = getAnalysis<SymbolUseGraph>();
    LLVM_DEBUG(useGraph.dumpToDotFile());

    SymbolTableCollection tables;
    FailureOr<InliningPlan> plan = makePlan(useGraph, tables);
    if (failed(plan)) {
      signalPassFailure(); // error already printed w/in makePlan()
      return;
    }

    if (failed(performInlining(tables, plan.value()))) {
      signalPassFailure();
      return;
    };
  }
};

} // namespace

std::unique_ptr<mlir::Pass> llzk::createInlineStructsPass() {
  return std::make_unique<InlineStructsPass>();
};
