//===-- PodToScalarPass.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-pod-to-scalar` pass.
///
/// The steps of this transformation are as follows:
///
/// 0. Scan to find `llzk.nondet` ops that allocate uninitialized pods and replace them with
///    an equivalent `pod.new`
///
/// 1. Run a dialect conversion that replaces `PodType` struct members with one scalar member per
///    record and remembers how each original member was split.
///
/// 2. Run a dialect conversion that does the following:
///
///    - Replace `MemberReadOp` and `MemberWriteOp` targeting the members that were split in step 1
///      so they instead perform scalar reads and writes from the new members. The transformation is
///      local to the current op. Therefore, when replacing the `MemberReadOp` a new pod is
///      created locally and all uses of the `MemberReadOp` are replaced with the new pod Value,
///      then each scalar member read is followed by scalar write into the new pod. Similarly,
///      when replacing a `MemberWriteOp`, each element in the pod operand needs a scalar read
///      from the pod followed by a scalar write to the new member. Making only local changes
///      keeps this step simple and later steps will optimize.
///
///    - Remove optional initialization from `NewPodOp` and instead insert a list of `WritePodOp`
///      immediately following.
///
///    - Split pods to scalars in `FuncDefOp`, `CallOp`, and `ReturnOp` and insert the necessary
///      create/read/write ops so the changes are as local as possible (just as described for
///      `MemberReadOp` and `MemberWriteOp`)
///
/// 3. Promote pod reads and writes out of `scf.if`, `scf.for`, and `scf.while` regions when the
///    access can be modeled as an SSA value flowing through the region boundary. This puts the
///    pod accesses that mem2reg must eliminate into a parent block or loop-carried value.
///
/// 4. Run MLIR "sroa" pass to split each pod with `N` records into `N` pods with 1 record each
///    (to prepare for the "mem2reg" pass because its API cannot split memory by itself).
///
/// 5. Run MLIR "mem2reg" pass to convert all single-record pod allocations and accesses into SSA
///    values.
///
/// ** Steps 4 and 5 are rerun while nested POD types are still being exposed, until a fixpoint.
///
/// Note: This transformation imposes a "last write wins" semantics on pod records. If
/// different/configurable semantics are added in the future, some additional transformation would
/// be necessary before/during this pass so that multiple writes to the same record can be handled
/// properly while they still exist.
///
/// Note: This transformation will introduce a `nondet` op when there exists a read from a pod
/// record that was not earlier written to.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Dialect.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Polymorphic/IR/Dialect.h"
#include "llzk/Dialect/RAM/IR/Dialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKConversionUtils.h"
#include "llzk/Transforms/SpecializedMemoryPasses.h"
#include "llzk/Util/Concepts.h"
#include "llzk/Util/Walk.h"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>

// Include the generated base pass class definitions.
namespace llzk::pod {
#define GEN_PASS_DEF_PODTOSCALARPASS
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h.inc"
} // namespace llzk::pod

using namespace mlir;
using namespace llzk;
using namespace llzk::pod;
using namespace llzk::function;
using namespace llzk::component;

#define DEBUG_TYPE "llzk-pod-to-scalar"

namespace {

/// Path of nested POD record names from the original member to a scalar leaf record.
struct RecordChain {
  SmallVector<StringAttr> nameList;

  RecordChain() = default;

  explicit RecordChain(ArrayRef<StringAttr> names) : nameList(names.begin(), names.end()) {}

  bool operator==(const RecordChain &other) const { return nameList == other.nameList; }
};

} // namespace

namespace llvm {
template <> struct DenseMapInfo<RecordChain> {
  static RecordChain getEmptyKey() {
    return RecordChain {{DenseMapInfo<StringAttr>::getEmptyKey()}};
  }

  static RecordChain getTombstoneKey() {
    return RecordChain {{DenseMapInfo<StringAttr>::getTombstoneKey()}};
  }

  static unsigned getHashValue(const RecordChain &chain) {
    return llvm::hash_combine_range(chain.nameList.begin(), chain.nameList.end());
  }

  static bool isEqual(const RecordChain &lhs, const RecordChain &rhs) { return lhs == rhs; }
};
} // namespace llvm

namespace {

/// If the given PodType can be split into scalars (always true for PodType), return it.
inline static PodType splittablePod(PodType pt) { return pt; }

/// If the given Type is a PodType that can be split into scalars, return it, otherwise nullptr.
inline static PodType splittablePod(Type t) {
  if (PodType pt = dyn_cast<PodType>(t)) {
    return splittablePod(pt);
  } else {
    return nullptr;
  }
}

/// Return `true` iff the given range contains any top-level PodType that this pass can split into
/// scalars.
inline static bool containsSplittablePodType(ArrayRef<Type> types) {
  for (Type t : types) {
    if (splittablePod(t)) {
      return true;
    }
  }
  return false;
}

/// Return `true` iff the given range contains any top-level PodType that this pass can split into
/// scalars.
template <typename T> static bool containsSplittablePodType(ValueTypeRange<T> types) {
  for (Type t : types) {
    if (splittablePod(t)) {
      return true;
    }
  }
  return false;
}

/// If the given Type is a PodType that can be split into scalars, append `collect` with all of
/// the scalar types that result from splitting the PodType. Otherwise, just push the `Type`.
size_t splitPodTypeTo(Type t, SmallVector<Type> &collect) {
  if (PodType pt = splittablePod(t)) {
    auto records = pt.getRecords();
    for (RecordAttr record : records) {
      collect.push_back(record.getType());
    }
    return records.size();
  } else {
    collect.push_back(t);
    return 1;
  }
}

/// For each Type in the given input collection, call `splitPodTypeTo(Type,...)`.
template <typename TypeCollection>
inline void splitPodTypeTo(
    TypeCollection types, SmallVector<Type> &collect, SmallVector<size_t> *originalIdxToSize
) {
  for (Type t : types) {
    size_t count = splitPodTypeTo(t, collect);
    if (originalIdxToSize) {
      originalIdxToSize->push_back(count);
    }
  }
}

/// Return a list such that each scalar Type is directly added to the list but for each splittable
/// PodType, the proper number of scalar element types are added instead.
template <typename TypeCollection>
inline SmallVector<Type>
splitPodType(TypeCollection types, SmallVector<size_t> *originalIdxToSize = nullptr) {
  SmallVector<Type> collect;
  splitPodTypeTo(types, collect, originalIdxToSize);
  return collect;
}

/// Create a `pod.read` for one record of `podRef`.
inline static ReadPodOp
genRead(Location loc, Value podRef, StringAttr recordName, OpBuilder &rewriter) {
  Type resultType =
      llvm::cast<PodType>(podRef.getType()).getRecordMap().lookup(recordName.getValue());
  return rewriter.create<ReadPodOp>(loc, resultType, podRef, recordName);
}

/// Create a `pod.write` for one record of `podRef`.
inline static WritePodOp
genWrite(Location loc, Value podRef, StringAttr recordName, Value value, OpBuilder &rewriter) {
  return rewriter.create<WritePodOp>(loc, podRef, recordName, value);
}

/// Return the suffixes to append to a function arg/result name when splitting the given type.
static SmallVector<std::string> getSplitRecordNameSuffixes(Type type) {
  SmallVector<std::string> suffixes;
  if (PodType pt = splittablePod(type)) {
    suffixes.reserve(pt.getRecords().size());
    for (RecordAttr record : pt.getRecords()) {
      StringRef name = record.getName().getValue();
      std::string result;
      result.reserve(name.size() + 1);
      result.push_back('.');
      result.append(name.data(), name.size());
      suffixes.push_back(result);
    }
  }
  return suffixes;
}

// If the operand has PodType, add reads from all pod records to the `newOperands` list otherwise
// add the original operand to the list.
static void processInputOperand(
    Location loc, Value operand, SmallVector<Value> &newOperands,
    ConversionPatternRewriter &rewriter
) {
  if (PodType pt = splittablePod(operand.getType())) {
    for (RecordAttr record : pt.getRecords()) {
      newOperands.push_back(genRead(loc, operand, record.getName(), rewriter));
    }
  } else {
    newOperands.push_back(operand);
  }
}

/// For each operand with PodType, add reads from all pod records in place of the original operand
/// and update the op to use the new operands.
static void processInputOperands(
    ValueRange operands, MutableOperandRange outputOpRef, Operation *op,
    ConversionPatternRewriter &rewriter
) {
  SmallVector<Value> newOperands;
  for (Value v : operands) {
    processInputOperand(op->getLoc(), v, newOperands, rewriter);
  }
  rewriter.modifyOpInPlace(op, [&outputOpRef, &newOperands]() {
    outputOpRef.assign(ValueRange(newOperands));
  });
}

/// Register the dialects and operations that remain legal across the conversion-based stages.
inline static void baseTargetSetup(ConversionTarget &target) {
  target.addLegalDialect<
      LLZKDialect, array::ArrayDialect, boolean::BoolDialect, cast::CastDialect,
      constrain::ConstrainDialect, component::StructDialect, felt::FeltDialect,
      function::FunctionDialect, global::GlobalDialect, include::IncludeDialect, pod::PODDialect,
      polymorphic::PolymorphicDialect, ram::RAMDialect, string::StringDialect, arith::ArithDialect,
      scf::SCFDialect>();
  target.addLegalOp<ModuleOp>();
}

/// Rewrite pod-typed `llzk.nondet` allocations into explicit `pod.new` allocations so the rest of
/// the pass only needs to reason about POD storage through POD dialect operations.
class NondetToNewPod : public OpConversionPattern<NonDetOp> {
  using OpConversionPattern<NonDetOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      NonDetOp nondetOp, OpAdaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (auto pt = dyn_cast<PodType>(nondetOp.getType())) {
      rewriter.replaceOpWithNewOp<NewPodOp>(nondetOp, pt);
      return success();
    }
    return failure();
  }
};

/// Prepare the module by replacing `llzk.nondet` pod allocation ops with `pod.new`.
static LogicalResult step0(ModuleOp modOp) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns {ctx};
  patterns.add<NondetToNewPod>(ctx);
  ConversionTarget target {*ctx};

  baseTargetSetup(target);
  target.addDynamicallyLegalOp<NonDetOp>([](NonDetOp op) { return !isa<PodType>(op.getType()); });

  return applyFullConversion(modOp, target, std::move(patterns));
}

/// new member name and type
using MemberInfo = std::pair<StringAttr, Type>;
/// original nested pod record name chain -> split scalar member info
using LocalMemberReplacementMap = DenseMap<RecordChain, MemberInfo>;
/// struct -> original pod-type member name -> LocalMemberReplacementMap
using MemberReplacementMap = DenseMap<StructDefOp, DenseMap<StringAttr, LocalMemberReplacementMap>>;

/// Build a flattened struct-member name like `member_outer_inner_leaf`.
static StringAttr
getFlattenedMemberName(MLIRContext *ctx, StringAttr memberName, ArrayRef<StringAttr> recordChain) {
  std::string flatName;
  llvm::raw_string_ostream os(flatName);
  os << memberName.getValue();
  for (StringAttr recordName : recordChain) {
    os << '_' << recordName.getValue();
  }
  return StringAttr::get(ctx, flatName);
}

/// Recursively create scalar leaf members for a POD-typed struct member.
static void flattenPodMemberIntoLeaves(
    MemberDefOp originalMember, PodType podTy, SmallVectorImpl<StringAttr> &recordChain,
    LocalMemberReplacementMap &localRepMapRef, SymbolTable &structSymbolTable,
    ConversionPatternRewriter &rewriter
) {
  for (RecordAttr record : podTy.getRecords()) {
    recordChain.push_back(record.getName());
    if (PodType nestedPodTy = dyn_cast<PodType>(record.getType())) {
      flattenPodMemberIntoLeaves(
          originalMember, nestedPodTy, recordChain, localRepMapRef, structSymbolTable, rewriter
      );
      recordChain.pop_back();
      continue;
    }

    StringAttr name = getFlattenedMemberName(
        originalMember.getContext(), originalMember.getSymNameAttr(), recordChain
    );
    Type ty = record.getType();
    MemberDefOp newMember = rewriter.create<MemberDefOp>(
        originalMember.getLoc(), name, ty, originalMember.getSignal(), originalMember.getColumn()
    );
    newMember.setPublicAttr(originalMember.hasPublicAttr());
    localRepMapRef[RecordChain(recordChain)] =
        std::make_pair(structSymbolTable.insert(newMember), ty);
    recordChain.pop_back();
  }
}

/// Split a pod-typed struct member definition into one scalar member definition per POD record.
///
/// The replacement map records the fresh member symbols so later rewrites can retarget
/// `struct.readm` and `struct.writem` operations to the split members.
class SplitPodInMemberDefOp : public OpConversionPattern<MemberDefOp> {
  SymbolTableCollection &tables;
  MemberReplacementMap &repMapRef;

public:
  SplitPodInMemberDefOp(
      MLIRContext *ctx, SymbolTableCollection &symTables, MemberReplacementMap &memberRepMap
  )
      : OpConversionPattern<MemberDefOp>(ctx), tables(symTables), repMapRef(memberRepMap) {}

  inline static bool legal(MemberDefOp op) { return !splittablePod(op.getType()); }

  LogicalResult match(MemberDefOp op) const override { return failure(legal(op)); }

  void
  rewrite(MemberDefOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    StructDefOp inStruct = op->getParentOfType<StructDefOp>();
    assert(inStruct);
    LocalMemberReplacementMap &localRepMapRef = repMapRef[inStruct][op.getSymNameAttr()];

    PodType podTy = llvm::cast<PodType>(adaptor.getType()); // safe per legal() check

    SymbolTable &structSymbolTable = tables.getSymbolTable(inStruct);
    SmallVector<StringAttr> recordChain;
    flattenPodMemberIntoLeaves(op, podTy, recordChain, localRepMapRef, structSymbolTable, rewriter);
    rewriter.eraseOp(op);
  }
};

/// Replace `PodType` struct members with scalar members.
static LogicalResult
step1(ModuleOp modOp, SymbolTableCollection &symTables, MemberReplacementMap &memberRepMap) {
  MLIRContext *ctx = modOp.getContext();

  RewritePatternSet patterns(ctx);

  patterns.add<SplitPodInMemberDefOp>(ctx, symTables, memberRepMap);

  ConversionTarget target(*ctx);
  baseTargetSetup(target);
  target.addDynamicallyLegalOp<MemberDefOp>(SplitPodInMemberDefOp::legal);

  LLVM_DEBUG(llvm::dbgs() << "Begin step 1: split pod-type members\n";);
  return applyFullConversion(modOp, target, std::move(patterns));
}

/// Split inline `pod.new` initializers into explicit `pod.write` operations.
class SplitInitFromNewPodOp : public OpConversionPattern<NewPodOp> {
public:
  using OpConversionPattern<NewPodOp>::OpConversionPattern;

  static bool legal(NewPodOp op) { return op.getInitialValues().empty(); }

  LogicalResult match(NewPodOp op) const override { return failure(legal(op)); }

  void rewrite(NewPodOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // Generate an individual write for each initialization
    rewriter.setInsertionPointAfter(op);
    Location loc = op.getLoc();
    for (auto [name, init] :
         llvm::zip_equal(adaptor.getInitializedRecords(), adaptor.getInitialValues())) {
      // Create the write
      rewriter.create<WritePodOp>(loc, op.getResult(), llvm::cast<StringAttr>(name), init);
    }
    // Remove initializations from `op`
    rewriter.modifyOpInPlace(op, [&op]() {
      op.getInitialValuesMutable().clear();
      op.setInitializedRecordsAttr(ArrayAttr::get(op.getContext(), {})); // DefaultValuedAttr:{}
    });
  }
};

/// Rewrite pod-typed function signatures to pass one scalar per POD record instead.
///
/// Each pod argument is expanded into one scalar argument per record, and each pod result is
/// expanded into one scalar result per record. Inside the rewritten function body, the original
/// POD-typed block arguments are reconstructed locally with `pod.new` plus `pod.write` so the
/// rest of the function can continue to use POD values until later cleanup passes scalarize those
/// local temporaries away.
class SplitPodInFuncDefOp : public OpConversionPattern<FuncDefOp> {
public:
  using OpConversionPattern<FuncDefOp>::OpConversionPattern;

  inline static bool legal(FuncDefOp op) {
    return !containsSplittablePodType(op.getArgumentTypes()) &&
           !containsSplittablePodType(op.getResultTypes());
  }

  LogicalResult match(FuncDefOp op) const override { return failure(legal(op)); }

  void rewrite(FuncDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    // Update in/out types of the function to replace pods with scalars
    class Impl : public FunctionTypeConverter {
      SmallVector<size_t> originalInputIdxToSize, originalResultIdxToSize;
      SplitFunctionNameInfo inputNameInfo;
      SplitFunctionNameInfo resultNameInfo;

    protected:
      SmallVector<Type> convertInputs(ArrayRef<Type> origTypes) override {
        return splitPodType(origTypes, &originalInputIdxToSize);
      }
      SmallVector<Type> convertResults(ArrayRef<Type> origTypes) override {
        return splitPodType(origTypes, &originalResultIdxToSize);
      }
      ArrayAttr convertInputAttrs(ArrayAttr origAttrs, SmallVector<Type> newTypes) override {
        return replicateFunctionNameAttrsAsNeeded(
            origAttrs, originalInputIdxToSize, newTypes, ARG_NAME_ATTR_NAME,
            inputNameInfo.originalNames, inputNameInfo.existingNames,
            inputNameInfo.splitNameSuffixes
        );
      }
      ArrayAttr convertResultAttrs(ArrayAttr origAttrs, SmallVector<Type> newTypes) override {
        return replicateFunctionNameAttrsAsNeeded(
            origAttrs, originalResultIdxToSize, newTypes, RES_NAME_ATTR_NAME,
            resultNameInfo.originalNames, resultNameInfo.existingNames,
            resultNameInfo.splitNameSuffixes
        );
      }

      /// For each argument to the Block that has a splittable PodType, replace it with the
      /// necessary number of scalar arguments, generate a NewPodOp, and generate writes from
      /// the new block scalar arguments to the new pod. All users of the original block
      /// argument are updated to target the result of the NewPodOp.
      void processBlockArgs(Block &entryBlock, RewriterBase &rewriter) override {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&entryBlock);

        for (unsigned i = 0; i < entryBlock.getNumArguments();) {
          Value oldV = entryBlock.getArgument(i);
          if (PodType pt = splittablePod(oldV.getType())) {
            Location loc = oldV.getLoc();
            // Generate `NewPodOp` and replace uses of the argument with it.
            auto newPod = rewriter.create<NewPodOp>(loc, pt);
            rewriter.replaceAllUsesWith(oldV, newPod);
            // Remove the argument from the block
            entryBlock.eraseArgument(i);
            // For all indices in the PodType (i.e., the element count), generate a new
            // block argument and a write of that argument to the new pod.
            for (RecordAttr record : pt.getRecords()) {
              BlockArgument newArg = entryBlock.insertArgument(i, record.getType(), loc);
              genWrite(loc, newPod, record.getName(), newArg, rewriter);
              ++i;
            }
          } else {
            ++i;
          }
        }
      }

    public:
      Impl(FuncDefOp op) {
        inputNameInfo = collectSplitFunctionNameInfo(op.getArgumentTypes(), [&op](unsigned i) {
          return op.getArgNameAttr(i);
        }, getSplitRecordNameSuffixes);
        resultNameInfo = collectSplitFunctionNameInfo(
            op.getResultTypes(), [resultAttrs = op.getAllResultAttrs()](unsigned i) {
          return getAttrAtIndexWithName(resultAttrs, i, RES_NAME_ATTR_NAME);
        }, getSplitRecordNameSuffixes
        );
      }
    };
    Impl(op).convert(op, rewriter);
  }
};

/// Rewrite `function.return` to flatten any POD operands into their scalar record values.
///
/// This mirrors the function-signature conversion performed by `SplitPodInFuncDefOp`: POD results
/// are returned as one SSA value per record, using local `pod.read` operations to extract the
/// scalar pieces immediately before the return.
class SplitPodInReturnOp : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  inline static bool legal(ReturnOp op) {
    return !containsSplittablePodType(op.getOperands().getTypes());
  }

  LogicalResult match(ReturnOp op) const override { return failure(legal(op)); }

  void rewrite(ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    processInputOperands(adaptor.getOperands(), op.getOperandsMutable(), op, rewriter);
  }
};

/// Rebuild a call with split scalar results, then reconstruct POD-typed results locally.
static CallOp newCallOpWithSplitResults(
    CallOp oldCall, CallOp::Adaptor adaptor, ConversionPatternRewriter &rewriter
) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(oldCall);

  Operation::result_range oldResults = oldCall.getResults();
  CallOp newCall = createCallPreservingInstantiationOperands(
      oldCall.getLoc(), splitPodType(oldResults.getTypes()), oldCall, adaptor.getMapOperands(),
      adaptor.getArgOperands(), rewriter
  );

  auto newResults = newCall.getResults().begin();
  for (Value oldVal : oldResults) {
    if (PodType pt = splittablePod(oldVal.getType())) {
      Location loc = oldVal.getLoc();
      // Generate `NewPodOp` and replace uses of the result with it.
      auto newPod = rewriter.create<NewPodOp>(loc, pt);
      rewriter.replaceAllUsesWith(oldVal, newPod);

      // For each record in the PodType, write the next result from the new CallOp to the new pod.
      for (RecordAttr record : pt.getRecords()) {
        genWrite(loc, newPod, record.getName(), *newResults, rewriter);
        newResults++;
      }
    } else {
      rewriter.replaceAllUsesWith(oldVal, *newResults);
      newResults++;
    }
  }
  // erase the original CallOp
  rewriter.eraseOp(oldCall);

  return newCall;
}

/// Rewrite calls whose arguments or results contain PODs to use flattened scalar signatures.
///
/// POD arguments are decomposed into scalar record operands before the new call is formed. POD
/// results are reconstructed locally after the call with `pod.new` plus `pod.write`, preserving
/// the original POD-typed uses in the caller until later optimization passes remove the temporary
/// POD allocations.
class SplitPodInCallOp : public OpConversionPattern<CallOp> {
public:
  using OpConversionPattern<CallOp>::OpConversionPattern;

  inline static bool legal(CallOp op) {
    return !containsSplittablePodType(op.getArgOperands().getTypes()) &&
           !containsSplittablePodType(op.getResultTypes());
  }

  LogicalResult match(CallOp op) const override { return failure(legal(op)); }

  void rewrite(CallOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // Create new CallOp with split results first so, then process its inputs to split types
    CallOp newCall = newCallOpWithSplitResults(op, adaptor, rewriter);
    processInputOperands(
        newCall.getArgOperands(), newCall.getArgOperandsMutable(), newCall, rewriter
    );
  }
};

/// Read a nested POD leaf by following each record name in `recordChain`.
static Value
genReadAlongPath(Location loc, Value podRef, RecordChain recordChain, OpBuilder &rewriter) {
  Value value = podRef;
  for (StringAttr attr : recordChain.nameList) {
    value = genRead(loc, value, attr, rewriter);
  }
  return value;
}

/// State used while rebuilding a POD from flattened struct-member leaves.
struct RebuildPodReadState {
  NewPodOp pod;
  DenseMap<RecordChain, Value> leafValues;
};

/// Reconstruct a POD record from the leaf values collected while splitting `struct.readm`.
static Value rebuildFlattenedPodRecord(
    Location loc, Type recordType, SmallVectorImpl<StringAttr> &recordChain,
    const DenseMap<RecordChain, Value> &leafValues, ConversionPatternRewriter &rewriter
) {
  if (PodType nestedPodTy = dyn_cast<PodType>(recordType)) {
    NewPodOp nestedPod = rewriter.create<NewPodOp>(loc, nestedPodTy);
    for (RecordAttr record : nestedPodTy.getRecords()) {
      recordChain.push_back(record.getName());
      Value recordValue =
          rebuildFlattenedPodRecord(loc, record.getType(), recordChain, leafValues, rewriter);
      genWrite(loc, nestedPod, record.getName(), recordValue, rewriter);
      recordChain.pop_back();
    }
    return nestedPod;
  }

  auto it = leafValues.find(RecordChain(recordChain));
  assert(it != leafValues.end() && "missing flattened POD leaf value");
  return it->second;
}

/// Rewrite a write to a pod-typed struct member into writes to the corresponding scalar leaves.
class SplitPodInMemberWriteOp : public SplitAggregateInMemberRefOp<
                                    SplitPodInMemberWriteOp, MemberWriteOp, void *, RecordChain> {
public:
  using SplitAggregateInMemberRefOp<
      SplitPodInMemberWriteOp, MemberWriteOp, void *, RecordChain>::SplitAggregateInMemberRefOp;

  static bool legal(MemberWriteOp op) { return !containsSplittablePodType(op.getVal().getType()); }

  static void *genHeader(MemberWriteOp, ConversionPatternRewriter &) { return nullptr; }

  static void forId(
      Location loc, void *&, RecordChain id, MemberInfo newMember, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter
  ) {
    Value scalarRead = genReadAlongPath(loc, adaptor.getVal(), id, rewriter);
    rewriter.create<MemberWriteOp>(
        loc, adaptor.getComponent(), FlatSymbolRefAttr::get(newMember.first), scalarRead
    );
  }
};

/// Rewrite a read from a pod-typed struct member into reads from the corresponding scalar leaves.
class SplitPodInMemberReadOp
    : public SplitAggregateInMemberRefOp<
          SplitPodInMemberReadOp, MemberReadOp, RebuildPodReadState, RecordChain> {
public:
  using SplitAggregateInMemberRefOp<
      SplitPodInMemberReadOp, MemberReadOp, RebuildPodReadState,
      RecordChain>::SplitAggregateInMemberRefOp;

  static bool legal(MemberReadOp op) {
    return !containsSplittablePodType(op.getResult().getType());
  }

  static RebuildPodReadState genHeader(MemberReadOp op, ConversionPatternRewriter &rewriter) {
    RebuildPodReadState state;
    state.pod = rewriter.create<NewPodOp>(op.getLoc(), llvm::cast<PodType>(op.getType()));
    rewriter.replaceAllUsesWith(op, state.pod);
    return state;
  }

  static void forId(
      Location loc, RebuildPodReadState &state, RecordChain id, MemberInfo newMember,
      OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) {
    Value scalarRead = rewriter.create<MemberReadOp>(
        loc, newMember.second, adaptor.getComponent(), newMember.first
    );
    state.leafValues[id] = scalarRead;
  }

  static void finalize(
      MemberReadOp op, RebuildPodReadState &state, OpAdaptor, ConversionPatternRewriter &rewriter
  ) {
    auto podTy = llvm::cast<PodType>(op.getType());
    SmallVector<StringAttr> recordChain;
    for (RecordAttr record : podTy.getRecords()) {
      recordChain.push_back(record.getName());
      Value recordValue = rebuildFlattenedPodRecord(
          op.getLoc(), record.getType(), recordChain, state.leafValues, rewriter
      );
      genWrite(op.getLoc(), state.pod, record.getName(), recordValue, rewriter);
      recordChain.pop_back();
    }
  }
};

/// Special handling to split pods in struct member refs and function signatures and desugar
/// initializations on pod.new into pod writes.
static LogicalResult
step2(ModuleOp modOp, SymbolTableCollection &symTables, const MemberReplacementMap &memberRepMap) {
  MLIRContext *ctx = modOp.getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<
      // clang-format off
      SplitInitFromNewPodOp,
      SplitPodInFuncDefOp,
      SplitPodInReturnOp,
      SplitPodInCallOp
      // clang-format on
      >(ctx);

  patterns.add<
      // clang-format off
      SplitPodInMemberWriteOp,
      SplitPodInMemberReadOp
      // clang-format on
      >(ctx, symTables, memberRepMap);

  ConversionTarget target(*ctx);
  baseTargetSetup(target);
  target.addDynamicallyLegalOp<NewPodOp>(SplitInitFromNewPodOp::legal);
  target.addDynamicallyLegalOp<FuncDefOp>(SplitPodInFuncDefOp::legal);
  target.addDynamicallyLegalOp<ReturnOp>(SplitPodInReturnOp::legal);
  target.addDynamicallyLegalOp<CallOp>(SplitPodInCallOp::legal);
  target.addDynamicallyLegalOp<MemberWriteOp>(SplitPodInMemberWriteOp::legal);
  target.addDynamicallyLegalOp<MemberReadOp>(SplitPodInMemberReadOp::legal);

  LLVM_DEBUG(llvm::dbgs() << "Begin step 2: update/split other pod ops\n";);
  return applyFullConversion(modOp, target, std::move(patterns));
}

/// Normalize the record name representation used by POD access ops to a plain `StringAttr`.
inline static StringAttr getRecordNameAsStringAttr(ReadPodOp readOp) {
  return readOp.getRecordNameAttr().getLeafReference();
}

/// Normalize the record name representation used by POD access ops to a plain `StringAttr`.
inline static StringAttr getRecordNameAsStringAttr(WritePodOp writeOp) {
  return writeOp.getRecordNameAttr().getLeafReference();
}

/// Return whether the given read/write access targets the same POD record.
inline static bool isSamePodRecord(ReadPodOp readOp, Value podRef, StringAttr recordName) {
  return readOp.getPodRef() == podRef && getRecordNameAsStringAttr(readOp) == recordName;
}

/// Return whether the given read/write access targets the same POD record.
inline static bool isSamePodRecord(WritePodOp writeOp, Value podRef, StringAttr recordName) {
  return writeOp.getPodRef() == podRef && getRecordNameAsStringAttr(writeOp) == recordName;
}

/// Return whether `op` contains a nested write to `podRef.recordName`.
static bool hasNestedWriteToRecord(Operation &op, Value podRef, StringAttr recordName) {
  return walkContainsMatch<WritePodOp>(op, [&](WritePodOp writeOp) {
    return writeOp.getOperation() != &op && isSamePodRecord(writeOp, podRef, recordName);
  });
}

/// Return whether `op` contains any read from `podRef.recordName`.
static bool hasReadFromRecord(Operation &op, Value podRef, StringAttr recordName) {
  return walkContainsMatch<ReadPodOp>(op, [&](ReadPodOp readOp) {
    return isSamePodRecord(readOp, podRef, recordName);
  });
}

/// Return whether `op` or any nested operation uses `value` as an operand.
static bool hasValueUse(Operation &op, Value value) {
  return walkContainsMatch<Operation *>(op, [&value](Operation *nestedOp) {
    return llvm::is_contained(nestedOp->getOperands(), value);
  });
}

/// Return whether the read is preceded by a write to the same pod record within its block.
static bool hasEarlierWriteInBlock(ReadPodOp readOp) {
  Value podRef = readOp.getPodRef();
  StringAttr recordName = getRecordNameAsStringAttr(readOp);

  for (Operation &op : *readOp->getBlock()) {
    if (&op == readOp.getOperation()) {
      return false;
    }

    if (auto writeOp = dyn_cast<WritePodOp>(&op)) {
      if (isSamePodRecord(writeOp, podRef, recordName)) {
        return true;
      }
      continue;
    }

    if (hasNestedWriteToRecord(op, podRef, recordName)) {
      return true;
    }
  }
  return false;
}

/// Return whether `value` is defined within `ancestor` or one of its nested regions.
///
/// Values defined inside a control-flow operation cannot be hoisted across that operation without
/// introducing an explicit region result or loop-carried value.
static bool isValueDefinedInside(Operation *ancestor, Value value) {
  if (Operation *defOp = value.getDefiningOp()) {
    return ancestor->isAncestor(defOp);
  }

  auto blockArg = llvm::dyn_cast<BlockArgument>(value);
  Operation *parentOp = blockArg.getOwner()->getParentOp();
  return parentOp && ancestor->isAncestor(parentOp);
}

/// Find a write before the parent `scf.if` that can be forwarded to `readOp`.
static WritePodOp findPrecedingWriteForIfRead(ReadPodOp readOp) {
  auto ifOp = readOp->getParentOfType<scf::IfOp>();
  if (!ifOp || readOp->getBlock()->getParentOp() != ifOp.getOperation()) {
    return nullptr;
  }
  if (hasEarlierWriteInBlock(readOp)) {
    return nullptr;
  }

  Block *ifBlock = ifOp->getBlock();
  if (!ifBlock) {
    return nullptr;
  }

  Value podRef = readOp.getPodRef();
  StringAttr recordName = getRecordNameAsStringAttr(readOp);
  WritePodOp replacement = nullptr;
  for (Operation &op : *ifBlock) {
    if (&op == ifOp.getOperation()) {
      break;
    }

    if (auto writeOp = dyn_cast<WritePodOp>(&op)) {
      if (isSamePodRecord(writeOp, podRef, recordName)) {
        replacement = writeOp;
      }
      continue;
    }

    if (hasNestedWriteToRecord(op, podRef, recordName)) {
      replacement = nullptr;
    }
  }

  return replacement;
}

/// Replace branch-local reads with a value available in the parent block.
static bool replaceIfReads(ModuleOp modOp) {
  SmallVector<std::pair<ReadPodOp, Value>> replacements;
  OpBuilder builder(modOp.getContext());
  modOp.walk([&](ReadPodOp readOp) {
    auto ifOp = readOp->getParentOfType<scf::IfOp>();
    if (!ifOp || readOp->getBlock()->getParentOp() != ifOp.getOperation()) {
      return;
    }
    if (isValueDefinedInside(ifOp, readOp.getPodRef())) {
      return;
    }
    if (hasEarlierWriteInBlock(readOp)) {
      return;
    }

    if (WritePodOp writeOp = findPrecedingWriteForIfRead(readOp)) {
      replacements.emplace_back(readOp, writeOp.getValue());
      return;
    }

    builder.setInsertionPoint(ifOp);
    replacements.emplace_back(
        readOp,
        genRead(readOp.getLoc(), readOp.getPodRef(), getRecordNameAsStringAttr(readOp), builder)
    );
  });

  for (auto [readOp, value] : replacements) {
    readOp.getResult().replaceAllUsesWith(value);
    readOp.erase();
  }
  return !replacements.empty();
}

/// Fold reads from an `scf.if`-carried POD result when the same record was just written from
/// another result of that same `scf.if`.
///
/// Pattern:
///   %if:2 = scf.if ... -> (!pod<...>, T) {
///     scf.yield %pod, %v0
///   } else {
///     scf.yield %pod, %v1
///   }
///   pod.write %pod[@r] = %if#1
///   %x = pod.read %if#0[@r]
///
/// Rewritten to `%x = %if#1` when all yielded values for `%if#0` are the same `%pod`.
static bool foldIfCarriedPodReadAfterWrite(ModuleOp modOp) {
  SmallVector<std::pair<ReadPodOp, Value>> replacements;

  modOp.walk([&](ReadPodOp readOp) {
    auto podRes = dyn_cast<OpResult>(readOp.getPodRef());
    if (!podRes) {
      return;
    }

    auto ifOp = dyn_cast<scf::IfOp>(podRes.getOwner());
    if (!ifOp) {
      return;
    }

    auto writeOp = dyn_cast_or_null<WritePodOp>(readOp->getPrevNode());
    if (!writeOp || getRecordNameAsStringAttr(writeOp) != getRecordNameAsStringAttr(readOp)) {
      return;
    }

    auto valueRes = dyn_cast<OpResult>(writeOp.getValue());
    if (!valueRes || valueRes.getOwner() != ifOp.getOperation()) {
      return;
    }

    Value carriedPod = writeOp.getPodRef();
    unsigned podResultIndex = podRes.getResultNumber();

    auto thenYield = dyn_cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    if (!thenYield || thenYield.getOperand(podResultIndex) != carriedPod) {
      return;
    }

    Region &elseRegion = ifOp.getElseRegion();
    if (Block *elseBlock = elseRegion.empty() ? nullptr : &elseRegion.front()) {
      auto elseYield = dyn_cast<scf::YieldOp>(elseBlock->getTerminator());
      if (!elseYield || elseYield.getOperand(podResultIndex) != carriedPod) {
        return;
      }
    }

    replacements.emplace_back(readOp, valueRes);
  });

  for (auto [readOp, replacement] : replacements) {
    readOp.getResult().replaceAllUsesWith(replacement);
    readOp.erase();
  }
  return !replacements.empty();
}

struct IfWriteSlot {
  Value podRef;
  StringAttr recordName;
  Type type;
  WritePodOp thenWrite;
  WritePodOp elseWrite;
  Value incomingValue;
};

/// Find the tracked branch-write slot for `podRef.recordName`.
static IfWriteSlot *
lookupSlot(SmallVectorImpl<IfWriteSlot> &slots, Value podRef, StringAttr recordName) {
  for (IfWriteSlot &slot : slots) {
    if (slot.podRef == podRef && slot.recordName == recordName) {
      return &slot;
    }
  }
  return nullptr;
}

/// Return the tracked branch-write slot for `podRef.recordName`, creating it on first use.
static IfWriteSlot &getOrCreateSlot(
    SmallVectorImpl<IfWriteSlot> &slots, Value podRef, StringAttr recordName, Type type
) {
  if (IfWriteSlot *slot = lookupSlot(slots, podRef, recordName)) {
    return *slot;
  }
  slots.push_back(IfWriteSlot {podRef, recordName, type, nullptr, nullptr, Value()});
  return slots.back();
}

/// Return the else block when present; otherwise null.
static Block *getElseBlockOrNull(scf::IfOp ifOp) {
  return ifOp.getElseRegion().empty() ? nullptr : &ifOp.getElseRegion().front();
}

/// Collect direct `pod.write` operations from a branch, grouped by POD reference and record name.
static void
collectDirectWrites(Block *block, bool isThenBlock, SmallVectorImpl<IfWriteSlot> &slots) {
  if (!block) {
    return;
  }

  for (Operation &op : *block) {
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      break;
    }

    auto writeOp = dyn_cast<WritePodOp>(&op);
    if (!writeOp) {
      continue;
    }

    IfWriteSlot &slot = getOrCreateSlot(
        slots, writeOp.getPodRef(), getRecordNameAsStringAttr(writeOp), writeOp.getValue().getType()
    );
    if (isThenBlock) {
      slot.thenWrite = writeOp;
    } else {
      slot.elseWrite = writeOp;
    }
  }
}

/// Return whether writes to `podRef.recordName` can be lifted out of the branch as an SSA result.
///
/// Lifting is rejected when nested writes may alias the same record or when a later read in the
/// branch would observe branch-local mutation ordering that the lifted form would not preserve.
static bool branchSlotCanBeLifted(Block *block, Value podRef, StringAttr recordName) {
  if (!block) {
    return true;
  }

  bool seenDirectWrite = false;
  for (Operation &op : *block) {
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      return true;
    }

    if (auto writeOp = dyn_cast<WritePodOp>(&op)) {
      if (isSamePodRecord(writeOp, podRef, recordName)) {
        seenDirectWrite = true;
        continue;
      }
    }

    if (hasNestedWriteToRecord(op, podRef, recordName)) {
      return false;
    }
    if (seenDirectWrite && (hasReadFromRecord(op, podRef, recordName) || hasValueUse(op, podRef))) {
      return false;
    }
  }
  return true;
}

/// Return whether `op` is one of the branch writes that will be recreated after the lifted `if`.
static bool isLiftedWrite(Operation &op, ArrayRef<IfWriteSlot> slots) {
  auto writeOp = dyn_cast<WritePodOp>(&op);
  return writeOp && llvm::any_of(slots, [&writeOp](const IfWriteSlot &slot) {
    return isSamePodRecord(writeOp, slot.podRef, slot.recordName);
  });
}

/// Return the `scf.yield` terminator from `block`.
static scf::YieldOp getYieldOp(Block &block) {
  auto yieldOp = dyn_cast<scf::YieldOp>(block.getTerminator());
  assert(yieldOp && "expected scf.if branch to terminate with scf.yield");
  return yieldOp;
}

/// Remove the default terminator from a freshly created SCF block before cloning contents into it.
static void dropTerminatorIfPresent(Block &block) {
  if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>()) {
    block.back().erase();
  }
}

/// Move non-lifted branch operations into the replacement branch block.
static void
moveBranchWithoutLiftedWrites(Block *srcBlock, Block &destBlock, ArrayRef<IfWriteSlot> slots) {
  if (srcBlock) {
    for (auto it = srcBlock->begin(), end = srcBlock->end(); it != end;) {
      Operation &op = *it++;
      if (op.hasTrait<OpTrait::IsTerminator>() || isLiftedWrite(op, slots)) {
        continue;
      }
      op.moveBefore(&destBlock, destBlock.end());
    }
  }
}

/// Finish a lifted branch by yielding the original branch results followed by one lifted POD value
/// per tracked record.
static void appendYield(
    Location loc, Block &block, ValueRange priorYieldValues, ArrayRef<IfWriteSlot> slots,
    bool isThenBlock, OpBuilder &builder
) {
  SmallVector<Value> yieldValues = llvm::to_vector(priorYieldValues);
  llvm::append_range(yieldValues, llvm::map_range(slots, [isThenBlock](const IfWriteSlot &slot) {
    WritePodOp writeOp = isThenBlock ? slot.thenWrite : slot.elseWrite;
    return writeOp ? writeOp.getValue() : slot.incomingValue;
  }));

  builder.setInsertionPointToEnd(&block);
  builder.create<scf::YieldOp>(loc, yieldValues);
}

struct LoopPodSlot {
  Value podRef;
  StringAttr recordName;
  Type type;

  /// Return whether this slot is `findPodRef.findRecordName`.
  bool matches(Value findPodRef, StringAttr findRecordName) const {
    return this->podRef == findPodRef && this->recordName == findRecordName;
  }
};

/// Return the tracked loop slot for `podRef.recordName`, or null if not found.
static LoopPodSlot *
lookupLoopSlot(SmallVectorImpl<LoopPodSlot> &slots, Value podRef, StringAttr recordName) {
  auto it = llvm::find_if(slots, [&podRef, &recordName](const LoopPodSlot &slot) {
    return slot.matches(podRef, recordName);
  });
  return it == slots.end() ? nullptr : &*it;
}

/// Return whether a loop slot is tracked for `podRef.recordName`.
static bool hasLoopSlot(ArrayRef<LoopPodSlot> slots, Value podRef, StringAttr recordName) {
  auto it = llvm::find_if(slots, [&podRef, &recordName](const LoopPodSlot &slot) {
    return slot.matches(podRef, recordName);
  });
  return it != slots.end();
}

/// Return the tracked loop slot for `podRef.recordName`, creating it on first use.
static LoopPodSlot &getOrCreateLoopSlot(
    SmallVectorImpl<LoopPodSlot> &slots, Value podRef, StringAttr recordName, Type type
) {
  if (LoopPodSlot *slot = lookupLoopSlot(slots, podRef, recordName)) {
    return *slot;
  }
  slots.push_back(LoopPodSlot {podRef, recordName, type});
  return slots.back();
}

/// Find the stable index of the tracked loop slot for `podRef.recordName`.
static std::optional<size_t>
findLoopSlotIndex(ArrayRef<LoopPodSlot> slots, Value podRef, StringAttr recordName) {
  for (auto [idx, slot] : llvm::enumerate(slots)) {
    if (slot.podRef == podRef && slot.recordName == recordName) {
      return idx;
    }
  }
  return std::nullopt;
}

/// Collect direct POD reads and writes that cross the loop boundary and therefore need an
/// explicit loop-carried scalar value when lifted.
static void
collectDirectLoopPodSlots(Block &block, Operation *ancestor, SmallVectorImpl<LoopPodSlot> &slots) {
  for (Operation &op : block) {
    if (auto readOp = dyn_cast<ReadPodOp>(&op)) {
      if (!isValueDefinedInside(ancestor, readOp.getPodRef())) {
        getOrCreateLoopSlot(
            slots, readOp.getPodRef(), getRecordNameAsStringAttr(readOp), readOp.getType()
        );
      }
      continue;
    }

    if (auto writeOp = dyn_cast<WritePodOp>(&op)) {
      if (!isValueDefinedInside(ancestor, writeOp.getPodRef())) {
        getOrCreateLoopSlot(
            slots, writeOp.getPodRef(), getRecordNameAsStringAttr(writeOp),
            writeOp.getValue().getType()
        );
      }
    }
  }
}

/// Return whether `op` directly uses a POD reference tracked for loop lifting.
static bool opUsesTrackedPodRefDirectly(Operation &op, ArrayRef<LoopPodSlot> slots) {
  return llvm::any_of(op.getOperands(), [&slots](Value operand) {
    return llvm::any_of(slots, [&operand](const LoopPodSlot &slot) {
      return slot.podRef == operand;
    });
  });
}

/// Return whether `op` contains nested POD accesses tracked for loop lifting.
static bool hasNestedTrackedPodAccess(Operation &op, ArrayRef<LoopPodSlot> slots) {
  return op
      .walk([&op, &slots](Operation *nestedOp) {
    if (nestedOp == &op) {
      return WalkResult::advance();
    }

    if (auto readOp = dyn_cast<ReadPodOp>(nestedOp)) {
      if (hasLoopSlot(slots, readOp.getPodRef(), getRecordNameAsStringAttr(readOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    if (auto writeOp = dyn_cast<WritePodOp>(nestedOp)) {
      if (hasLoopSlot(slots, writeOp.getPodRef(), getRecordNameAsStringAttr(writeOp))) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  }).wasInterrupted();
}

/// Return whether the loop body contains non-POD operations that still observe the tracked POD
/// references directly, which would make the simple lifting rewrite invalid.
static bool hasUnliftableLoopPodUses(Block &block, ArrayRef<LoopPodSlot> slots) {
  for (Operation &op : block) {
    if (isa<ReadPodOp, WritePodOp>(op)) {
      continue;
    }
    if (opUsesTrackedPodRefDirectly(op, slots) || hasNestedTrackedPodAccess(op, slots)) {
      return true;
    }
  }
  return false;
}

/// Rewrite loop-local POD reads and writes in an `scf.for` into extra iter args/results carrying
/// one SSA value per touched POD record.
static bool liftPodAccessesFromForLoop(scf::ForOp forOp) {
  Block &body = *forOp.getBody();
  SmallVector<LoopPodSlot> slots;
  collectDirectLoopPodSlots(body, forOp.getOperation(), slots);
  if (slots.empty() || hasUnliftableLoopPodUses(body, slots)) {
    return false;
  }

  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();

  SmallVector<Value> newInitArgs = llvm::to_vector(forOp.getInitArgs());
  builder.setInsertionPoint(forOp);
  for (const LoopPodSlot &slot : slots) {
    newInitArgs.push_back(genRead(loc, slot.podRef, slot.recordName, builder).getResult());
  }

  auto newFor = builder.create<scf::ForOp>(
      loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(), newInitArgs
  );
  newFor->setAttrs(forOp->getAttrs());

  Block &newBody = *newFor.getBody();
  dropTerminatorIfPresent(newBody);

  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newFor.getInductionVar());
  for (auto [idx, oldArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
    mapping.map(oldArg, newFor.getRegionIterArg(idx));
  }

  SmallVector<Value> slotValues = llvm::map_to_vector(
      llvm::seq<size_t>(0, slots.size()),
      [base = static_cast<size_t>(forOp.getNumRegionIterArgs()), &newFor](size_t idx) -> Value {
    return newFor.getRegionIterArg(llzk::checkedCast<unsigned>(base + idx));
  }
  );

  builder.setInsertionPointToEnd(&newBody);
  for (Operation &op : body) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(&op)) {
      auto yieldValues = llvm::map_to_vector(yieldOp.getOperands(), [&mapping](Value operand) {
        return mapping.lookupOrDefault(operand);
      });
      llvm::append_range(yieldValues, slotValues);
      builder.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);
      continue;
    }

    if (auto readOp = dyn_cast<ReadPodOp>(&op)) {
      if (std::optional<size_t> slotIdx =
              findLoopSlotIndex(slots, readOp.getPodRef(), getRecordNameAsStringAttr(readOp))) {
        mapping.map(readOp.getResult(), slotValues[*slotIdx]);
        continue;
      }
    }

    if (auto writeOp = dyn_cast<WritePodOp>(&op)) {
      if (std::optional<size_t> slotIdx =
              findLoopSlotIndex(slots, writeOp.getPodRef(), getRecordNameAsStringAttr(writeOp))) {
        slotValues[*slotIdx] = mapping.lookupOrDefault(writeOp.getValue());
        continue;
      }
    }

    builder.clone(op, mapping);
  }

  builder.setInsertionPointAfter(newFor);
  for (auto [idx, slot] : llvm::enumerate(slots)) {
    genWrite(
        loc, slot.podRef, slot.recordName, newFor.getResult(forOp.getNumResults() + idx), builder
    );
  }

  for (auto [oldResult, newResult] :
       llvm::zip_equal(forOp.getResults(), newFor.getResults().take_front(forOp.getNumResults()))) {
    oldResult.replaceAllUsesWith(newResult);
  }

  forOp.erase();
  return true;
}

/// Rewrite loop-local POD reads and writes in an `scf.while` into extra block arguments/results
/// carrying one SSA value per touched POD record.
static bool liftPodAccessesFromWhileLoop(scf::WhileOp whileOp) {
  Block &beforeBody = *whileOp.getBeforeBody();
  Block &afterBody = *whileOp.getAfterBody();

  SmallVector<LoopPodSlot> slots;
  collectDirectLoopPodSlots(beforeBody, whileOp.getOperation(), slots);
  collectDirectLoopPodSlots(afterBody, whileOp.getOperation(), slots);
  if (slots.empty() || hasUnliftableLoopPodUses(beforeBody, slots) ||
      hasUnliftableLoopPodUses(afterBody, slots)) {
    return false;
  }

  OpBuilder builder(whileOp);
  Location loc = whileOp.getLoc();

  SmallVector<Value> newInits = llvm::to_vector(whileOp.getInits());
  SmallVector<Type> newResultTypes = llvm::to_vector(whileOp.getResultTypes());
  builder.setInsertionPoint(whileOp);
  for (const LoopPodSlot &slot : slots) {
    newInits.push_back(genRead(loc, slot.podRef, slot.recordName, builder).getResult());
    newResultTypes.push_back(slot.type);
  }

  auto newWhile = builder.create<scf::WhileOp>(loc, newResultTypes, newInits, nullptr, nullptr);
  newWhile->setAttrs(whileOp->getAttrs());

  Block &newBeforeBody = *newWhile.getBeforeBody();
  Block &newAfterBody = *newWhile.getAfterBody();
  dropTerminatorIfPresent(newBeforeBody);
  dropTerminatorIfPresent(newAfterBody);

  IRMapping beforeMapping;
  for (auto [oldArg, newArg] : llvm::zip_equal(
           whileOp.getBeforeArguments(),
           newWhile.getBeforeArguments().take_front(whileOp.getBeforeArguments().size())
       )) {
    beforeMapping.map(oldArg, newArg);
  }

  SmallVector<Value> beforeSlotValues = llvm::map_to_vector(
      llvm::seq<size_t>(0, slots.size()),
      [base = whileOp.getBeforeArguments().size(), &newWhile](size_t idx) -> Value {
    return newWhile.getBeforeArguments()[llzk::checkedCast<unsigned>(base + idx)];
  }
  );

  builder.setInsertionPointToEnd(&newBeforeBody);
  for (Operation &op : beforeBody) {
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(&op)) {
      SmallVector<Value> conditionArgs =
          llvm::map_to_vector(conditionOp.getArgs(), [&beforeMapping](Value a) {
        return beforeMapping.lookupOrDefault(a);
      });
      llvm::append_range(conditionArgs, beforeSlotValues);
      builder.create<scf::ConditionOp>(
          conditionOp.getLoc(), beforeMapping.lookupOrDefault(conditionOp.getCondition()),
          conditionArgs
      );
      continue;
    }

    if (auto readOp = dyn_cast<ReadPodOp>(&op)) {
      if (std::optional<size_t> slotIdx =
              findLoopSlotIndex(slots, readOp.getPodRef(), getRecordNameAsStringAttr(readOp))) {
        beforeMapping.map(readOp.getResult(), beforeSlotValues[*slotIdx]);
        continue;
      }
    }

    if (auto writeOp = dyn_cast<WritePodOp>(&op)) {
      if (std::optional<size_t> slotIdx =
              findLoopSlotIndex(slots, writeOp.getPodRef(), getRecordNameAsStringAttr(writeOp))) {
        beforeSlotValues[*slotIdx] = beforeMapping.lookupOrDefault(writeOp.getValue());
        continue;
      }
    }

    builder.clone(op, beforeMapping);
  }

  IRMapping afterMapping;
  for (auto [oldArg, newArg] : llvm::zip_equal(
           whileOp.getAfterArguments(),
           newWhile.getAfterArguments().take_front(whileOp.getAfterArguments().size())
       )) {
    afterMapping.map(oldArg, newArg);
  }

  SmallVector<Value> afterSlotValues = llvm::map_to_vector(
      llvm::seq<size_t>(0, slots.size()),
      [base = whileOp.getAfterArguments().size(), &newWhile](size_t idx) -> Value {
    return newWhile.getAfterArguments()[llzk::checkedCast<unsigned>(base + idx)];
  }
  );

  builder.setInsertionPointToEnd(&newAfterBody);
  for (Operation &op : afterBody) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(&op)) {
      SmallVector<Value> yieldValues =
          llvm::map_to_vector(yieldOp.getOperands(), [&afterMapping](Value v) {
        return afterMapping.lookupOrDefault(v);
      });
      llvm::append_range(yieldValues, afterSlotValues);
      builder.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);
      continue;
    }

    if (auto readOp = dyn_cast<ReadPodOp>(&op)) {
      if (std::optional<size_t> slotIdx =
              findLoopSlotIndex(slots, readOp.getPodRef(), getRecordNameAsStringAttr(readOp))) {
        afterMapping.map(readOp.getResult(), afterSlotValues[*slotIdx]);
        continue;
      }
    }

    if (auto writeOp = dyn_cast<WritePodOp>(&op)) {
      if (std::optional<size_t> slotIdx =
              findLoopSlotIndex(slots, writeOp.getPodRef(), getRecordNameAsStringAttr(writeOp))) {
        afterSlotValues[*slotIdx] = afterMapping.lookupOrDefault(writeOp.getValue());
        continue;
      }
    }

    builder.clone(op, afterMapping);
  }

  builder.setInsertionPointAfter(newWhile);
  for (auto [idx, slot] : llvm::enumerate(slots)) {
    genWrite(
        loc, slot.podRef, slot.recordName, newWhile.getResult(whileOp.getNumResults() + idx),
        builder
    );
  }

  for (auto [oldResult, newResult] : llvm::zip_equal(
           whileOp.getResults(), newWhile.getResults().take_front(whileOp.getNumResults())
       )) {
    oldResult.replaceAllUsesWith(newResult);
  }

  whileOp.erase();
  return true;
}

/// Lift direct branch-local writes out of `scf.if` as yielded values, then write those values in
/// the parent block. Existing `scf.if` results are preserved as a prefix of the new result list,
/// which gives mem2reg parent-block pod writes instead of nested-region writes.
static bool liftPodWritesFromIfBlocks(scf::IfOp ifOp) {
  SmallVector<IfWriteSlot> slots;
  Block &thenBlock = *ifOp.thenBlock();
  Block *elseBlock = getElseBlockOrNull(ifOp);
  collectDirectWrites(&thenBlock, true, slots);
  collectDirectWrites(elseBlock, false, slots);
  if (slots.empty()) {
    return false;
  }

  llvm::erase_if(slots, [&](const IfWriteSlot &slot) {
    return isValueDefinedInside(ifOp, slot.podRef) ||
           !branchSlotCanBeLifted(&thenBlock, slot.podRef, slot.recordName) ||
           !branchSlotCanBeLifted(elseBlock, slot.podRef, slot.recordName);
  });
  if (slots.empty()) {
    return false;
  }

  OpBuilder builder(ifOp);
  for (IfWriteSlot &slot : slots) {
    if (slot.thenWrite && slot.elseWrite) {
      continue;
    }
    builder.setInsertionPoint(ifOp);
    slot.incomingValue = genRead(ifOp.getLoc(), slot.podRef, slot.recordName, builder);
  }

  SmallVector<Type> resultTypes = llvm::to_vector(ifOp.getResultTypes());
  llvm::append_range(resultTypes, llvm::map_range(slots, [](auto slot) { return slot.type; }));

  SmallVector<Value> originalThenYields;
  if (!ifOp.getResults().empty()) {
    scf::YieldOp thenYieldOp = getYieldOp(thenBlock);
    originalThenYields.append(thenYieldOp.getOperands().begin(), thenYieldOp.getOperands().end());
  }

  SmallVector<Value> originalElseYields;
  if (elseBlock && !ifOp.getResults().empty()) {
    scf::YieldOp elseYieldOp = getYieldOp(*elseBlock);
    originalElseYields.append(elseYieldOp.getOperands().begin(), elseYieldOp.getOperands().end());
  }

  builder.setInsertionPoint(ifOp);
  auto newIf = builder.create<scf::IfOp>(ifOp.getLoc(), resultTypes, ifOp.getCondition(), true);
  Block &newThenBlock = *newIf.thenBlock();
  Block &newElseBlock = *newIf.elseBlock();
  dropTerminatorIfPresent(newThenBlock);
  dropTerminatorIfPresent(newElseBlock);

  moveBranchWithoutLiftedWrites(&thenBlock, newThenBlock, slots);
  moveBranchWithoutLiftedWrites(elseBlock, newElseBlock, slots);
  appendYield(ifOp.getLoc(), newThenBlock, originalThenYields, slots, true, builder);
  appendYield(ifOp.getLoc(), newElseBlock, originalElseYields, slots, false, builder);

  builder.setInsertionPointAfter(newIf);
  unsigned originalResultCount = ifOp.getNumResults();
  for (auto [oldResult, newResult] :
       llvm::zip_equal(ifOp.getResults(), newIf.getResults().take_front(originalResultCount))) {
    oldResult.replaceAllUsesWith(newResult);
  }

  for (auto [idx, slot] : llvm::enumerate(slots)) {
    genWrite(
        ifOp.getLoc(), slot.podRef, slot.recordName, newIf.getResult(originalResultCount + idx),
        builder
    );
  }

  ifOp.erase();
  return true;
}

/// Repeatedly lift pod accesses out of supported SCF regions so SROA + mem2reg can eliminate the
/// remaining POD storage.
static void step3(ModuleOp modOp) {
  bool changed;
  do {
    changed = replaceIfReads(modOp);

    SmallVector<scf::IfOp> ifOps;
    modOp.walk([&ifOps](scf::IfOp ifOp) { ifOps.push_back(ifOp); });
    for (scf::IfOp ifOp : ifOps) {
      changed |= liftPodWritesFromIfBlocks(ifOp);
    }

    SmallVector<scf::ForOp> forOps;
    modOp.walk([&forOps](scf::ForOp forOp) { forOps.push_back(forOp); });
    for (scf::ForOp forOp : forOps) {
      changed |= liftPodAccessesFromForLoop(forOp);
    }

    SmallVector<scf::WhileOp> whileOps;
    modOp.walk([&whileOps](scf::WhileOp whileOp) { whileOps.push_back(whileOp); });
    for (scf::WhileOp whileOp : whileOps) {
      changed |= liftPodAccessesFromWhileLoop(whileOp);
    }

    changed |= foldIfCarriedPodReadAfterWrite(modOp);
  } while (changed);
}

/// Return a simple measure of how many POD allocation layers are represented by this type.
/// Non-POD records have weight zero; each POD layer contributes one plus its nested POD records.
static size_t podTypeScalarizationWeight(Type type) {
  auto podTy = dyn_cast<PodType>(type);
  if (!podTy) {
    return 0;
  }

  size_t weight = 1;
  for (RecordAttr record : podTy.getRecords()) {
    weight += podTypeScalarizationWeight(record.getType());
  }
  return weight;
}

/// Return the total remaining POD allocation work in the module. This is used to rerun
/// SROA+mem2reg while recursive POD layers are being exposed, and to stop if a pass round cannot
/// make progress.
static size_t podAllocScalarizationWeight(ModuleOp modOp) {
  size_t weight = 0;
  modOp.walk([&weight](NewPodOp newPodOp) {
    weight += podTypeScalarizationWeight(newPodOp.getType());
  });
  return weight;
}

/// Pass driver for the full POD-to-scalar lowering pipeline described above.
class PodToScalarPass : public llzk::pod::impl::PodToScalarPassBase<PodToScalarPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (failed(step0(module))) {
      return signalPassFailure();
    }
    LLVM_DEBUG({
      llvm::dbgs() << "After step 0:\n";
      module.dump();
    });

    {
      // This is divided into 2 steps to simplify the implementation for member-related ops. The
      // issue is that the conversions for member read/write expect the mapping of record name to
      // member name+type to already be populated for the referenced member (although this could be
      // computed on demand if desired but it complicates the implementation a bit).
      SymbolTableCollection symTables;
      MemberReplacementMap memberRepMap;
      if (failed(step1(module, symTables, memberRepMap))) {
        return signalPassFailure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "After step 1:\n";
        module.dump();
      });

      if (failed(step2(module, symTables, memberRepMap))) {
        return signalPassFailure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "After step 2:\n";
        module.dump();
      });
    }

    step3(module);
    LLVM_DEBUG({
      llvm::dbgs() << "After step 3:\n";
      module.dump();
    });

    // 1. Use SROA (Destructurable* interfaces) to split each pod with `N` records into `N` pods
    // with 1 record each. This is necessary because the mem2reg pass cannot deal with splitting
    // up memory, i.e., it can only convert scalar memory access into SSA values.
    // 2. The mem2reg pass converts the size 1 pod allocations and accesses into SSA values.
    OpPassManager scalarizePM(ModuleOp::getOperationName());
    scalarizePM.addPass(createSpecializedSROAPass<NewPodOp>());
    scalarizePM.addPass(createSpecializedMem2RegPass<NewPodOp>());

    // Cleanup SSA values made dead by the transformations
    OpPassManager cleanupPM(ModuleOp::getOperationName());
    cleanupPM.addPass(createRemoveDeadValuesPass());

    size_t podAllocWeight = podAllocScalarizationWeight(module);
    while (podAllocWeight != 0) {
      if (failed(runPipeline(scalarizePM, module))) {
        signalPassFailure();
        return;
      }

      // SROA+mem2reg can expose `scf.if`-carried POD values that become redundant after a
      // same-record write from another `scf.if` result. Fold those reads and clean up before
      // checking convergence.
      bool foldedIfCarriedRead = foldIfCarriedPodReadAfterWrite(module);
      if (failed(runPipeline(cleanupPM, module))) {
        signalPassFailure();
        return;
      }

      // Nested PODs can become visible only after an outer single-record POD has been promoted,
      // and SROA can transiently increase allocation count while splitting aggregates. Keep
      // iterating until the allocation-weight heuristic reaches a fixed point.
      size_t nextPodAllocWeight = podAllocScalarizationWeight(module);
      if (!foldedIfCarriedRead && nextPodAllocWeight == podAllocWeight) {
        break;
      }
      podAllocWeight = nextPodAllocWeight;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "After SROA+Mem2Reg pipeline:\n";
      module.dump();
    });
  }
};

} // namespace

/// Create the pass that rewrites eligible POD storage into scalar SSA values.
std::unique_ptr<Pass> llzk::pod::createPodToScalarPass() {
  return std::make_unique<PodToScalarPass>();
};
