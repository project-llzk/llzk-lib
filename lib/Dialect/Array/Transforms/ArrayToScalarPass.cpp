//===-- ArrayToScalarPass.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-array-to-scalar` pass.
///
/// The steps of this transformation are as follows:
///
/// 0. Scan to find `llzk.nondet` ops that allocate uninitialized arrays and replace them with
///    an equivalent `array.new`
///
/// 1. Run a dialect conversion that replaces `ArrayType` struct members with `N` scalar members.
///
/// 2. Run a dialect conversion that does the following:
///
///    - Replace `MemberReadOp` and `MemberWriteOp` targeting the members that were split in step 1
///      so they instead perform scalar reads and writes from the new members. The transformation is
///      local to the current op. Therefore, when replacing the `MemberReadOp` a new array is
///      created locally and all uses of the `MemberReadOp` are replaced with the new array Value,
///      then each scalar member read is followed by scalar write into the new array. Similarly,
///      when replacing a `MemberWriteOp`, each element in the array operand needs a scalar read
///      from the array followed by a scalar write to the new member. Making only local changes
///      keeps this step simple and later steps will optimize.
///
///    - Replace `ArrayLengthOp` with the constant size of the selected dimension.
///
///    - Remove element initialization from `CreateArrayOp` and instead insert a list of
///      `WriteArrayOp` immediately following.
///
///    - Desugar `InsertArrayOp` and `ExtractArrayOp` into their element-wise scalar reads/writes.
///
///    - Split arrays to scalars in `FuncDefOp`, `CallOp`, and `ReturnOp` and insert the necessary
///      create/read/write ops so the changes are as local as possible (just as described for
///      `MemberReadOp` and `MemberWriteOp`)
///
/// 3. Replace branch-local reads (in `scf.if`) with the value written by a same-index write op that
///    dominates the parent `scf.if` (because the passes below cannot handle that case).
///
/// 4. Run MLIR "sroa" pass to split each array with linear size `N` into `N` arrays of size 1
///    (to prepare for "mem2reg" pass because its API cannot deal with splitting up memory).
///
/// 5. Run MLIR "mem2reg" pass to convert all of the size 1 array allocation and access into SSA
///    values. This pass also runs several standard optimizations so the final result is condensed.
///
/// Note: This transformation imposes a "last write wins" semantics on array elements. If
/// different/configurable semantics are added in the future, some additional transformation would
/// be necessary before/during this pass so that multiple writes to the same index can be handled
/// properly while they still exist.
///
/// Note: This transformation will introduce a `nondet` op when there exists a read from an array
/// index that was not earlier written to.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Array/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Array/Util/ArrayTypeHelper.h"
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
#include "llzk/Dialect/Polymorphic/IR/Dialect.h"
#include "llzk/Dialect/RAM/IR/Dialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Verif/IR/Dialect.h"
#include "llzk/Dialect/Verif/IR/Ops.h"
#include "llzk/Transforms/LLZKConversionUtils.h"
#include "llzk/Transforms/SpecializedMemoryPasses.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/Concepts.h"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/Debug.h>

#include <optional>

// Include the generated base pass class definitions.
namespace llzk::array {
#define GEN_PASS_DEF_ARRAYTOSCALARPASS
#include "llzk/Dialect/Array/Transforms/TransformationPasses.h.inc"
} // namespace llzk::array

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::function;
using namespace llzk::verif;

#define DEBUG_TYPE "llzk-array-to-scalar"

namespace {

/// If the given ArrayType can be split into scalars, return it, otherwise nullptr.
inline ArrayType splittableArray(ArrayType at) { return at.hasStaticShape() ? at : nullptr; }

/// If the given Type is an ArrayType that can be split into scalars, return it, otherwise nullptr.
inline ArrayType splittableArray(Type t) {
  if (ArrayType at = dyn_cast<ArrayType>(t)) {
    return splittableArray(at);
  } else {
    return nullptr;
  }
}

/// Return `true` iff the given range contains any top-level ArrayType that can be split into
/// scalars.
inline bool containsSplittableArrayType(ArrayRef<Type> types) {
  for (Type t : types) {
    if (splittableArray(t)) {
      return true;
    }
  }
  return false;
}

/// Return `true` iff the given range contains any ArrayType that can be split into scalars.
template <typename T> bool containsSplittableArrayType(ValueTypeRange<T> types) {
  for (Type t : types) {
    if (splittableArray(t)) {
      return true;
    }
  }
  return false;
}

/// If the given Type is an ArrayType that can be split into scalars, append `collect` with all of
/// the scalar types that result from splitting the ArrayType. Otherwise, just push the `Type`.
size_t splitArrayTypeTo(Type t, SmallVector<Type> &collect) {
  if (ArrayType at = splittableArray(t)) {
    size_t size = llzk::checkedCast<size_t>(at.getNumElements());
    collect.append(size, at.getElementType());
    return size;
  } else {
    collect.push_back(t);
    return 1;
  }
}

/// For each Type in the given input collection, call `splitArrayTypeTo(Type,...)`.
template <typename TypeCollection>
inline void splitArrayTypeTo(
    TypeCollection types, SmallVector<Type> &collect, SmallVector<size_t> *originalIdxToSize
) {
  for (Type t : types) {
    size_t count = splitArrayTypeTo(t, collect);
    if (originalIdxToSize) {
      originalIdxToSize->push_back(count);
    }
  }
}

/// Return a list such that each scalar Type is directly added to the list but for each splittable
/// ArrayType, the proper number of scalar element types are added instead.
template <typename TypeCollection>
inline SmallVector<Type>
splitArrayType(TypeCollection types, SmallVector<size_t> *originalIdxToSize = nullptr) {
  SmallVector<Type> collect;
  splitArrayTypeTo(types, collect, originalIdxToSize);
  return collect;
}

/// Generate `arith::ConstantOp` at the current position of the `rewriter` for each int attribute in
/// the ArrayAttr.
SmallVector<Value> genIndexConstants(ArrayAttr index, Location loc, RewriterBase &rewriter) {
  SmallVector<Value> operands;
  for (Attribute a : index) {
    // ASSERT: Attributes are index constants, created by ArrayType::getSubelementIndices().
    IntegerAttr ia = llvm::dyn_cast<IntegerAttr>(a);
    assert(ia && ia.getType().isIndex());
    operands.push_back(rewriter.create<arith::ConstantOp>(loc, ia));
  }
  return operands;
}

/// Create an `array.write` for one scalar element of `baseArrayOp`.
inline WriteArrayOp
genWrite(Location loc, Value baseArrayOp, ArrayAttr index, Value init, RewriterBase &rewriter) {
  SmallVector<Value> readOperands = genIndexConstants(index, loc, rewriter);
  return rewriter.create<WriteArrayOp>(loc, baseArrayOp, ValueRange(readOperands), init);
}

/// Return the suffix for one split scalar element of an array, using its multidimensional index.
static std::string formatSplitArrayIndexSuffix(ArrayAttr index) {
  std::string suffix;
  llvm::raw_string_ostream os(suffix);
  for (Attribute attr : index) {
    os << '[';
    attr.print(os, true);
    os << ']';
  }
  return suffix;
}

/// Return the suffixes to append to a function arg/result name when splitting the given type.
static SmallVector<std::string> getSplitArrayIndexSuffixes(Type type) {
  SmallVector<std::string> suffixes;
  if (ArrayType at = splittableArray(type)) {
    std::optional<SmallVector<ArrayAttr>> indices = at.getSubelementIndices();
    assert(indices.has_value() && "static-shape arrays must provide subelement indices");
    suffixes.reserve(indices->size());
    for (ArrayAttr index : *indices) {
      suffixes.push_back(formatSplitArrayIndexSuffix(index));
    }
  }
  return suffixes;
}

/// Rebuild a call with split scalar results, then reconstruct array-typed results locally.
CallOp newCallOpWithSplitResults(
    CallOp oldCall, CallOp::Adaptor adaptor, ConversionPatternRewriter &rewriter
) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(oldCall);

  Operation::result_range oldResults = oldCall.getResults();
  CallOp newCall = createCallPreservingInstantiationOperands(
      oldCall.getLoc(), splitArrayType(oldResults.getTypes()), oldCall, adaptor.getMapOperands(),
      adaptor.getArgOperands(), rewriter
  );

  auto newResults = newCall.getResults().begin();
  for (Value oldVal : oldResults) {
    if (ArrayType at = splittableArray(oldVal.getType())) {
      Location loc = oldVal.getLoc();
      // Generate `CreateArrayOp` and replace uses of the result with it.
      auto newArray = rewriter.create<CreateArrayOp>(loc, at);
      rewriter.replaceAllUsesWith(oldVal, newArray);

      // For all indices in the ArrayType (i.e., the element count), write the next
      // result from the new CallOp to the new array.
      std::optional<SmallVector<ArrayAttr>> allIndices = at.getSubelementIndices();
      assert(allIndices); // follows from legal() check
      assert(std::cmp_equal(allIndices->size(), at.getNumElements()));
      for (ArrayAttr subIdx : allIndices.value()) {
        genWrite(loc, newArray, subIdx, *newResults, rewriter);
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

/// Create an `array.read` for one scalar element of `baseArrayOp`.
inline ReadArrayOp
genRead(Location loc, Value baseArrayOp, ArrayAttr index, ConversionPatternRewriter &rewriter) {
  SmallVector<Value> readOperands = genIndexConstants(index, loc, rewriter);
  return rewriter.create<ReadArrayOp>(loc, baseArrayOp, ValueRange(readOperands));
}

/// If the operand has ArrayType, add N reads from the array to `newOperands`; otherwise add the
/// original operand unchanged.
void processInputOperand(
    Location loc, Value operand, SmallVector<Value> &newOperands,
    ConversionPatternRewriter &rewriter
) {
  if (ArrayType at = splittableArray(operand.getType())) {
    std::optional<SmallVector<ArrayAttr>> indices = at.getSubelementIndices();
    assert(indices.has_value() && "passed earlier hasStaticShape() check");
    for (ArrayAttr index : indices.value()) {
      newOperands.push_back(genRead(loc, operand, index, rewriter));
    }
  } else {
    newOperands.push_back(operand);
  }
}

/// Replace each array operand with its scalar element reads in `outputOpRef`.
void processInputOperands(
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

template <typename FunctionLikeOp>
class SplitArrayInFunctionLikeOpImpl : public FunctionTypeConverter {
  SmallVector<size_t> originalInputIdxToSize, originalResultIdxToSize;
  SplitFunctionNameInfo inputNameInfo;
  SplitFunctionNameInfo resultNameInfo;

  static constexpr bool supportsResultAttrs() {
    return requires(FunctionLikeOp op, ArrayAttr attrs) {
      op.getResAttrsAttr();
      op.setResAttrsAttr(attrs);
    };
  }

protected:
  SmallVector<Type> convertInputs(ArrayRef<Type> origTypes) override {
    return splitArrayType(origTypes, &originalInputIdxToSize);
  }
  SmallVector<Type> convertResults(ArrayRef<Type> origTypes) override {
    return splitArrayType(origTypes, &originalResultIdxToSize);
  }
  ArrayAttr convertInputAttrs(ArrayAttr origAttrs, SmallVector<Type> newTypes) override {
    return replicateFunctionNameAttrsAsNeeded(
        origAttrs, originalInputIdxToSize, newTypes, ARG_NAME_ATTR_NAME,
        inputNameInfo.originalNames, inputNameInfo.existingNames, inputNameInfo.splitNameSuffixes
    );
  }
  ArrayAttr convertResultAttrs(ArrayAttr origAttrs, SmallVector<Type> newTypes) override {
    if constexpr (!supportsResultAttrs()) {
      return nullptr;
    }
    return replicateFunctionNameAttrsAsNeeded(
        origAttrs, originalResultIdxToSize, newTypes, RES_NAME_ATTR_NAME,
        resultNameInfo.originalNames, resultNameInfo.existingNames, resultNameInfo.splitNameSuffixes
    );
  }

  void processBlockArgs(Block &entryBlock, RewriterBase &rewriter) override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&entryBlock);

    for (unsigned i = 0; i < entryBlock.getNumArguments();) {
      Value oldV = entryBlock.getArgument(i);
      if (ArrayType at = splittableArray(oldV.getType())) {
        Location loc = oldV.getLoc();
        auto newArray = rewriter.create<CreateArrayOp>(loc, at);
        rewriter.replaceAllUsesWith(oldV, newArray);
        entryBlock.eraseArgument(i);
        std::optional<SmallVector<ArrayAttr>> allIndices = at.getSubelementIndices();
        assert(allIndices && "static-shape arrays must provide subelement indices");
        assert(std::cmp_equal(allIndices->size(), at.getNumElements()));
        for (ArrayAttr subIdx : *allIndices) {
          BlockArgument newArg = entryBlock.insertArgument(i, at.getElementType(), loc);
          genWrite(loc, newArray, subIdx, newArg, rewriter);
          ++i;
        }
      } else {
        ++i;
      }
    }
  }

public:
  explicit SplitArrayInFunctionLikeOpImpl(FunctionLikeOp op) {
    inputNameInfo = collectSplitFunctionNameInfo(op.getArgumentTypes(), [&](unsigned i) {
      return op.getArgNameAttr(i);
    }, getSplitArrayIndexSuffixes);
    if constexpr (supportsResultAttrs()) {
      ArrayAttr resultAttrs = op.getAllResultAttrs();
      resultNameInfo = collectSplitFunctionNameInfo(op.getResultTypes(), [&](unsigned i) {
        return getAttrAtIndexWithName(resultAttrs, i, RES_NAME_ATTR_NAME);
      }, getSplitArrayIndexSuffixes);
    }
  }
};

namespace {

enum Direction : std::uint8_t {
  /// Copying a smaller array into a larger one, i.e., `InsertArrayOp`
  SMALL_TO_LARGE,
  /// Copying a larger array into a smaller one, i.e., `ExtractArrayOp`
  LARGE_TO_SMALL,
};

/// Common implementation for handling `InsertArrayOp` and `ExtractArrayOp`. For all indices in the
/// given ArrayType, perform writes from one array to the other, in the specified Direction.
template <Direction dir>
inline void rewriteImpl(
    ArrayAccessOpInterface op, ArrayType smallType, Value smallArr, Value largeArr,
    ConversionPatternRewriter &rewriter
) {
  assert(smallType); // follows from legal() check
  Location loc = op.getLoc();
  MLIRContext *ctx = op.getContext();

  ArrayAttr indexAsAttr = op.indexOperandsToAttributeArray();
  assert(indexAsAttr); // follows from legal() check

  // For all indices in the ArrayType (i.e., the element count), read from one array into the other
  // (depending on direction flag).
  std::optional<SmallVector<ArrayAttr>> subIndices = smallType.getSubelementIndices();
  assert(subIndices); // follows from legal() check
  assert(std::cmp_equal(subIndices->size(), smallType.getNumElements()));
  for (ArrayAttr indexingTail : subIndices.value()) {
    SmallVector<Attribute> joined;
    joined.append(indexAsAttr.begin(), indexAsAttr.end());
    joined.append(indexingTail.begin(), indexingTail.end());
    ArrayAttr fullIndex = ArrayAttr::get(ctx, joined);

    if constexpr (dir == Direction::SMALL_TO_LARGE) {
      auto init = genRead(loc, smallArr, indexingTail, rewriter);
      genWrite(loc, largeArr, fullIndex, init, rewriter);
    } else if constexpr (dir == Direction::LARGE_TO_SMALL) {
      auto init = genRead(loc, largeArr, fullIndex, rewriter);
      genWrite(loc, smallArr, indexingTail, init, rewriter);
    }
  }
}

} // namespace

/// Rewrite `array.insert` of a splittable subarray into element-wise scalar writes.
class SplitInsertArrayOp : public OpConversionPattern<InsertArrayOp> {
public:
  using OpConversionPattern<InsertArrayOp>::OpConversionPattern;

  static bool legal(InsertArrayOp op) {
    return !containsSplittableArrayType(op.getRvalue().getType());
  }

  LogicalResult match(InsertArrayOp op) const override { return failure(legal(op)); }

  void
  rewrite(InsertArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    ArrayType at = splittableArray(op.getRvalue().getType());
    rewriteImpl<SMALL_TO_LARGE>(
        llvm::cast<ArrayAccessOpInterface>(op.getOperation()), at, adaptor.getRvalue(),
        adaptor.getArrRef(), rewriter
    );
    rewriter.eraseOp(op);
  }
};

/// Rewrite `array.extract` of a splittable subarray into element-wise scalar reads.
class SplitExtractArrayOp : public OpConversionPattern<ExtractArrayOp> {
public:
  using OpConversionPattern<ExtractArrayOp>::OpConversionPattern;

  static bool legal(ExtractArrayOp op) {
    return !containsSplittableArrayType(op.getResult().getType());
  }

  LogicalResult match(ExtractArrayOp op) const override { return failure(legal(op)); }

  void rewrite(
      ExtractArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    ArrayType at = splittableArray(op.getResult().getType());
    // Generate `CreateArrayOp` in place of the current op.
    auto newArray = rewriter.replaceOpWithNewOp<CreateArrayOp>(op, at);
    rewriteImpl<LARGE_TO_SMALL>(
        llvm::cast<ArrayAccessOpInterface>(op.getOperation()), at, newArray, adaptor.getArrRef(),
        rewriter
    );
  }
};

/// Split inline `array.new` element initializers into explicit `array.write` operations.
class SplitInitFromCreateArrayOp : public OpConversionPattern<CreateArrayOp> {
public:
  using OpConversionPattern<CreateArrayOp>::OpConversionPattern;

  static bool legal(CreateArrayOp op) { return op.getElements().empty(); }

  LogicalResult match(CreateArrayOp op) const override { return failure(legal(op)); }

  void
  rewrite(CreateArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // Remove elements from `op`
    rewriter.modifyOpInPlace(op, [&op]() { op.getElementsMutable().clear(); });
    // Generate an individual write for each initialization element
    rewriter.setInsertionPointAfter(op);
    Location loc = op.getLoc();
    ArrayIndexGen idxGen = ArrayIndexGen::from(op.getType());
    for (auto [i, init] : llvm::enumerate(adaptor.getElements())) {
      // Convert the linear index 'i' into a multi-dim index
      std::optional<SmallVector<Value>> multiDimIdxVals =
          idxGen.delinearize(llzk::checkedCast<int64_t>(i), loc, rewriter);
      // ASSERT: CreateArrayOp verifier ensures the number of elements provided matches the full
      // linear array size so delinearization of `i` will not fail.
      assert(multiDimIdxVals.has_value());
      // Create the write
      rewriter.create<WriteArrayOp>(loc, op.getResult(), ValueRange(*multiDimIdxVals), init);
    }
  }
};

/// Rewrite array-typed function signatures to pass one scalar per array element instead.
class SplitArrayInFuncDefOp : public OpConversionPattern<FuncDefOp> {
public:
  using OpConversionPattern<FuncDefOp>::OpConversionPattern;

  inline static bool legal(FuncDefOp op) {
    return !containsSplittableArrayType(op.getArgumentTypes()) &&
           !containsSplittableArrayType(op.getResultTypes());
  }

  LogicalResult match(FuncDefOp op) const override { return failure(legal(op)); }

  void rewrite(FuncDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    SplitArrayInFunctionLikeOpImpl<FuncDefOp>(op).convert(op, rewriter);
  }
};

/// Rewrite array-typed contract signatures to pass one scalar per array element instead.
class SplitArrayInContractOp : public OpConversionPattern<ContractOp> {
public:
  using OpConversionPattern<ContractOp>::OpConversionPattern;

  inline static bool legal(ContractOp op) {
    return !containsSplittableArrayType(op.getArgumentTypes()) &&
           !containsSplittableArrayType(op.getResultTypes());
  }

  LogicalResult match(ContractOp op) const override { return failure(legal(op)); }

  void rewrite(ContractOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    SplitArrayInFunctionLikeOpImpl<ContractOp>(op).convert(op, rewriter);
  }
};

/// Rewrite `function.return` to flatten any array operands into scalar element values.
class SplitArrayInReturnOp : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  inline static bool legal(ReturnOp op) {
    return !containsSplittableArrayType(op.getOperands().getTypes());
  }

  LogicalResult match(ReturnOp op) const override { return failure(legal(op)); }

  void rewrite(ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    processInputOperands(adaptor.getOperands(), op.getOperandsMutable(), op, rewriter);
  }
};

/// Rewrite calls whose arguments or results contain arrays to use flattened scalar signatures.
class SplitArrayInCallOp : public OpConversionPattern<CallOp> {
public:
  using OpConversionPattern<CallOp>::OpConversionPattern;

  inline static bool legal(CallOp op) {
    return !containsSplittableArrayType(op.getArgOperands().getTypes()) &&
           !containsSplittableArrayType(op.getResultTypes());
  }

  LogicalResult match(CallOp op) const override { return failure(legal(op)); }

  void rewrite(CallOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // Create new CallOp with split results first so, then process its inputs to split types
    CallOp newCall = newCallOpWithSplitResults(op, adaptor, rewriter);
    rewriter.setInsertionPoint(newCall);
    processInputOperands(
        newCall.getArgOperands(), newCall.getArgOperandsMutable(), newCall, rewriter
    );
  }
};

/// Rewrite included-contract calls whose arguments contain arrays to use flattened scalar
/// signatures.
class SplitArrayInIncludeOp : public OpConversionPattern<IncludeOp> {
public:
  using OpConversionPattern<IncludeOp>::OpConversionPattern;

  inline static bool legal(IncludeOp op) {
    return !containsSplittableArrayType(op.getArgOperands().getTypes());
  }

  LogicalResult match(IncludeOp op) const override { return failure(legal(op)); }

  void
  rewrite(IncludeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    IncludeOp newInclude = createIncludePreservingInstantiationOperands(
        op.getLoc(), op, adaptor.getMapOperands(), adaptor.getArgOperands(), rewriter
    );
    rewriter.setInsertionPoint(newInclude);
    processInputOperands(
        newInclude.getArgOperands(), newInclude.getArgOperandsMutable(), newInclude, rewriter
    );
    rewriter.eraseOp(op);
  }
};

/// Replace `array.length` with a constant when the selected dimension size is statically known.
class ReplaceKnownArrayLengthOp : public OpConversionPattern<ArrayLengthOp> {
public:
  using OpConversionPattern<ArrayLengthOp>::OpConversionPattern;

  /// If 'dimIdx' is constant and that dimension of the ArrayType has static size, return it.
  static std::optional<llvm::APInt> getDimSizeIfKnown(Value dimIdx, ArrayType baseArrType) {
    if (splittableArray(baseArrType)) {
      llvm::APInt idxAP;
      if (mlir::matchPattern(dimIdx, mlir::m_ConstantInt(&idxAP))) {
        std::optional<int64_t> signedIdx = idxAP.trySExtValue();
        if (!signedIdx || *signedIdx < 0) {
          return std::nullopt;
        }
        size_t idx = llzk::checkedCast<size_t>(*signedIdx);
        ArrayRef<Attribute> dimSizes = baseArrType.getDimensionSizes();
        if (idx >= dimSizes.size()) {
          return std::nullopt;
        }
        Attribute dimSizeAttr = dimSizes[idx];
        if (mlir::matchPattern(dimSizeAttr, mlir::m_ConstantInt(&idxAP))) {
          return idxAP;
        }
      }
    }
    return std::nullopt;
  }

  inline static bool legal(ArrayLengthOp op) {
    // rewrite() can only work with constant dim size, i.e., must consider it legal otherwise
    return !getDimSizeIfKnown(op.getDim(), op.getArrRefType()).has_value();
  }

  LogicalResult match(ArrayLengthOp op) const override { return failure(legal(op)); }

  void
  rewrite(ArrayLengthOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    ArrayType arrTy = dyn_cast<ArrayType>(adaptor.getArrRef().getType());
    assert(arrTy); // must have array type per ODS spec of ArrayLengthOp
    std::optional<llvm::APInt> len = getDimSizeIfKnown(adaptor.getDim(), arrTy);
    assert(len.has_value()); // follows from legal() check
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, llzk::fromAPInt(len.value()));
  }
};

/// member name and type
using MemberInfo = std::pair<StringAttr, Type>;
/// original multi-dimensional index -> scalar member info
using LocalMemberReplacementMap = DenseMap<ArrayAttr, MemberInfo>;
/// struct -> original array-type member name -> LocalMemberReplacementMap
using MemberReplacementMap = DenseMap<StructDefOp, DenseMap<StringAttr, LocalMemberReplacementMap>>;

/// Split an array-typed struct member definition into one scalar member per array element.
class SplitArrayInMemberDefOp : public OpConversionPattern<MemberDefOp> {
  SymbolTableCollection &tables;
  MemberReplacementMap &repMapRef;

public:
  SplitArrayInMemberDefOp(
      MLIRContext *ctx, SymbolTableCollection &symTables, MemberReplacementMap &memberRepMap
  )
      : OpConversionPattern<MemberDefOp>(ctx), tables(symTables), repMapRef(memberRepMap) {}

  inline static bool legal(MemberDefOp op) { return !containsSplittableArrayType(op.getType()); }

  LogicalResult match(MemberDefOp op) const override { return failure(legal(op)); }

  void rewrite(MemberDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    StructDefOp inStruct = op->getParentOfType<StructDefOp>();
    assert(inStruct);
    LocalMemberReplacementMap &localRepMapRef = repMapRef[inStruct][op.getSymNameAttr()];

    ArrayType arrTy = dyn_cast<ArrayType>(op.getType());
    assert(arrTy); // follows from legal() check
    auto subIdxs = arrTy.getSubelementIndices();
    assert(subIdxs.has_value());
    Type elemTy = arrTy.getElementType();

    SymbolTable &structSymbolTable = tables.getSymbolTable(inStruct);
    for (ArrayAttr idx : subIdxs.value()) {
      // Create scalar version of the member
      MemberDefOp newMember = rewriter.create<MemberDefOp>(
          op.getLoc(), op.getSymNameAttr(), elemTy, op.getSignal(), op.getColumn()
      );
      newMember.setPublicAttr(op.hasPublicAttr());
      // Use SymbolTable to give it a unique name and store to the replacement map
      localRepMapRef[idx] = std::make_pair(structSymbolTable.insert(newMember), elemTy);
    }
    rewriter.eraseOp(op);
  }
};

/// Rewrite a write to an array-typed struct member into writes to the corresponding scalar leaves.
class SplitArrayInMemberWriteOp : public SplitAggregateInMemberRefOp<
                                      SplitArrayInMemberWriteOp, MemberWriteOp, void *, ArrayAttr> {
public:
  using SplitAggregateInMemberRefOp<
      SplitArrayInMemberWriteOp, MemberWriteOp, void *, ArrayAttr>::SplitAggregateInMemberRefOp;

  static bool legal(MemberWriteOp op) {
    return !containsSplittableArrayType(op.getVal().getType());
  }

  static void *genHeader(MemberWriteOp, ConversionPatternRewriter &) { return nullptr; }

  static void forId(
      Location loc, void *, ArrayAttr idx, MemberInfo newMember, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter
  ) {
    ReadArrayOp scalarRead = genRead(loc, adaptor.getVal(), idx, rewriter);
    rewriter.create<MemberWriteOp>(
        loc, adaptor.getComponent(), FlatSymbolRefAttr::get(newMember.first), scalarRead
    );
  }
};

/// Rewrite a read from an array-typed struct member into reads from the corresponding scalar
/// leaves followed by local array reconstruction.
class SplitArrayInMemberReadOp
    : public SplitAggregateInMemberRefOp<
          SplitArrayInMemberReadOp, MemberReadOp, CreateArrayOp, ArrayAttr> {
public:
  using SplitAggregateInMemberRefOp<
      SplitArrayInMemberReadOp, MemberReadOp, CreateArrayOp,
      ArrayAttr>::SplitAggregateInMemberRefOp;

  static bool legal(MemberReadOp op) {
    return !containsSplittableArrayType(op.getResult().getType());
  }

  static CreateArrayOp genHeader(MemberReadOp op, ConversionPatternRewriter &rewriter) {
    CreateArrayOp newArray =
        rewriter.create<CreateArrayOp>(op.getLoc(), llvm::cast<ArrayType>(op.getType()));
    rewriter.replaceAllUsesWith(op, newArray);
    return newArray;
  }

  static void forId(
      Location loc, CreateArrayOp newArray, ArrayAttr idx, MemberInfo newMember, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter
  ) {
    MemberReadOp scalarRead = rewriter.create<MemberReadOp>(
        loc, newMember.second, adaptor.getComponent(), newMember.first
    );
    genWrite(loc, newArray, idx, scalarRead, rewriter);
  }
};

/// Register the dialects and operations that remain legal across the conversion-based stages.
static void baseTargetSetup(ConversionTarget &target) {
  target.addLegalDialect<
      LLZKDialect, array::ArrayDialect, boolean::BoolDialect, cast::CastDialect,
      constrain::ConstrainDialect, component::StructDialect, felt::FeltDialect,
      function::FunctionDialect, global::GlobalDialect, include::IncludeDialect, pod::PODDialect,
      polymorphic::PolymorphicDialect, ram::RAMDialect, string::StringDialect, verif::VerifDialect,
      arith::ArithDialect, scf::SCFDialect>();
  target.addLegalOp<ModuleOp>();
}

/// Rewrite array-typed `llzk.nondet` allocations into explicit `array.new` allocations.
class NondetToNewArray : public OpConversionPattern<NonDetOp> {
  using OpConversionPattern<NonDetOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      NonDetOp nondetOp, OpAdaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (auto at = dyn_cast<ArrayType>(nondetOp.getType())) {
      rewriter.replaceOpWithNewOp<CreateArrayOp>(nondetOp, at);
      return success();
    }
    return failure();
  }
};

/// Prepare the module by replacing `llzk.nondet` array allocation ops with `array.new`.
static LogicalResult step0(ModuleOp modOp) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns {ctx};
  patterns.add<NondetToNewArray>(ctx);
  ConversionTarget target {*ctx};

  baseTargetSetup(target);
  target.addDynamicallyLegalOp<NonDetOp>([](NonDetOp op) { return !isa<ArrayType>(op.getType()); });

  return applyFullConversion(modOp, target, std::move(patterns));
}

/// Replace `ArrayType` struct members with `N` scalar members.
static LogicalResult
step1(ModuleOp modOp, SymbolTableCollection &symTables, MemberReplacementMap &memberRepMap) {
  MLIRContext *ctx = modOp.getContext();

  RewritePatternSet patterns(ctx);

  patterns.add<SplitArrayInMemberDefOp>(ctx, symTables, memberRepMap);

  ConversionTarget target(*ctx);
  baseTargetSetup(target);
  target.addDynamicallyLegalOp<MemberDefOp>(SplitArrayInMemberDefOp::legal);

  LLVM_DEBUG(llvm::dbgs() << "Begin step 1: split array-type members\n";);
  return applyFullConversion(modOp, target, std::move(patterns));
}

/// Special handling to split arrays in struct member refs and function signatures, desugar ranged
/// array access ops to scalar access ops, and replace `ArrayLengthOp` with the known size.
static LogicalResult
step2(ModuleOp modOp, SymbolTableCollection &symTables, const MemberReplacementMap &memberRepMap) {
  MLIRContext *ctx = modOp.getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<
      // clang-format off
      SplitInitFromCreateArrayOp,
      SplitInsertArrayOp,
      SplitExtractArrayOp,
      SplitArrayInFuncDefOp,
      SplitArrayInContractOp,
      SplitArrayInReturnOp,
      SplitArrayInCallOp,
      SplitArrayInIncludeOp,
      ReplaceKnownArrayLengthOp
      // clang-format on
      >(ctx);

  patterns.add<
      // clang-format off
      SplitArrayInMemberWriteOp,
      SplitArrayInMemberReadOp
      // clang-format on
      >(ctx, symTables, memberRepMap);

  ConversionTarget target(*ctx);
  baseTargetSetup(target);
  target.addDynamicallyLegalOp<CreateArrayOp>(SplitInitFromCreateArrayOp::legal);
  target.addDynamicallyLegalOp<InsertArrayOp>(SplitInsertArrayOp::legal);
  target.addDynamicallyLegalOp<ExtractArrayOp>(SplitExtractArrayOp::legal);
  target.addDynamicallyLegalOp<FuncDefOp>(SplitArrayInFuncDefOp::legal);
  target.addDynamicallyLegalOp<ContractOp>(SplitArrayInContractOp::legal);
  target.addDynamicallyLegalOp<ReturnOp>(SplitArrayInReturnOp::legal);
  target.addDynamicallyLegalOp<CallOp>(SplitArrayInCallOp::legal);
  target.addDynamicallyLegalOp<IncludeOp>(SplitArrayInIncludeOp::legal);
  target.addDynamicallyLegalOp<ArrayLengthOp>(ReplaceKnownArrayLengthOp::legal);
  target.addDynamicallyLegalOp<MemberWriteOp>(SplitArrayInMemberWriteOp::legal);
  target.addDynamicallyLegalOp<MemberReadOp>(SplitArrayInMemberReadOp::legal);

  LLVM_DEBUG(llvm::dbgs() << "Begin step 2: update/split other array ops\n";);
  return applyFullConversion(modOp, target, std::move(patterns));
}

/// Return a static index attribute for an array access, or null if any index is dynamic.
inline static ArrayAttr getIndexAsAttr(ArrayAccessOpInterface op) {
  return op.indexOperandsToAttributeArray();
}

/// Return whether `writeOp` may update `index`.
static bool mayWriteToIndex(WriteArrayOp writeOp, ArrayAttr index) {
  ArrayAttr writeIndex = getIndexAsAttr(writeOp);
  return !writeIndex || writeIndex == index;
}

/// Return whether the read is preceded by a write to the same array and index within its block.
static bool hasEarlierWriteInBlock(ReadArrayOp readOp, ArrayAttr readIndex) {
  Value arrRef = readOp.getArrRef();
  for (Operation &op : *readOp->getBlock()) {
    if (&op == readOp.getOperation()) {
      return false;
    }

    if (auto writeOp = dyn_cast<WriteArrayOp>(&op)) {
      if (writeOp.getArrRef() == arrRef && mayWriteToIndex(writeOp, readIndex)) {
        return true;
      }
      continue;
    }

    // Writes nested inside earlier operations may conditionally clobber the read's value.
    if (op.walk([arrRef, readIndex](WriteArrayOp writeOp) {
      if (writeOp.getArrRef() == arrRef && mayWriteToIndex(writeOp, readIndex)) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }).wasInterrupted()) {
      return true;
    }
  }
  return false;
}

/// Find a statically indexed write before the parent `scf.if` that can be forwarded to `readOp`.
static std::optional<WriteArrayOp> findPrecedingWriteForIfRead(ReadArrayOp readOp) {
  ArrayAttr readIndex = getIndexAsAttr(readOp);
  if (!readIndex) {
    return std::nullopt;
  }

  // Only handle reads that are direct children of an `scf.if` branch.
  auto ifOp = readOp->getParentOfType<scf::IfOp>();
  if (!ifOp || readOp->getBlock()->getParentOp() != ifOp.getOperation()) {
    return std::nullopt;
  }
  if (hasEarlierWriteInBlock(readOp, readIndex)) {
    return std::nullopt;
  }

  Block *ifBlock = ifOp->getBlock();
  if (!ifBlock) {
    return std::nullopt;
  }

  Value arrRef = readOp.getArrRef();
  WriteArrayOp replacement;
  for (Operation &op : *ifBlock) {
    if (&op == ifOp.getOperation()) {
      break;
    }

    if (auto writeOp = dyn_cast<WriteArrayOp>(&op)) {
      if (writeOp.getArrRef() != arrRef) {
        continue;
      }

      if (mayWriteToIndex(writeOp, readIndex)) {
        ArrayAttr writeIndex = getIndexAsAttr(writeOp);
        replacement = writeIndex == readIndex ? writeOp : WriteArrayOp();
      }
      continue;
    }

    // A nested write before the `scf.if` may overwrite the current candidate.
    if (op.walk([arrRef, readIndex, &replacement](WriteArrayOp writeOp) {
      if (writeOp.getArrRef() == arrRef && mayWriteToIndex(writeOp, readIndex)) {
        replacement = WriteArrayOp();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }).wasInterrupted()) {
      continue;
    }
  }

  return replacement ? std::make_optional(replacement) : std::nullopt;
}

/// Replace branch-local reads (in `scf.if`) with the value written by a same-index
/// write op that dominates the parent `scf.if`.
static void step3(ModuleOp modOp) {
  SmallVector<std::pair<ReadArrayOp, Value>> replacements;
  modOp.walk([&replacements](ReadArrayOp readOp) {
    if (std::optional<WriteArrayOp> writeOp = findPrecedingWriteForIfRead(readOp)) {
      replacements.emplace_back(readOp, writeOp->getRvalue());
    }
  });

  for (auto [readOp, value] : replacements) {
    readOp.getResult().replaceAllUsesWith(value);
    readOp.erase();
  }
}

/// Pass driver for the full array-to-scalar lowering pipeline described above.
class ArrayToScalarPass : public llzk::array::impl::ArrayToScalarPassBase<ArrayToScalarPass> {
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
      // issue is that the conversions for member read/write expect the mapping of array index to
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

    OpPassManager nestedPM(ModuleOp::getOperationName());
    // Use SROA (Destructurable* interfaces) to split each array with linear size `N` into `N`
    // arrays of size 1. This is necessary because the mem2reg pass cannot deal with indexing
    // and splitting up memory, i.e., it can only convert scalar memory access into SSA values.
    nestedPM.addPass(createSpecializedSROAPass<CreateArrayOp>());
    // The mem2reg pass converts all of the size-1 array allocation and access into SSA values.
    nestedPM.addPass(createSpecializedMem2RegPass<CreateArrayOp>());
    // Cleanup SSA values made dead by the transformations
    nestedPM.addPass(createRemoveDeadValuesPass());
    if (failed(runPipeline(nestedPM, module))) {
      signalPassFailure();
      return;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "After SROA+Mem2Reg pipeline:\n";
      module.dump();
    });
  }
};

} // namespace

/// Create the pass that rewrites eligible arrays into scalar SSA values.
std::unique_ptr<Pass> llzk::array::createArrayToScalarPass() {
  return std::make_unique<ArrayToScalarPass>();
};
