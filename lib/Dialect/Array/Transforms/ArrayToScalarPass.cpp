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
/// 1. Run a dialect conversion that replaces `ArrayType` members with `N` scalar members.
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
/// 3. Run MLIR "sroa" pass to split each array with linear size `N` into `N` arrays of size 1 (to
///    prepare for "mem2reg" pass because it's API does not allow for indexing to split aggregates).
///
/// 4. Run MLIR "mem2reg" pass to convert all of the size 1 array allocation and access into SSA
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

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Array/Util/ArrayTypeHelper.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Transforms/LLZKConversionUtils.h"
#include "llzk/Util/Concepts.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/Debug.h>

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

/// Return `true` iff the given type is or contains an ArrayType that can be split into scalars.
inline bool containsSplittableArrayType(Type t) {
  return t
      .walk([](ArrayType a) {
    return splittableArray(a) ? WalkResult::interrupt() : WalkResult::skip();
  }).wasInterrupted();
}

/// Return `true` iff the given range contains any ArrayType that can be split into scalars.
template <typename T> bool containsSplittableArrayType(ValueTypeRange<T> types) {
  for (Type t : types) {
    if (containsSplittableArrayType(t)) {
      return true;
    }
  }
  return false;
}

/// If the given Type is an ArrayType that can be split into scalars, append `collect` with all of
/// the scalar types that result from splitting the ArrayType. Otherwise, just push the `Type`.
size_t splitArrayTypeTo(Type t, SmallVector<Type> &collect) {
  if (ArrayType at = splittableArray(t)) {
    int64_t n = at.getNumElements();
    assert(n >= 0);
    assert(std::cmp_less_equal(n, std::numeric_limits<size_t>::max()));
    size_t size = n;
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

inline WriteArrayOp
genWrite(Location loc, Value baseArrayOp, ArrayAttr index, Value init, RewriterBase &rewriter) {
  SmallVector<Value> readOperands = genIndexConstants(index, loc, rewriter);
  return rewriter.create<WriteArrayOp>(loc, baseArrayOp, ValueRange(readOperands), init);
}

/// Replace the given CallOp with a new one where any ArrayType in the results are split into their
/// scalar elements. Also, after the CallOp, generate a CreateArrayOp for each ArrayType result and
/// generate writes from the corresponding scalar result values to the new array.
CallOp newCallOpWithSplitResults(
    CallOp oldCall, CallOp::Adaptor adaptor, ConversionPatternRewriter &rewriter
) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(oldCall);

  Operation::result_range oldResults = oldCall.getResults();
  CallOp newCall = rewriter.create<CallOp>(
      oldCall.getLoc(), splitArrayType(oldResults.getTypes()), oldCall.getCallee(),
      adaptor.getArgOperands()
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
      newResults++;
    }
  }
  // erase the original CallOp
  rewriter.eraseOp(oldCall);

  return newCall;
}

inline ReadArrayOp
genRead(Location loc, Value baseArrayOp, ArrayAttr index, ConversionPatternRewriter &rewriter) {
  SmallVector<Value> readOperands = genIndexConstants(index, loc, rewriter);
  return rewriter.create<ReadArrayOp>(loc, baseArrayOp, ValueRange(readOperands));
}

// If the operand has ArrayType, add N reads from the array to the `newOperands` list otherwise add
// the original operand to the list.
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

// For each operand with ArrayType, add N reads from the array and use those N values instead.
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
      assert(std::cmp_less_equal(i, std::numeric_limits<int64_t>::max()));
      std::optional<SmallVector<Value>> multiDimIdxVals =
          idxGen.delinearize(static_cast<int64_t>(i), loc, rewriter);
      // ASSERT: CreateArrayOp verifier ensures the number of elements provided matches the full
      // linear array size so delinearization of `i` will not fail.
      assert(multiDimIdxVals.has_value());
      // Create the write
      rewriter.create<WriteArrayOp>(loc, op.getResult(), ValueRange(*multiDimIdxVals), init);
    }
  }
};

class SplitArrayInFuncDefOp : public OpConversionPattern<FuncDefOp> {
public:
  using OpConversionPattern<FuncDefOp>::OpConversionPattern;

  inline static bool legal(FuncDefOp op) {
    return !containsSplittableArrayType(op.getFunctionType());
  }

  // Create a new ArrayAttr like the one given but with repetitions of the elements according to the
  // mapping defined by `originalIdxToSize`. In other words, if `originalIdxToSize[i] = n`, then `n`
  // copies of `origAttrs[i]` are appended in its place.
  static ArrayAttr replicateAttributesAsNeeded(
      ArrayAttr origAttrs, const SmallVector<size_t> &originalIdxToSize,
      const SmallVector<Type> &newTypes
  ) {
    if (origAttrs) {
      assert(originalIdxToSize.size() == origAttrs.size());
      if (originalIdxToSize.size() != newTypes.size()) {
        SmallVector<Attribute> newArgAttrs;
        for (auto [i, s] : llvm::enumerate(originalIdxToSize)) {
          newArgAttrs.append(s, origAttrs[i]);
        }
        return ArrayAttr::get(origAttrs.getContext(), newArgAttrs);
      }
    }
    return nullptr;
  }

  LogicalResult match(FuncDefOp op) const override { return failure(legal(op)); }

  void rewrite(FuncDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    // Update in/out types of the function to replace arrays with scalars
    class Impl : public FunctionTypeConverter {
      SmallVector<size_t> originalInputIdxToSize, originalResultIdxToSize;

    protected:
      SmallVector<Type> convertInputs(ArrayRef<Type> origTypes) override {
        return splitArrayType(origTypes, &originalInputIdxToSize);
      }
      SmallVector<Type> convertResults(ArrayRef<Type> origTypes) override {
        return splitArrayType(origTypes, &originalResultIdxToSize);
      }
      ArrayAttr convertInputAttrs(ArrayAttr origAttrs, SmallVector<Type> newTypes) override {
        return replicateAttributesAsNeeded(origAttrs, originalInputIdxToSize, newTypes);
      }
      ArrayAttr convertResultAttrs(ArrayAttr origAttrs, SmallVector<Type> newTypes) override {
        return replicateAttributesAsNeeded(origAttrs, originalResultIdxToSize, newTypes);
      }

      /// For each argument to the Block that has a splittable ArrayType, replace it with the
      /// necessary number of scalar arguments, generate a CreateArrayOp, and generate writes from
      /// the new block scalar arguments to the new array. All users of the original block argument
      /// are updated to target the result of the CreateArrayOp.
      void processBlockArgs(Block &entryBlock, RewriterBase &rewriter) override {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&entryBlock);

        for (unsigned i = 0; i < entryBlock.getNumArguments();) {
          Value oldV = entryBlock.getArgument(i);
          if (ArrayType at = splittableArray(oldV.getType())) {
            Location loc = oldV.getLoc();
            // Generate `CreateArrayOp` and replace uses of the argument with it.
            auto newArray = rewriter.create<CreateArrayOp>(loc, at);
            rewriter.replaceAllUsesWith(oldV, newArray);
            // Remove the argument from the block
            entryBlock.eraseArgument(i);
            // For all indices in the ArrayType (i.e., the element count), generate a new block
            // argument and a write of that argument to the new array.
            std::optional<SmallVector<ArrayAttr>> allIndices = at.getSubelementIndices();
            assert(allIndices); // follows from legal() check
            assert(std::cmp_equal(allIndices->size(), at.getNumElements()));
            for (ArrayAttr subIdx : allIndices.value()) {
              BlockArgument newArg = entryBlock.insertArgument(i, at.getElementType(), loc);
              genWrite(loc, newArray, subIdx, newArg, rewriter);
              ++i;
            }
          } else {
            ++i;
          }
        }
      }
    };
    Impl().convert(op, rewriter);
  }
};

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

class SplitArrayInCallOp : public OpConversionPattern<CallOp> {
public:
  using OpConversionPattern<CallOp>::OpConversionPattern;

  inline static bool legal(CallOp op) {
    return !containsSplittableArrayType(op.getArgOperands().getTypes()) &&
           !containsSplittableArrayType(op.getResultTypes());
  }

  LogicalResult match(CallOp op) const override { return failure(legal(op)); }

  void rewrite(CallOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    assert(isNullOrEmpty(op.getMapOpGroupSizesAttr()) && "structs must be previously flattened");

    // Create new CallOp with split results first so, then process its inputs to split types
    CallOp newCall = newCallOpWithSplitResults(op, adaptor, rewriter);
    processInputOperands(
        newCall.getArgOperands(), newCall.getArgOperandsMutable(), newCall, rewriter
    );
  }
};

class ReplaceKnownArrayLengthOp : public OpConversionPattern<ArrayLengthOp> {
public:
  using OpConversionPattern<ArrayLengthOp>::OpConversionPattern;

  /// If 'dimIdx' is constant and that dimension of the ArrayType has static size, return it.
  static std::optional<llvm::APInt> getDimSizeIfKnown(Value dimIdx, ArrayType baseArrType) {
    if (splittableArray(baseArrType)) {
      llvm::APInt idxAP;
      if (mlir::matchPattern(dimIdx, mlir::m_ConstantInt(&idxAP))) {
        uint64_t idx64 = idxAP.getZExtValue();
        assert(std::cmp_less_equal(idx64, std::numeric_limits<size_t>::max()));
        Attribute dimSizeAttr = baseArrType.getDimensionSizes()[static_cast<size_t>(idx64)];
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
/// ArrayAttr index -> scalar member info
using LocalMemberReplacementMap = DenseMap<ArrayAttr, MemberInfo>;
/// struct -> array-type member name -> LocalMemberReplacementMap
using MemberReplacementMap = DenseMap<StructDefOp, DenseMap<StringAttr, LocalMemberReplacementMap>>;

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
          op.getLoc(), op.getSymNameAttr(), elemTy, op.getColumn(), op.getSignal()
      );
      newMember.setPublicAttr(op.hasPublicAttr());
      // Use SymbolTable to give it a unique name and store to the replacement map
      localRepMapRef[idx] = std::make_pair(structSymbolTable.insert(newMember), elemTy);
    }
    rewriter.eraseOp(op);
  }
};

/// Common implementation for handling `MemberWriteOp` and `MemberReadOp`.
///
/// @tparam ImplClass         the concrete subclass
/// @tparam MemberRefOpClass   the concrete op class
/// @tparam GenHeaderType     return type of `genHeader()`, used to pass data to `forIndex()`
template <
    typename ImplClass, HasInterface<MemberRefOpInterface> MemberRefOpClass, typename GenHeaderType>
class SplitArrayInMemberRefOp : public OpConversionPattern<MemberRefOpClass> {
  SymbolTableCollection &tables;
  const MemberReplacementMap &repMapRef;

  // static check to ensure the functions are implemented in all subclasses
  inline static void ensureImplementedAtCompile() {
    static_assert(
        sizeof(MemberRefOpClass) == 0, "SplitArrayInMemberRefOp not implemented for requested type."
    );
  }

protected:
  using OpAdaptor = typename MemberRefOpClass::Adaptor;

  /// Executed at the start of `rewrite()` to (optionally) generate anything that should be before
  /// the element-wise operations that will be added by `forIndex()`.
  static GenHeaderType genHeader(MemberRefOpClass, ConversionPatternRewriter &) {
    ensureImplementedAtCompile();
    llvm_unreachable("must have concrete instantiation");
  }

  /// Executed for each multi-dimensional array index in the ArrayType of the original member to
  /// generate the element-wise scalar operations on the new scalar members.
  static void
  forIndex(Location, GenHeaderType, ArrayAttr, MemberInfo, OpAdaptor, ConversionPatternRewriter &) {
    ensureImplementedAtCompile();
    llvm_unreachable("must have concrete instantiation");
  }

public:
  SplitArrayInMemberRefOp(
      MLIRContext *ctx, SymbolTableCollection &symTables, const MemberReplacementMap &memberRepMap
  )
      : OpConversionPattern<MemberRefOpClass>(ctx), tables(symTables), repMapRef(memberRepMap) {}

  static bool legal(MemberRefOpClass) {
    ensureImplementedAtCompile();
    llvm_unreachable("must have concrete instantiation");
    return false;
  }

  LogicalResult match(MemberRefOpClass op) const override { return failure(ImplClass::legal(op)); }

  void rewrite(
      MemberRefOpClass op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    StructType tgtStructTy = llvm::cast<MemberRefOpInterface>(op.getOperation()).getStructType();
    assert(tgtStructTy);
    auto tgtStructDef = tgtStructTy.getDefinition(tables, op);
    assert(succeeded(tgtStructDef));

    GenHeaderType prefixResult = ImplClass::genHeader(op, rewriter);

    const LocalMemberReplacementMap &idxToName =
        repMapRef.at(tgtStructDef->get()).at(op.getMemberNameAttr().getAttr());
    // Split the array member write into a series of read array + write scalar member
    for (auto [idx, newMember] : idxToName) {
      ImplClass::forIndex(op.getLoc(), prefixResult, idx, newMember, adaptor, rewriter);
    }
    rewriter.eraseOp(op);
  }
};

class SplitArrayInMemberWriteOp
    : public SplitArrayInMemberRefOp<SplitArrayInMemberWriteOp, MemberWriteOp, void *> {
public:
  using SplitArrayInMemberRefOp<
      SplitArrayInMemberWriteOp, MemberWriteOp, void *>::SplitArrayInMemberRefOp;

  static bool legal(MemberWriteOp op) {
    return !containsSplittableArrayType(op.getVal().getType());
  }

  static void *genHeader(MemberWriteOp, ConversionPatternRewriter &) { return nullptr; }

  static void forIndex(
      Location loc, void *, ArrayAttr idx, MemberInfo newMember, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter
  ) {
    ReadArrayOp scalarRead = genRead(loc, adaptor.getVal(), idx, rewriter);
    rewriter.create<MemberWriteOp>(
        loc, adaptor.getComponent(), FlatSymbolRefAttr::get(newMember.first), scalarRead
    );
  }
};

class SplitArrayInMemberReadOp
    : public SplitArrayInMemberRefOp<SplitArrayInMemberReadOp, MemberReadOp, CreateArrayOp> {
public:
  using SplitArrayInMemberRefOp<
      SplitArrayInMemberReadOp, MemberReadOp, CreateArrayOp>::SplitArrayInMemberRefOp;

  static bool legal(MemberReadOp op) {
    return !containsSplittableArrayType(op.getResult().getType());
  }

  static CreateArrayOp genHeader(MemberReadOp op, ConversionPatternRewriter &rewriter) {
    CreateArrayOp newArray =
        rewriter.create<CreateArrayOp>(op.getLoc(), llvm::cast<ArrayType>(op.getType()));
    rewriter.replaceAllUsesWith(op, newArray);
    return newArray;
  }

  static void forIndex(
      Location loc, CreateArrayOp newArray, ArrayAttr idx, MemberInfo newMember, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter
  ) {
    MemberReadOp scalarRead = rewriter.create<MemberReadOp>(
        loc, newMember.second, adaptor.getComponent(), newMember.first
    );
    genWrite(loc, newArray, idx, scalarRead, rewriter);
  }
};

LogicalResult
step1(ModuleOp modOp, SymbolTableCollection &symTables, MemberReplacementMap &memberRepMap) {
  MLIRContext *ctx = modOp.getContext();

  RewritePatternSet patterns(ctx);

  patterns.add<SplitArrayInMemberDefOp>(ctx, symTables, memberRepMap);

  ConversionTarget target(*ctx);
  target.addLegalDialect<
      LLZKDialect, array::ArrayDialect, boolean::BoolDialect, felt::FeltDialect,
      function::FunctionDialect, global::GlobalDialect, include::IncludeDialect,
      component::StructDialect, constrain::ConstrainDialect, arith::ArithDialect,
      scf::SCFDialect>();
  target.addLegalOp<ModuleOp>();
  target.addDynamicallyLegalOp<MemberDefOp>(SplitArrayInMemberDefOp::legal);

  LLVM_DEBUG(llvm::dbgs() << "Begin step 1: split array members\n";);
  return applyFullConversion(modOp, target, std::move(patterns));
}

LogicalResult
step2(ModuleOp modOp, SymbolTableCollection &symTables, const MemberReplacementMap &memberRepMap) {
  MLIRContext *ctx = modOp.getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<
      // clang-format off
      SplitInitFromCreateArrayOp,
      SplitInsertArrayOp,
      SplitExtractArrayOp,
      SplitArrayInFuncDefOp,
      SplitArrayInReturnOp,
      SplitArrayInCallOp,
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
  target.addLegalDialect<
      LLZKDialect, array::ArrayDialect, boolean::BoolDialect, component::StructDialect,
      constrain::ConstrainDialect, felt::FeltDialect, function::FunctionDialect,
      global::GlobalDialect, include::IncludeDialect, arith::ArithDialect, scf::SCFDialect>();
  target.addLegalOp<ModuleOp>();
  target.addDynamicallyLegalOp<CreateArrayOp>(SplitInitFromCreateArrayOp::legal);
  target.addDynamicallyLegalOp<InsertArrayOp>(SplitInsertArrayOp::legal);
  target.addDynamicallyLegalOp<ExtractArrayOp>(SplitExtractArrayOp::legal);
  target.addDynamicallyLegalOp<FuncDefOp>(SplitArrayInFuncDefOp::legal);
  target.addDynamicallyLegalOp<ReturnOp>(SplitArrayInReturnOp::legal);
  target.addDynamicallyLegalOp<CallOp>(SplitArrayInCallOp::legal);
  target.addDynamicallyLegalOp<ArrayLengthOp>(ReplaceKnownArrayLengthOp::legal);
  target.addDynamicallyLegalOp<MemberWriteOp>(SplitArrayInMemberWriteOp::legal);
  target.addDynamicallyLegalOp<MemberReadOp>(SplitArrayInMemberReadOp::legal);

  LLVM_DEBUG(llvm::dbgs() << "Begin step 2: update/split other array ops\n";);
  return applyFullConversion(modOp, target, std::move(patterns));
}

LogicalResult splitArrayCreateInit(ModuleOp modOp) {
  SymbolTableCollection symTables;
  MemberReplacementMap memberRepMap;

  // This is divided into 2 steps to simplify the implementation for member-related ops. The issue
  // is that the conversions for member read/write expect the mapping of array index to member
  // name+type to already be populated for the referenced member (although this could be computed on
  // demand if desired but it complicates the implementation a bit).
  if (failed(step1(modOp, symTables, memberRepMap))) {
    return failure();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "After step 1:\n";
    modOp.dump();
  });
  if (failed(step2(modOp, symTables, memberRepMap))) {
    return failure();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "After step 2:\n";
    modOp.dump();
  });
  return success();
}

class ArrayToScalarPass : public llzk::array::impl::ArrayToScalarPassBase<ArrayToScalarPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Separate array initialization from creation by removing the initialization list from
    // CreateArrayOp and inserting the corresponding WriteArrayOp following it.
    if (failed(splitArrayCreateInit(module))) {
      signalPassFailure();
      return;
    }
    OpPassManager nestedPM(ModuleOp::getOperationName());
    // Use SROA (Destructurable* interfaces) to split each array with linear size N into N arrays of
    // size 1. This is necessary because the mem2reg pass cannot deal with indexing and splitting up
    // memory, i.e., it can only convert scalar memory access into SSA values.
    nestedPM.addPass(createSROA());
    // The mem2reg pass converts all of the size 1 array allocation and access into SSA values.
    nestedPM.addPass(createMem2Reg());
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

std::unique_ptr<Pass> llzk::array::createArrayToScalarPass() {
  return std::make_unique<ArrayToScalarPass>();
};
