//===-- Ops.cpp - Array operation implementations ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/Util/ArrayTypeHelper.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Util/BuilderHelper.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Twine.h>

// TableGen'd implementation files
#include "llzk/Dialect/Array/IR/OpInterfaces.cpp.inc"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Array/IR/Ops.cpp.inc"

using namespace mlir;

namespace llzk::array {

//===------------------------------------------------------------------===//
// CreateArrayOp
//===------------------------------------------------------------------===//

void CreateArrayOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, ArrayType result, ValueRange elements
) {
  odsState.addTypes(result);
  odsState.addOperands(elements);
  // This builds CreateArrayOp from a list of elements. In that case, the dimensions of the array
  // type cannot be defined via an affine map which means there are no affine map operands.
  affineMapHelpers::buildInstantiationAttrsEmpty<CreateArrayOp>(
      odsBuilder, odsState, static_cast<int32_t>(elements.size())
  );
}

void CreateArrayOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, ArrayType result,
    ArrayRef<ValueRange> mapOperands, DenseI32ArrayAttr numDimsPerMap
) {
  odsState.addTypes(result);
  affineMapHelpers::buildInstantiationAttrs<CreateArrayOp>(
      odsBuilder, odsState, mapOperands, numDimsPerMap
  );
}

LogicalResult CreateArrayOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, *this, llvm::cast<Type>(getType()));
}

void CreateArrayOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "array");
}

llvm::SmallVector<Type> CreateArrayOp::resultTypeToElementsTypes(Type resultType) {
  // The ODS restricts $result with LLZK_ArrayType so this cast is safe.
  ArrayType a = llvm::cast<ArrayType>(resultType);
  return llvm::SmallVector<Type>(a.getNumElements(), a.getElementType());
}

ParseResult CreateArrayOp::parseInferredArrayType(
    OpAsmParser &parser, llvm::SmallVector<Type, 1> &elementsTypes,
    ArrayRef<OpAsmParser::UnresolvedOperand> elements, Type resultType
) {
  assert(elementsTypes.size() == 0); // it was not yet initialized
  // If the '$elements' operand is not empty, then the expected type for the operand
  //  is computed to match the type of the '$result'. Otherwise, it remains empty.
  if (elements.size() > 0) {
    elementsTypes.append(resultTypeToElementsTypes(resultType));
  }
  return success();
}

void CreateArrayOp::printInferredArrayType(
    OpAsmPrinter &printer, CreateArrayOp, TypeRange, OperandRange, Type
) {
  // nothing to print, it's derived and therefore not represented in the output
}

LogicalResult CreateArrayOp::verify() {
  Type retTy = getResult().getType();
  assert(llvm::isa<ArrayType>(retTy)); // per ODS spec of CreateArrayOp

  // Collect the array dimensions that are defined via AffineMapAttr
  SmallVector<AffineMapAttr> mapAttrs;
  // Extend the lifetime of the temporary to suppress warnings.
  ArrayType arrTy = llvm::cast<ArrayType>(retTy);
  for (Attribute a : arrTy.getDimensionSizes()) {
    if (AffineMapAttr m = dyn_cast<AffineMapAttr>(a)) {
      mapAttrs.push_back(m);
    }
  }
  return affineMapHelpers::verifyAffineMapInstantiations(
      getMapOperands(), getNumDimsPerMap(), mapAttrs, *this
  );
}

/// Required by DestructurableAllocationOpInterface / SROA pass
SmallVector<DestructurableMemorySlot> CreateArrayOp::getDestructurableSlots() {
  assert(getElements().empty() && "must run after initialization is split from allocation");
  ArrayType arrType = getType();
  if (!arrType.hasStaticShape() || arrType.getNumElements() == 1) {
    return {};
  }
  if (auto destructured = arrType.getSubelementIndexMap()) {
    return {DestructurableMemorySlot {{getResult(), arrType}, std::move(*destructured)}};
  }
  return {};
}

/// Required by DestructurableAllocationOpInterface / SROA pass
DenseMap<Attribute, MemorySlot> CreateArrayOp::destructure(
    const DestructurableMemorySlot &slot, const SmallPtrSetImpl<Attribute> &usedIndices,
    OpBuilder &builder, SmallVectorImpl<DestructurableAllocationOpInterface> &newAllocators
) {
  assert(slot.ptr == getResult());
  assert(slot.elemType == getType());

  builder.setInsertionPointAfter(*this);

  DenseMap<Attribute, MemorySlot> slotMap; // result
  for (Attribute index : usedIndices) {
    // This is an ArrayAttr since indexing is multi-dimensional
    ArrayAttr indexAsArray = llvm::dyn_cast<ArrayAttr>(index);
    assert(indexAsArray && "expected ArrayAttr");

    Type destructAs = getType().getTypeAtIndex(indexAsArray);
    assert(destructAs == slot.subelementTypes.lookup(indexAsArray));

    ArrayType destructAsArrayTy = llvm::dyn_cast<ArrayType>(destructAs);
    assert(destructAsArrayTy && "expected ArrayType");

    auto subCreate = builder.create<CreateArrayOp>(getLoc(), destructAsArrayTy);
    newAllocators.push_back(subCreate);
    slotMap.try_emplace<MemorySlot>(index, {subCreate.getResult(), destructAs});
  }

  return slotMap;
}

/// Required by DestructurableAllocationOpInterface / SROA pass
std::optional<DestructurableAllocationOpInterface> CreateArrayOp::handleDestructuringComplete(
    const DestructurableMemorySlot &slot, OpBuilder & /*builder*/
) {
  assert(slot.ptr == getResult());
  this->erase();
  return std::nullopt;
}

/// Required by PromotableAllocationOpInterface / mem2reg pass
SmallVector<MemorySlot> CreateArrayOp::getPromotableSlots() {
  ArrayType arrType = getType();
  if (!arrType.hasStaticShape()) {
    return {};
  }
  // Can only support arrays containing a single element (the SROA pass can be run first to
  // destructure all arrays into size-1 arrays).
  if (arrType.getNumElements() != 1) {
    return {};
  }
  return {MemorySlot {getResult(), arrType.getElementType()}};
}

/// Required by PromotableAllocationOpInterface / mem2reg pass
Value CreateArrayOp::getDefaultValue(const MemorySlot &slot, OpBuilder &builder) {
  return builder.create<llzk::NonDetOp>(getLoc(), slot.elemType);
}

/// Required by PromotableAllocationOpInterface / mem2reg pass
void CreateArrayOp::handleBlockArgument(const MemorySlot &, BlockArgument, OpBuilder &) {}

/// Required by PromotableAllocationOpInterface / mem2reg pass
std::optional<PromotableAllocationOpInterface> CreateArrayOp::handlePromotionComplete(
    const MemorySlot & /*slot*/, Value defaultValue, OpBuilder & /*builder*/
) {
  if (defaultValue.use_empty()) {
    defaultValue.getDefiningOp()->erase();
  } else {
    this->erase();
  }
  // Return `nullopt` because it produces only a single slot
  return std::nullopt;
}

//===------------------------------------------------------------------===//
// ArrayAccessOpInterface
//===------------------------------------------------------------------===//

/// Returns the multi-dimensional indices of the array access as an Attribute
/// array or a null pointer if a valid index cannot be computed for any dimension.
ArrayAttr ArrayAccessOpInterface::indexOperandsToAttributeArray() {
  ArrayType arrTy = getArrRefType();
  if (arrTy.hasStaticShape()) {
    if (auto converted = ArrayIndexGen::from(arrTy).checkAndConvert(getIndices())) {
      return ArrayAttr::get(getContext(), *converted);
    }
  }
  return nullptr;
}

/// Required by DestructurableAllocationOpInterface / SROA pass
bool ArrayAccessOpInterface::canRewire(
    const DestructurableMemorySlot &slot, SmallPtrSetImpl<Attribute> &usedIndices,
    SmallVectorImpl<MemorySlot> & /*mustBeSafelyUsed*/, const DataLayout & /*dataLayout*/
) {
  if (slot.ptr != getArrRef()) {
    return false;
  }

  ArrayAttr indexAsAttr = indexOperandsToAttributeArray();
  if (!indexAsAttr) {
    return false;
  }

  // Scalar read/write case has 0 dimensions in the read/write value.
  if (!getValueOperandDims().empty()) {
    return false;
  }

  // Just insert the index.
  usedIndices.insert(indexAsAttr);
  return true;
}

/// Required by DestructurableAllocationOpInterface / SROA pass
DeletionKind ArrayAccessOpInterface::rewire(
    const DestructurableMemorySlot &slot, DenseMap<Attribute, MemorySlot> &subslots,
    OpBuilder &builder, const DataLayout & /*dataLayout*/
) {
  assert(slot.ptr == getArrRef());
  assert(slot.elemType == getArrRefType());
  // ASSERT: non-scalar read/write should have been desugared earlier
  assert(getValueOperandDims().empty() && "only scalar read/write supported");

  ArrayAttr indexAsAttr = indexOperandsToAttributeArray();
  assert(indexAsAttr && "canRewire() should have returned false");
  const MemorySlot &memorySlot = subslots.at(indexAsAttr);

  // Temporarily set insertion point before the current op for what's built below
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(this->getOperation());

  //  Write to the sub-slot created for the index of `this`, using index 0
  getArrRefMutable().set(memorySlot.ptr);
  getIndicesMutable().clear();
  getIndicesMutable().assign(builder.create<arith::ConstantIndexOp>(getLoc(), 0));

  return DeletionKind::Keep;
}

//===------------------------------------------------------------------===//
// ReadArrayOp
//===------------------------------------------------------------------===//

namespace {

LogicalResult
ensureNumIndicesMatchDims(ArrayType ty, size_t numIndices, const OwningEmitErrorFn &errFn) {
  ArrayRef<Attribute> dims = ty.getDimensionSizes();
  // Ensure the number of provided indices matches the array dimensions
  auto compare = numIndices <=> dims.size();
  if (compare != 0) {
    return errFn().append(
        "has ", (compare < 0 ? "insufficient" : "too many"), " indexed dimensions: expected ",
        dims.size(), " but found ", numIndices
    );
  }
  return success();
}

} // namespace

LogicalResult ReadArrayOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, *this, ArrayRef<Type> {getArrRef().getType(), getType()});
}

LogicalResult ReadArrayOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ReadArrayOpAdaptor adaptor,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes
) {
  inferredReturnTypes.resize(1);
  Type lvalType = adaptor.getArrRef().getType();
  assert(llvm::isa<ArrayType>(lvalType)); // per ODS spec of ReadArrayOp
  inferredReturnTypes[0] = llvm::cast<ArrayType>(lvalType).getElementType();
  return success();
}

bool ReadArrayOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  return singletonTypeListsUnify(l, r);
}

LogicalResult ReadArrayOp::verify() {
  // Ensure the number of indices used match the shape of the array exactly.
  return ensureNumIndicesMatchDims(getArrRefType(), getIndices().size(), getEmitOpErrFn(this));
}

/// Required by PromotableMemOpInterface / mem2reg pass
bool ReadArrayOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses, const DataLayout & /*datalayout*/
) {
  if (blockingUses.size() != 1) {
    return false;
  }
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getArrRef() == slot.ptr &&
         getResult().getType() == slot.elemType;
}

/// Required by PromotableMemOpInterface / mem2reg pass
DeletionKind ReadArrayOp::removeBlockingUses(
    const MemorySlot & /*slot*/, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder & /*builder*/, Value reachingDefinition, const DataLayout & /*dataLayout*/
) {
  // `canUsesBeRemoved` checked this blocking use must be the loaded `slot.ptr`
  getResult().replaceAllUsesWith(reachingDefinition);
  return DeletionKind::Delete;
}

//===------------------------------------------------------------------===//
// WriteArrayOp
//===------------------------------------------------------------------===//

LogicalResult WriteArrayOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(
      tables, *this, ArrayRef<Type> {getArrRefType(), getRvalue().getType()}
  );
}

LogicalResult WriteArrayOp::verify() {
  // Ensure the number of indices used match the shape of the array exactly.
  return ensureNumIndicesMatchDims(getArrRefType(), getIndices().size(), getEmitOpErrFn(this));
}

/// Required by PromotableMemOpInterface / mem2reg pass
bool WriteArrayOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses, const DataLayout & /*datalayout*/
) {
  if (blockingUses.size() != 1) {
    return false;
  }
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getArrRef() == slot.ptr && getRvalue() != slot.ptr &&
         getRvalue().getType() == slot.elemType;
}

/// Required by PromotableMemOpInterface / mem2reg pass
DeletionKind WriteArrayOp::removeBlockingUses(
    const MemorySlot &, const SmallPtrSetImpl<OpOperand *> &, OpBuilder &, Value, const DataLayout &
) {
  return DeletionKind::Delete;
}

//===------------------------------------------------------------------===//
// ExtractArrayOp
//===------------------------------------------------------------------===//

LogicalResult ExtractArrayOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, *this, getArrRefType());
}

LogicalResult ExtractArrayOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ExtractArrayOpAdaptor adaptor,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes
) {
  size_t numToSkip = adaptor.getIndices().size();
  Type arrRefType = adaptor.getArrRef().getType();
  assert(llvm::isa<ArrayType>(arrRefType)); // per ODS spec of ExtractArrayOp
  ArrayType arrRefArrType = llvm::cast<ArrayType>(arrRefType);
  ArrayRef<Attribute> arrRefDimSizes = arrRefArrType.getDimensionSizes();

  // Check for invalid cases
  auto compare = numToSkip <=> arrRefDimSizes.size();
  if (compare == 0) {
    return mlir::emitOptionalError(
        location, '\'', ExtractArrayOp::getOperationName(),
        "' op cannot select all dimensions of an array. Use '", ReadArrayOp::getOperationName(),
        "' instead."
    );
  } else if (compare > 0) {
    return mlir::emitOptionalError(
        location, '\'', ExtractArrayOp::getOperationName(),
        "' op cannot select more dimensions than exist in the source array"
    );
  }

  // Generate and store reduced array type
  inferredReturnTypes.resize(1);
  inferredReturnTypes[0] =
      ArrayType::get(arrRefArrType.getElementType(), arrRefDimSizes.drop_front(numToSkip));
  return success();
}

bool ExtractArrayOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  return singletonTypeListsUnify(l, r);
}

//===------------------------------------------------------------------===//
// InsertArrayOp
//===------------------------------------------------------------------===//

LogicalResult InsertArrayOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the types are valid
  return verifyTypeResolution(
      tables, *this, ArrayRef<Type> {getArrRefType(), getRvalue().getType()}
  );
}

LogicalResult InsertArrayOp::verify() {
  ArrayType baseArrRefArrType = getArrRefType();
  Type rValueType = getRvalue().getType();
  assert(llvm::isa<ArrayType>(rValueType)); // per ODS spec of InsertArrayOp
  ArrayType rValueArrType = llvm::cast<ArrayType>(rValueType);

  // size of lhs dimensions == numIndices + size of rhs dimensions
  size_t lhsDims = baseArrRefArrType.getDimensionSizes().size();
  size_t numIndices = getIndices().size();
  size_t rhsDims = rValueArrType.getDimensionSizes().size();

  // Ensure the number of indices specified does not exceed base dimension count.
  if (numIndices > lhsDims) {
    return emitOpError("cannot select more dimensions than exist in the source array");
  }

  // Ensure the rValue dimension count equals the base reduced dimension count
  auto compare = (numIndices + rhsDims) <=> lhsDims;
  if (compare != 0) {
    return emitOpError().append(
        "has ", (compare < 0 ? "insufficient" : "too many"), " indexed dimensions: expected ",
        (lhsDims - rhsDims), " but found ", numIndices
    );
  }

  // Having verified the indices are of appropriate size, we verify the subarray type.
  // This will verify the dimensions of the subarray, which is why we only check the
  // size of the indices above.
  return verifySubArrayType(getEmitOpErrFn(this), baseArrRefArrType, rValueArrType);
}

//===------------------------------------------------------------------===//
// ArrayLengthOp
//===------------------------------------------------------------------===//

LogicalResult ArrayLengthOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, *this, getArrRefType());
}

} // namespace llzk::array
