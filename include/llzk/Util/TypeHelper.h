//===-- TypeHelper.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/ErrorHelper.h"

#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>

namespace llzk {

// Forward declarations
namespace component {
class StructType;
} // namespace component
namespace array {
class ArrayType;
} // namespace array

/// Note: If any symbol refs in an input Type/Attribute use any of the special characters that this
/// class generates, they are not escaped. That means these string representations are not safe to
/// reverse back into a Type. It's only intended to produce a unique name for instantiated structs
/// that may give some hint when debugging regarding the original struct name and the params used.
class BuildShortTypeString {
  static constexpr char PLACEHOLDER = '\x1A';

  std::string ret;
  llvm::raw_string_ostream ss;

  BuildShortTypeString() : ret(), ss(ret) {}
  BuildShortTypeString &append(mlir::Type);
  BuildShortTypeString &append(mlir::ArrayRef<mlir::Attribute>);
  BuildShortTypeString &append(mlir::Attribute);

  void appendSymRef(mlir::SymbolRefAttr);
  void appendSymName(mlir::StringRef);

public:
  /// Return a brief string representation of the given LLZK type.
  static inline std::string from(mlir::Type type) {
    return BuildShortTypeString().append(type).ret;
  }

  /// Return a brief string representation of the attribute list from a parameterized type.
  /// Occurrences of `nullptr` are represented with a `PLACEHOLDER` character.
  static inline std::string from(mlir::ArrayRef<mlir::Attribute> attrs) {
    return BuildShortTypeString().append(attrs).ret;
  }

  /// Take an existing name prefix/base that contains N>=0 `PLACEHOLDER` character(s) and the
  /// Attribute list (size>=N) from a parameterized type. The first N elements in the list are
  /// formatted and used to replace the `PLACEHOLDER` character(s) in the base string. The remaining
  /// Attribute elements, if any, are formatted and appended to the end. Occurrences of `nullptr` in
  /// the Attribute list are formatted as the `PLACEHOLDER` character itself to allow for partial
  /// instantiation of a parameterized type, preserving the location of attributes that were not
  /// available in an earlier instantiation so they can be added by a later instantiation.
  static std::string from(const std::string &base, mlir::ArrayRef<mlir::Attribute> attrs);
};

// This function asserts that the given Attribute kind is legal within the LLZK types that can
// contain Attribute parameters (i.e., ArrayType, StructType, and TypeVarType). This should be used
// in any function that examines the attribute parameters within parameterized LLZK types to ensure
// that the function handles all possible cases properly, especially if more legal attributes are
// added in the future. Throw a fatal error if anything illegal is found, indicating that the caller
// of this function should be updated.
void assertValidAttrForParamOfType(mlir::Attribute attr);

/// valid types: {I1, Index, String, FeltType, StructType, ArrayType, TypeVarType}
bool isValidType(mlir::Type type);

/// valid types: {FeltType, StructType (with columns), ArrayType (that contains a valid column
/// type)}
bool isValidColumnType(
    mlir::Type type, mlir::SymbolTableCollection &symbolTable, mlir::Operation *op
);

/// valid types: isValidType() - {TypeVarType} - {types with variable parameters}
bool isValidGlobalType(mlir::Type type);

/// valid types: isValidType() - {String, StructType} (excluded via any type parameter nesting)
bool isValidEmitEqType(mlir::Type type);

/// valid types: {I1, Index, FeltType, TypeVarType}
bool isValidConstReadType(mlir::Type type);

/// valid types: isValidType() - {ArrayType}
bool isValidArrayElemType(mlir::Type type);

/// Checks if the type is a LLZK Array and it also contains a valid LLZK type.
bool isValidArrayType(mlir::Type type);

/// Return `false` iff the type contains any `TypeVarType`
bool isConcreteType(mlir::Type type, bool allowStructParams = true);

inline mlir::LogicalResult checkValidType(EmitErrorFn emitError, mlir::Type type) {
  if (!isValidType(type)) {
    return emitError() << "expected a valid LLZK type but found " << type;
  } else {
    return mlir::success();
  }
}

/// Return `true` iff the given type is a StructType referencing the `COMPONENT_NAME_SIGNAL` struct.
bool isSignalType(mlir::Type type);

/// Return `true` iff the given StructType is referencing the `COMPONENT_NAME_SIGNAL` struct.
bool isSignalType(component::StructType sType);

/// @brief Return `true` iff the given type contains an AffineMapAttr.
bool hasAffineMapAttr(mlir::Type type);

enum class Side : std::uint8_t { EMPTY = 0, LHS, RHS, TOMB };
static inline mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const Side &val) {
  switch (val) {
  case Side::EMPTY:
    os << "EMPTY";
    break;
  case Side::TOMB:
    os << "TOMB";
    break;
  case Side::LHS:
    os << "LHS";
    break;
  case Side::RHS:
    os << "RHS";
    break;
  }
  return os;
}

inline Side reverse(Side in) {
  switch (in) {
  case Side::LHS:
    return Side::RHS;
  case Side::RHS:
    return Side::LHS;
  default:
    return in;
  }
}

} // namespace llzk

namespace llvm {
template <> struct DenseMapInfo<llzk::Side> {
  using T = llzk::Side;
  static inline T getEmptyKey() { return T::EMPTY; }
  static inline T getTombstoneKey() { return T::TOMB; }
  static unsigned getHashValue(const T &val) {
    using UT = std::underlying_type_t<T>;
    return llvm::DenseMapInfo<UT>::getHashValue(static_cast<UT>(val));
  }
  static bool isEqual(const T &lhs, const T &rhs) { return lhs == rhs; }
};
} // namespace llvm

namespace llzk {

bool isDynamic(mlir::IntegerAttr intAttr);

/// Compute the cardinality (i.e. number of scalar constraints) for an EmitEqualityOp type since the
/// op can be used to constrain two same-size arrays.
uint64_t computeEmitEqCardinality(mlir::Type type);

/// Optional result from type unifications. Maps `SymbolRefAttr` appearing in one type to the
/// associated `Attribute` from the other type at the same nested position. The `Side` enum in the
/// key indicates which input expression the `SymbolRefAttr` is from. Additionally, if a conflict is
/// found (i.e., multiple Occurrences of a specific `SymbolRefAttr` on the same side map to
/// different Attributes from the other side). The mapped value will be `nullptr`.
///
/// This map is used by the `llzk-flatten` pass to replace struct parameter `SymbolRefAttr` with
/// static concrete values to produce the flattened versions of structs.
using UnificationMap = mlir::DenseMap<std::pair<mlir::SymbolRefAttr, Side>, mlir::Attribute>;

/// Return `true` iff the two ArrayRef instances containing StructType or ArrayType parameters
/// are equivalent or could be equivalent after full instantiation of struct parameters.
bool typeParamsUnify(
    const mlir::ArrayRef<mlir::Attribute> &lhsParams,
    const mlir::ArrayRef<mlir::Attribute> &rhsParams, UnificationMap *unifications = nullptr
);

/// Return `true` iff the two ArrayAttr instances containing StructType or ArrayType parameters
/// are equivalent or could be equivalent after full instantiation of struct parameters.
bool typeParamsUnify(
    const mlir::ArrayAttr &lhsParams, const mlir::ArrayAttr &rhsParams,
    UnificationMap *unifications = nullptr
);

/// Return `true` iff the two ArrayType instances are equivalent or could be equivalent after full
/// instantiation of struct parameters.
bool arrayTypesUnify(
    array::ArrayType lhs, array::ArrayType rhs,
    mlir::ArrayRef<llvm::StringRef> rhsReversePrefix = {}, UnificationMap *unifications = nullptr
);

/// Return `true` iff the two StructType instances are equivalent or could be equivalent after full
/// instantiation of struct parameters.
bool structTypesUnify(
    component::StructType lhs, component::StructType rhs,
    mlir::ArrayRef<llvm::StringRef> rhsReversePrefix = {}, UnificationMap *unifications = nullptr
);

/// Return `true` iff the two Type instances are equivalent or could be equivalent after full
/// instantiation of struct parameters (if applicable within the given types).
bool typesUnify(
    mlir::Type lhs, mlir::Type rhs, mlir::ArrayRef<llvm::StringRef> rhsReversePrefix = {},
    UnificationMap *unifications = nullptr
);

/// Return `true` iff the two lists of Type instances are equivalent or could be equivalent after
/// full instantiation of struct parameters (if applicable within the given types).
template <typename Iter1, typename Iter2>
inline bool typeListsUnify(
    Iter1 lhs, Iter2 rhs, mlir::ArrayRef<llvm::StringRef> rhsReversePrefix = {},
    UnificationMap *unifications = nullptr
) {
  return (lhs.size() == rhs.size()) &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(), [&](mlir::Type a, mlir::Type b) {
    return typesUnify(a, b, rhsReversePrefix, unifications);
  });
}

template <typename Iter1, typename Iter2>
inline bool singletonTypeListsUnify(
    Iter1 lhs, Iter2 rhs, mlir::ArrayRef<llvm::StringRef> rhsReversePrefix = {},
    UnificationMap *unifications = nullptr
) {
  return lhs.size() == 1 && rhs.size() == 1 &&
         typesUnify(lhs.front(), rhs.front(), rhsReversePrefix, unifications);
}

/// Return `true` iff the types unify and `newTy` is "more concrete" than `oldTy`.
///
/// The types `i1`, `index`, `felt.type`, and `string.type` are concrete whereas `poly.tvar` is
/// not (because it may be substituted with any type during struct instantiation). When considering
/// the attributes with `array.type` and `struct.type` types, we define IntegerAttr and TypeAttr
/// as concrete, AffineMapAttr as less concrete than those, and SymbolRefAttr as least concrete.
bool isMoreConcreteUnification(
    mlir::Type oldTy, mlir::Type newTy,
    llvm::function_ref<bool(mlir::Type oldTy, mlir::Type newTy)> knownOldToNew = nullptr
);

template <typename TypeClass> inline TypeClass getIfSingleton(mlir::TypeRange types) {
  return (types.size() == 1) ? llvm::dyn_cast<TypeClass>(types.front()) : nullptr;
}

template <typename TypeClass> inline TypeClass getAtIndex(mlir::TypeRange types, size_t index) {
  return (types.size() > index) ? llvm::dyn_cast<TypeClass>(types[index]) : nullptr;
}

/// Convert an IntegerAttr with a type other than IndexType to use IndexType.
mlir::FailureOr<mlir::IntegerAttr> forceIntType(mlir::IntegerAttr attr, EmitErrorFn emitError);

/// Convert any IntegerAttr with a type other than IndexType to use IndexType.
mlir::FailureOr<mlir::Attribute> forceIntAttrType(mlir::Attribute attr, EmitErrorFn emitError);

/// Convert any IntegerAttr with a type other than IndexType to use IndexType.
mlir::FailureOr<llvm::SmallVector<mlir::Attribute>>
forceIntAttrTypes(llvm::ArrayRef<mlir::Attribute> attrList, EmitErrorFn emitError);

/// Verify that all IntegerAttr have type IndexType.
mlir::LogicalResult verifyIntAttrType(EmitErrorFn emitError, mlir::Attribute in);

/// Verify that all AffineMapAttr only have a single result.
mlir::LogicalResult verifyAffineMapAttrType(EmitErrorFn emitError, mlir::Attribute in);

/// Verify that the StructType parameters are valid.
mlir::LogicalResult verifyStructTypeParams(EmitErrorFn emitError, mlir::ArrayAttr params);

/// Verify that the array dimensions are valid.
mlir::LogicalResult
verifyArrayDimSizes(EmitErrorFn emitError, mlir::ArrayRef<mlir::Attribute> dimensionSizes);

/// Verify that the ArrayType is valid.
mlir::LogicalResult verifyArrayType(
    EmitErrorFn emitError, mlir::Type elementType, mlir::ArrayRef<mlir::Attribute> dimensionSizes
);

/// Determine if the `subArrayType` is a valid subarray of `arrayType`.
/// `arrayType` must be an array of dimension N and `subArrayType` must be
/// an array of dimension M, where N > M >= 1.
/// For example, <3,7 x int> is a valid subarray of <5,3,7 x int>, but
/// <8 x int> is not an neither is <3,7 x string>.
mlir::LogicalResult verifySubArrayType(
    EmitErrorFn emitError, array::ArrayType arrayType, array::ArrayType subArrayType
);

/// Determine if the `subArrayOrElemType` is either a valid subarray of `arrayType`
/// (see `verifySubArrayType`), or if `subArrayOrElemType` matches the element
/// type of `arrayType`.
mlir::LogicalResult verifySubArrayOrElementType(
    EmitErrorFn emitError, array::ArrayType arrayType, mlir::Type subArrayOrElemType
);

} // namespace llzk
