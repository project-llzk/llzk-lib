//===-- AttributeHelper.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/StreamHelper.h"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/Hashing.h>

#include <utility>

template <> struct mlir::FieldParser<llvm::APInt> {
  static mlir::FailureOr<llvm::APInt> parse(mlir::AsmParser &parser) {
    auto loc = parser.getCurrentLocation();
    llvm::APInt val;
    auto result = parser.parseOptionalInteger(val);
    if (!result.has_value() || *result) {
      return parser.emitError(loc, "expected integer value");
    } else {
      return val;
    }
  }
};

namespace llzk {

/// Attribute storage adapter providing numeric hashing for DynamicAPInt.
///
/// This is meant as a thin adaptor for storing DynamicAPInt in attribute storage generated
/// by tablegen. It should not generally be used outside of attribute storage contexts.
/// Public accessors on the attribute expose the underlying value directly rather than
/// exposing this wrapper.
class DynamicAPIntValue {
public:
  DynamicAPIntValue(const llvm::DynamicAPInt &integer) : value(integer) {}

  /// Borrow the owned integer.
  ///
  /// The returned reference must not outlive this wrapper. For the expected use case of
  /// wrappers held in attribute storage, its lifetime is that of the owning MLIRContext
  /// so this borrow is safe. References obtained from temporary instances of this wrapper
  /// must not be held beyond the full expression that creates the wrapper.
  operator const llvm::DynamicAPInt &() const { return value; }

  friend bool operator==(const DynamicAPIntValue &a, const DynamicAPIntValue &b) {
    return a.value == b.value;
  }

  friend llvm::hash_code hash_value(const DynamicAPIntValue &key) {
    return hashDynamicAPInt(key.value);
  }

private:
  llvm::DynamicAPInt value;
};

/// Parse MLIR's signed integer literal and discard its encoding width.
inline mlir::FailureOr<DynamicAPIntValue> parseDynamicAPIntValue(mlir::AsmParser &parser) {
  auto value = mlir::FieldParser<llvm::APInt>::parse(parser);
  if (mlir::failed(value)) {
    return mlir::failure();
  }
  return DynamicAPIntValue(llvm::DynamicAPInt(*value));
}

/// Decode an MLIR integer using its signedness; signless i1 denotes a boolean.
/// Other signless integers and index values use signed interpretation.
inline llvm::DynamicAPInt integerAttrToDynamicAPInt(mlir::IntegerAttr attr) {
  auto type = llvm::dyn_cast<mlir::IntegerType>(attr.getType());
  bool isUnsigned = type && (type.isUnsigned() || (type.isSignless() && type.getWidth() == 1));
  return isUnsigned ? toDynamicAPInt(attr.getValue()) : llvm::DynamicAPInt(attr.getValue());
}

/// Extract a verified MLIR index value; malformed internal IR is a fatal error.
inline int64_t fromAPInt(const llvm::APInt &i) {
  if (!i.isSignedIntN(64)) {
    llvm::report_fatal_error("integer is not representable as an index");
  }
  return i.getSExtValue();
}

inline bool isNullOrEmpty(mlir::ArrayAttr a) { return !a || a.empty(); }
inline bool isNullOrEmpty(mlir::DenseArrayAttr a) { return !a || a.empty(); }
inline bool isNullOrEmpty(mlir::DictionaryAttr a) { return !a || a.empty(); }

inline void appendWithoutType(mlir::raw_ostream &os, mlir::Attribute a) { a.print(os, true); }
inline std::string stringWithoutType(mlir::Attribute a) {
  return buildStringViaCallback(appendWithoutType, a);
}

void printAttrs(
    mlir::AsmPrinter &printer, mlir::ArrayRef<mlir::Attribute> attrs,
    const mlir::StringRef &separator
);

} // namespace llzk
