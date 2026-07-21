//===-- AttributeHelper.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

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

/// Storage key for APInts whose numeric value is independent of bit width.
class APIntValue {
public:
  APIntValue(llvm::APInt apInt) : value(std::move(apInt)) {}

  const llvm::APInt &getValue() const { return value; }
  operator const llvm::APInt &() const { return value; }

  friend bool operator==(const APIntValue &lhs, const APIntValue &rhs) {
    return llvm::APInt::isSameValue(lhs.value, rhs.value);
  }

  friend llvm::hash_code hash_value(const APIntValue &key) {
    unsigned activeBits = key.value.getActiveBits();
    llvm::APInt canonical = key.value.trunc(activeBits == 0 ? 1 : activeBits);
    return llvm::hash_value(canonical);
  }

private:
  llvm::APInt value;
};

inline mlir::FailureOr<APIntValue> parseAPIntValue(mlir::AsmParser &parser) {
  mlir::FailureOr<llvm::APInt> value = mlir::FieldParser<llvm::APInt>::parse(parser);
  if (mlir::failed(value)) {
    return mlir::failure();
  }
  return APIntValue(std::move(*value));
}

inline llvm::APInt toAPInt(int64_t i) { return llvm::APInt(64, i); }
inline int64_t fromAPInt(const llvm::APInt &i) { return i.getSExtValue(); }

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
