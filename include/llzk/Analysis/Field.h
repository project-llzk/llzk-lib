//===-- Field.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/DynamicAPIntHelper.h"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/SMTAPI.h>

#include <string_view>

namespace llzk {

/// @brief Information about the prime finite field used for the interval analysis.
/// @note See implementation of initKnownFields for supported primes.
/// @note We use DynamicAPInt to support arithmetic that may require increasing
/// or signed arithmetic (e.g., multiplying field elements before applying the
/// modulus).
class Field {
public:
  /// @brief Get a Field from a given field name string, or failure if the
  /// field is unsupported.
  /// @param fieldName The name of the field.
  static llvm::FailureOr<std::reference_wrapper<const Field>> tryGetField(const char *fieldName);

  /// @brief Get a Field from a given field name string. Throws a fatal error
  /// if the field is unsupported.
  /// @param fieldName The name of the field.
  static const Field &getField(const char *fieldName);

  Field() = delete;
  Field(const Field &) = default;
  Field(Field &&) noexcept = default;
  Field &operator=(const Field &) = default;

  /// @brief For the prime field p, returns p.
  llvm::DynamicAPInt prime() const { return primeMod; }

  /// @brief Returns p / 2.
  llvm::DynamicAPInt half() const { return halfPrime; }

  /// @brief Returns i as a signed field element
  inline llvm::DynamicAPInt felt(int i) const { return reduce(i); }

  /// @brief Returns 0 at the bitwidth of the field.
  inline llvm::DynamicAPInt zero() const { return felt(0); }

  /// @brief Returns 1 at the bitwidth of the field.
  inline llvm::DynamicAPInt one() const { return felt(1); }

  /// @brief Returns p - 1, which is the max value possible in a prime field described by p.
  inline llvm::DynamicAPInt maxVal() const { return prime() - one(); }

  /// @brief Returns the multiplicative inverse of `i` in prime field `p`.
  llvm::DynamicAPInt inv(const llvm::DynamicAPInt &i) const;

  llvm::DynamicAPInt inv(const llvm::APInt &i) const;

  /// @brief Returns i mod p and reduces the result into the appropriate bitwidth.
  /// Field elements are returned as signed integers so that negation functions
  /// as expected (i.e., reducing -1 will yield p-1).
  llvm::DynamicAPInt reduce(const llvm::DynamicAPInt &i) const;
  inline llvm::DynamicAPInt reduce(int i) const { return reduce(llvm::DynamicAPInt(i)); }
  llvm::DynamicAPInt reduce(const llvm::APInt &i) const;

  inline unsigned bitWidth() const { return bitwidth; }

  inline llvm::StringRef name() const { return primeName; }

  /// @brief Create a SMT solver symbol with the current field's bitwidth.
  llvm::SMTExprRef createSymbol(llvm::SMTSolverRef solver, const char *name) const {
    return solver->mkSymbol(name, solver->getBitvectorSort(bitWidth()));
  }

  friend bool operator==(const Field &lhs, const Field &rhs) {
    return lhs.primeMod == rhs.primeMod;
  }

private:
  Field(std::string_view primeStr, llvm::StringRef name);

  /// Name of the prime for debugging purposes
  llvm::StringRef primeName;
  llvm::DynamicAPInt primeMod, halfPrime;
  unsigned bitwidth;

  static void initKnownFields(llvm::DenseMap<llvm::StringRef, Field> &knownFields);
};

/// @brief Get a list of primes that the circuit is compatible with.
/// @param modOp a ModuleOp in the circuit. The search for the field attribute begins
/// at this module and continues until a field attribute is encountered.
/// @return Failure if the field attribute is malformed (i.e., is the wrong type of attribute).
/// Otherwise, a list of supported fields if specified. If the returned list is empty,
/// then any field is presumed to be supported.
llvm::FailureOr<llvm::SmallVector<std::reference_wrapper<const Field>>>
getSupportedFields(mlir::ModuleOp modOp);

/// @brief Return true if the list of fields is empty (meaning any field is supported)
/// or the given field is contained within the given list of fields
bool supportsField(
    const llvm::SmallVector<std::reference_wrapper<const Field>> &fields, const Field &f
);

} // namespace llzk
