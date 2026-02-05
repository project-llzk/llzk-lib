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
#include "llzk/Util/ErrorHelper.h"

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
  /// @brief Add a new field to the set of available prime fields.
  /// Reports an error if the field is invalid or conflicts with an existing definition.
  inline static void addField(llvm::StringRef fieldName, llvm::APInt prime, EmitErrorFn errFn) {
    return addField(Field(prime, fieldName), errFn);
  }
  inline static void addField(llvm::StringRef fieldName, llvm::StringRef primeStr, EmitErrorFn errFn) {
    return addField(Field(primeStr, fieldName), errFn);
  }

  /// @brief Get a Field from a given field name string, or failure if the
  /// field is not defined.
  /// @param fieldName The name of the field.
  static llvm::FailureOr<std::reference_wrapper<const Field>> tryGetField(llvm::StringRef fieldName);

  /// @brief Get a Field from a given field name string. Throws a fatal error
  /// if the field is unsupported.
  /// @param fieldName The name of the field.
  static const Field &getField(llvm::StringRef fieldName, EmitErrorFn errFn);
  inline static const Field &getField(llvm::StringRef fieldName) {
    return getField(fieldName, nullptr);
  }

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
  Field(llvm::APInt primeInt, llvm::StringRef name);


  /// Name of the prime for debugging purposes
  llvm::StringRef primeName;
  llvm::DynamicAPInt primeMod, halfPrime;
  unsigned bitwidth;

  /// Initialize known prime fields.
  static void initKnownFields();

  /// @brief Add a new field to the set of available prime fields.
  /// @return Failure if the field is invalid or conflicts with an existing definition.
  static void addField(Field &&f, EmitErrorFn errFn);
  inline static void addField(Field &&f) {
    addField(std::move(f), nullptr);
  }
};

/// @brief Update the set of available prime fields with the fields specified on the
/// root module.
/// @param modOp a ModuleOp in the circuit. The search for the field attribute begins
/// at this module and continues until a field attribute is encountered.
/// The operation is recursive as include operations introduce their own root modules,
/// which may include new prime specifications.
/// @return Failure if the field attribute is malformed (i.e., is the wrong type of attribute).
llvm::LogicalResult addSpecifiedFields(mlir::ModuleOp modOp);

} // namespace llzk
