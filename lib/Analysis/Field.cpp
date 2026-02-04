//===-- Field.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/Field.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/DynamicAPIntHelper.h"

#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/SlowDynamicAPInt.h>
#include <llvm/ADT/Twine.h>
#include <llvm/ADT/TypeSwitch.h>

#include <algorithm>
#include <mutex>

using namespace llvm;

namespace llzk {

Field::Field(std::string_view primeStr, llvm::StringRef name) : primeName(name) {
  APSInt parsedInt(primeStr);

  primeMod = toDynamicAPInt(parsedInt);
  halfPrime = (primeMod + felt(1)) / felt(2);
  bitwidth = parsedInt.getBitWidth();
}

FailureOr<std::reference_wrapper<const Field>> Field::tryGetField(const char *fieldName) {
  static DenseMap<StringRef, Field> knownFields;
  static std::once_flag fieldsInit;
  std::call_once(fieldsInit, initKnownFields, knownFields);

  if (auto it = knownFields.find(fieldName); it != knownFields.end()) {
    return {it->second};
  }
  return failure();
}

const Field &Field::getField(const char *fieldName) {
  auto res = tryGetField(fieldName);
  if (mlir::failed(res)) {
    report_fatal_error("field \"" + Twine(fieldName) + "\" is unsupported");
  }
  return res.value().get();
}

void Field::initKnownFields(DenseMap<StringRef, Field> &knownFields) {
  static constexpr const char BN128[] = "bn128", BN254[] = "bn254", BABYBEAR[] = "babybear",
                              GOLDILOCKS[] = "goldilocks", MERSENNE31[] = "mersenne31",
                              KOALABEAR[] = "koalabear";
  // bn128/254, default for circom
  knownFields.try_emplace(
      BN128,
      Field("21888242871839275222246405745257275088696311157297823662689037894645226208583", BN128)
  );
  knownFields.try_emplace(
      BN254,
      Field("21888242871839275222246405745257275088696311157297823662689037894645226208583", BN254)
  );
  // 15 * 2^27 + 1, default for zirgen
  knownFields.try_emplace(BABYBEAR, Field("2013265921", BABYBEAR));
  // 2^64 - 2^32 + 1, used for plonky2
  knownFields.try_emplace(GOLDILOCKS, Field("18446744069414584321", GOLDILOCKS));
  // 2^31 - 1, used for Plonky3
  knownFields.try_emplace(MERSENNE31, Field("2147483647", MERSENNE31));
  // 2^31 - 2^24 + 1, also for Plonky3
  knownFields.try_emplace(KOALABEAR, Field("2130706433", KOALABEAR));
}

DynamicAPInt Field::reduce(const DynamicAPInt &i) const {
  DynamicAPInt m = i % prime();
  if (m < 0) {
    return prime() + m;
  }
  return m;
}

DynamicAPInt Field::reduce(const APInt &i) const { return reduce(toDynamicAPInt(i)); }

DynamicAPInt Field::inv(const DynamicAPInt &i) const { return modInversePrime(i, prime()); }

DynamicAPInt Field::inv(const llvm::APInt &i) const {
  return modInversePrime(toDynamicAPInt(i), prime());
}

// Parses Fields from the given attribute, if able.
static FailureOr<SmallVector<std::reference_wrapper<const Field>>> parseFields(mlir::Attribute a) {
  return llvm::TypeSwitch<
             mlir::Attribute, FailureOr<SmallVector<std::reference_wrapper<const Field>>>>(a)
      .Case<mlir::UnitAttr>([](auto _) {
    return SmallVector<std::reference_wrapper<const Field>> {};
  })
      .Case<mlir::StringAttr>(
          [](auto s) -> FailureOr<SmallVector<std::reference_wrapper<const Field>>> {
    auto fieldRes = Field::tryGetField(s.getValue().data());
    if (mlir::failed(fieldRes)) {
      return failure();
    }
    return SmallVector<std::reference_wrapper<const Field>> {fieldRes.value()};
  }
      )
      .Case<mlir::ArrayAttr>(
          [](auto arr) -> FailureOr<SmallVector<std::reference_wrapper<const Field>>> {
    // An ArrayAttr may only contain inner StringAttr
    SmallVector<std::reference_wrapper<const Field>> res;
    for (mlir::Attribute a : arr) {
      if (auto s = llvm::dyn_cast<mlir::StringAttr>(a)) {
        auto fieldRes = Field::tryGetField(s.getValue().data());
        if (mlir::failed(fieldRes)) {
          return failure();
        }
        res.push_back(fieldRes.value());
      } else {
        return failure();
      }
    }
    return res;
  }
      )
      .Default([](auto _) { return failure(); });
}

FailureOr<SmallVector<std::reference_wrapper<const Field>>>
getSupportedFields(mlir::ModuleOp modOp) {
  if (mlir::Attribute a = modOp->getAttr(FIELD_ATTR_NAME)) {
    return parseFields(a);
  } else if (mlir::ModuleOp parentMod = modOp->getParentOfType<mlir::ModuleOp>()) {
    return getSupportedFields(parentMod);
  }
  return SmallVector<std::reference_wrapper<const Field>> {};
}

bool supportsField(const SmallVector<std::reference_wrapper<const Field>> &fields, const Field &f) {
  return fields.empty() || std::find(fields.begin(), fields.end(), f) != fields.end();
}

} // namespace llzk
