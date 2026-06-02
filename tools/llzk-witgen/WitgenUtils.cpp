//===-- WitgenUtils.cpp - llzk-witgen shared helpers ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "WitgenUtils.h"

#include "Errors.h"

#include "llzk/Util/Compare.h"
#include "llzk/Util/DynamicAPIntHelper.h"

#include <mlir/IR/BuiltinTypes.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/FormatVariadic.h>

#include <algorithm>
#include <climits>
#include <limits>
#include <random>

using namespace mlir;

namespace llzk::witgen {

static llvm::Expected<size_t>
dynamicAPIntToSize(const llvm::DynamicAPInt &value, llvm::Twine context) {
  if (value < 0) {
    return makeError(context + " would underflow size_t");
  }
  llvm::APSInt as = llzk::toAPSInt(value);
  if (as.getActiveBits() > std::numeric_limits<size_t>::digits) {
    return makeError(context + " would overflow size_t");
  }
  return static_cast<size_t>(as.getZExtValue());
}

llvm::Expected<size_t>
checkedDynamicAPIntToSize(const llvm::DynamicAPInt &value, llvm::StringRef context) {
  return dynamicAPIntToSize(value, context);
}

llvm::Expected<size_t> checkedShapeDimToSize(int64_t dim, llvm::StringRef context) {
  if (ShapedType::isDynamic(dim)) {
    return makeError(llvm::Twine(context) + " requires a static shape");
  }
  if (dim < 0) {
    return makeError(llvm::Twine(context) + " has a negative dimension");
  }
  llvm::DynamicAPInt value(dim);
  return dynamicAPIntToSize(value, llvm::Twine(context) + " dimension");
}

llvm::Expected<size_t>
getStaticShapeElementCount(llvm::ArrayRef<int64_t> shape, llvm::StringRef context) {
  llvm::DynamicAPInt count(1);
  for (int64_t dim : shape) {
    auto dimSize = checkedShapeDimToSize(dim, context);
    if (!dimSize) {
      return dimSize.takeError();
    }
    count *= toDynamicAPInt(*dimSize);
  }
  return dynamicAPIntToSize(count, context);
}

llvm::Expected<size_t> getStaticElementCount(ShapedType type, llvm::StringRef context) {
  if (!type.hasStaticShape()) {
    return makeError(llvm::Twine(context) + " requires a static shape");
  }
  int64_t count = type.getNumElements();
  if (count < 0) {
    return makeError(llvm::Twine(context) + " has an invalid negative element count");
  }
  return dynamicAPIntToSize(llvm::DynamicAPInt(count), context);
}

std::mt19937_64 makeDefaultValueRng(const WitgenOptions &options) {
  if (options.randomSeed) {
    return std::mt19937_64(*options.randomSeed);
  }
  std::random_device rd;
  std::seed_seq seed {rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
  return std::mt19937_64(seed);
}

llvm::DynamicAPInt randomFieldElement(std::mt19937_64 &rng, const Field &field) {
  const llvm::DynamicAPInt prime = field.prime();
  if (prime == 0) {
    return prime;
  }

  // Use rejection sampling to produce a uniform value in [0, prime).
  // Generate a random value with the same bit width as the prime; if it is
  // >= prime, discard and retry. The rejection probability is < 50%, so the
  // expected number of iterations is less than 2.
  const unsigned bitWidth = field.bitWidth();
  const unsigned numWords = (bitWidth + 63U) / 64U;
  std::uniform_int_distribution<uint64_t> wordDist;
  llvm::SmallVector<uint64_t, 4> words(numWords);
  while (true) {
    for (uint64_t &word : words) {
      word = wordDist(rng);
    }
    // APInt truncates the top word to exactly bitWidth bits, keeping the
    // candidate in [0, 2^bitWidth).
    llvm::DynamicAPInt value = toDynamicAPInt(llvm::APInt(bitWidth, words));
    if (value < prime) {
      return field.reduce(value);
    }
  }
}

int64_t randomIndexValue(std::mt19937_64 &rng) {
  return std::uniform_int_distribution<
      int64_t>(std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max())(rng);
}

bool randomBoolValue(std::mt19937_64 &rng) {
  return std::uniform_int_distribution<int>(0, 1)(rng) != 0;
}

static std::string renderJSON(const llvm::json::Value &value) {
  return llvm::formatv("{0:2}", value).str();
}

static bool isDecimalString(llvm::StringRef str) {
  if (str.empty()) {
    return false;
  }
  size_t start = 0;
  if (str.front() == '-') {
    if (str.size() == 1) {
      return false;
    }
    start = 1;
  }
  for (size_t i = start; i < str.size(); ++i) {
    if (!llvm::isDigit(str[i])) {
      return false;
    }
  }
  return true;
}

static std::optional<llvm::DynamicAPInt>
parseFieldScalar(const llvm::json::Value &value, const Field &field) {
  if (std::optional<llvm::StringRef> str = value.getAsString()) {
    if (!isDecimalString(*str)) {
      return std::nullopt;
    }
    return field.reduce(llzk::toDynamicAPInt(*str));
  }
  if (std::optional<int64_t> integer = value.getAsInteger()) {
    return field.reduce(*integer);
  }
  return std::nullopt;
}

static bool
shouldUseFieldComparison(const llvm::json::Value &expected, const llvm::json::Value &actual) {
  return expected.getAsString().has_value() || actual.getAsString().has_value();
}

static std::string formatPath(llvm::StringRef path) { return path.empty() ? "<root>" : path.str(); }

static std::string appendObjectPath(llvm::StringRef path, llvm::StringRef key) {
  if (path.empty()) {
    return key.str();
  }
  return path.str() + "." + key.str();
}

static std::string appendArrayPath(llvm::StringRef path, size_t index) {
  return llvm::formatv("{0}[{1}]", formatPath(path), index).str();
}

static void addDiff(
    JSONDiffResult &result, llvm::StringRef path, llvm::StringRef message, size_t maxDifferences
) {
  if (result.entries.size() >= maxDifferences) {
    result.truncated = true;
    return;
  }
  result.entries.push_back(JSONDiffEntry {formatPath(path), message.str()});
}

static void compareJSONValuesImpl(
    const llvm::json::Value &expected, const llvm::json::Value &actual, const Field &field,
    llvm::StringRef path, size_t maxDifferences, JSONDiffResult &result
) {
  if (result.entries.size() >= maxDifferences) {
    result.truncated = true;
    return;
  }

  const auto *expectedObject = expected.getAsObject();
  const auto *actualObject = actual.getAsObject();
  if (expectedObject || actualObject) {
    if (!expectedObject || !actualObject) {
      addDiff(
          result, path,
          llvm::formatv(
              "value mismatch: expected {0}, actual {1}", renderJSON(expected), renderJSON(actual)
          )
              .str(),
          maxDifferences
      );
      return;
    }

    for (const auto &entry : *expectedObject) {
      const llvm::json::Value *actualValue = actualObject->get(entry.first);
      std::string childPath = appendObjectPath(path, entry.first);
      if (!actualValue) {
        addDiff(
            result, childPath,
            llvm::formatv("missing field; expected {0}", renderJSON(entry.second)).str(),
            maxDifferences
        );
        if (result.entries.size() >= maxDifferences) {
          result.truncated = true;
          return;
        }
        continue;
      }
      compareJSONValuesImpl(entry.second, *actualValue, field, childPath, maxDifferences, result);
      if (result.entries.size() >= maxDifferences) {
        result.truncated = true;
        return;
      }
    }

    for (const auto &entry : *actualObject) {
      if (expectedObject->get(entry.first)) {
        continue;
      }
      addDiff(
          result, appendObjectPath(path, entry.first),
          llvm::formatv("unexpected field: actual {0}", renderJSON(entry.second)).str(),
          maxDifferences
      );
      if (result.entries.size() >= maxDifferences) {
        result.truncated = true;
        return;
      }
    }
    return;
  }

  const auto *expectedArray = expected.getAsArray();
  const auto *actualArray = actual.getAsArray();
  if (expectedArray || actualArray) {
    if (!expectedArray || !actualArray) {
      addDiff(
          result, path,
          llvm::formatv(
              "value mismatch: expected {0}, actual {1}", renderJSON(expected), renderJSON(actual)
          )
              .str(),
          maxDifferences
      );
      return;
    }

    size_t commonSize = std::min(expectedArray->size(), actualArray->size());
    for (size_t i = 0; i < commonSize; ++i) {
      compareJSONValuesImpl(
          (*expectedArray)[i], (*actualArray)[i], field, appendArrayPath(path, i), maxDifferences,
          result
      );
      if (result.entries.size() >= maxDifferences) {
        result.truncated = true;
        return;
      }
    }

    for (size_t i = commonSize; i < expectedArray->size(); ++i) {
      addDiff(
          result, appendArrayPath(path, i),
          llvm::formatv("missing element; expected {0}", renderJSON((*expectedArray)[i])).str(),
          maxDifferences
      );
      if (result.entries.size() >= maxDifferences) {
        result.truncated = true;
        return;
      }
    }

    for (size_t i = commonSize; i < actualArray->size(); ++i) {
      addDiff(
          result, appendArrayPath(path, i),
          llvm::formatv("unexpected element: actual {0}", renderJSON((*actualArray)[i])).str(),
          maxDifferences
      );
      if (result.entries.size() >= maxDifferences) {
        result.truncated = true;
        return;
      }
    }
    return;
  }

  if (shouldUseFieldComparison(expected, actual)) {
    std::optional<llvm::DynamicAPInt> expectedField = parseFieldScalar(expected, field);
    std::optional<llvm::DynamicAPInt> actualField = parseFieldScalar(actual, field);
    if (expectedField && actualField) {
      if (*expectedField != *actualField) {
        addDiff(
            result, path,
            llvm::formatv(
                "value mismatch: expected {0}, actual {1}", renderJSON(expected), renderJSON(actual)
            )
                .str(),
            maxDifferences
        );
      }
      return;
    }
  }

  if (std::optional<bool> expectedBool = expected.getAsBoolean()) {
    std::optional<bool> actualBool = actual.getAsBoolean();
    if (!actualBool || *expectedBool != *actualBool) {
      addDiff(
          result, path,
          llvm::formatv(
              "value mismatch: expected {0}, actual {1}", renderJSON(expected), renderJSON(actual)
          )
              .str(),
          maxDifferences
      );
    }
    return;
  }

  if (std::optional<int64_t> expectedInt = expected.getAsInteger()) {
    std::optional<int64_t> actualInt = actual.getAsInteger();
    if (!actualInt || *expectedInt != *actualInt) {
      addDiff(
          result, path,
          llvm::formatv(
              "value mismatch: expected {0}, actual {1}", renderJSON(expected), renderJSON(actual)
          )
              .str(),
          maxDifferences
      );
    }
    return;
  }

  if (std::optional<llvm::StringRef> expectedString = expected.getAsString()) {
    std::optional<llvm::StringRef> actualString = actual.getAsString();
    if (!actualString || *expectedString != *actualString) {
      addDiff(
          result, path,
          llvm::formatv(
              "value mismatch: expected {0}, actual {1}", renderJSON(expected), renderJSON(actual)
          )
              .str(),
          maxDifferences
      );
    }
    return;
  }

  if (renderJSON(expected) != renderJSON(actual)) {
    addDiff(
        result, path,
        llvm::formatv(
            "value mismatch: expected {0}, actual {1}", renderJSON(expected), renderJSON(actual)
        )
            .str(),
        maxDifferences
    );
  }
}

llvm::Expected<JSONDiffResult> compareJSONValues(
    const llvm::json::Value &expected, const llvm::json::Value &actual, const Field &field,
    size_t maxDifferences
) {
  if (maxDifferences == 0) {
    return makeError("maxDifferences must be greater than zero");
  }

  JSONDiffResult result;
  compareJSONValuesImpl(expected, actual, field, "", maxDifferences, result);
  return result;
}

void printJSONDiffReport(llvm::raw_ostream &os, const JSONDiffResult &diff) {
  os << "llzk-witgen output mismatch:\n";
  for (const JSONDiffEntry &entry : diff.entries) {
    os << "  " << entry.path << ": " << entry.message << '\n';
  }
  if (diff.truncated) {
    os << "  ... additional differences omitted\n";
  }
}

} // namespace llzk::witgen
