//===-- LLZKTestUtils.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/Debug.h"
#include "llzk/Util/DynamicAPIntHelper.h"

#include <gtest/gtest.h>

/// @brief Check a given condition, outputing an error using the debug::Appender
/// if `cond` is false. Use this wrapper for checks on type `T` where `T` cannot
/// be written directly to std C++ ostreams.
template <typename T>
[[maybe_unused]] static testing::AssertionResult
checkCond(const T &expected, const T &actual, bool cond) {
  if (cond) {
    return testing::AssertionSuccess();
  }
  std::string errMsg;
  llzk::debug::Appender(errMsg) << "expected " << expected << ", actual is " << actual;
  return testing::AssertionFailure() << errMsg;
}

namespace llvm {

/// @brief GoogleTest printer for DynamicAPInt to control how parameter values are displayed.
/// It cannot contain spaces, dashes, or any non-alphanumeric characters other than underscores.
/// This prevents GoogleTest from printing the raw memory representation of DynamicAPInt objects.
inline void PrintTo(const DynamicAPInt &val, std::ostream *os) {
  if (val < 0) {
    *os << "neg_" << llzk::debug::toStringOne(-val);
  } else {
    *os << llzk::debug::toStringOne(val);
  }
}

} // namespace llvm
