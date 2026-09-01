//===-- TypeConverter.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Transforms/DialectConversion.h>

namespace pcl::lowering {

/// Type converter from LLZK types to PCL.
struct PCLTypeConverter : public mlir::TypeConverter {
  PCLTypeConverter();
};
} // namespace pcl::lowering
