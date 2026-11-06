//===-- ExposeChildMembersPass.h ------------------------------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Struct/IR/Ops.h"

#include <mlir/Support/LogicalResult.h>

namespace llzk::component {

/// Make reachable concrete child members and matching rolled family declarations
/// public while preserving root's visibility and all signal annotations.
/// Root must be attached to an LLZK module, with templates monomorphized and includes
/// inlined. Emit diagnostics on failure without changing member visibility.
mlir::LogicalResult exposeChildMembers(StructDefOp root);

} // namespace llzk::component
