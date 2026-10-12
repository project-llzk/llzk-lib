//===-- ExposeChildMembersPass.h -------------------------------------------===//
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

/// Add llzk.pub to members of concrete children reachable through root's storage,
/// including arrays, PODs, and rolled families. Arrays are followed by element
/// type; rolled storage visits all matching family specializations without
/// enumerating coordinates or validating complete array coverage. Root must be a monomorphized
/// definition attached to an LLZK module; includes must be inlined. Root's own
/// attributes, source templates, and unreachable definitions remain unchanged.
/// Preserve signal annotations, storage types, and function bodies. Diagnose
/// unresolved referenced families without changing any members on failure.
/// Repeated calls have no additional effect. Child visibility does not imply
/// public circuit signals. This utility does not require llzk.main metadata.
mlir::LogicalResult exposeChildMembers(StructDefOp root);

} // namespace llzk::component
