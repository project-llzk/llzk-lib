//===-- ZKBuilderTypes.td ----------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DIALECT_ZKBUILDER_ZKBUILDERTYPES_TD_
#define LIB_DIALECT_ZKBUILDER_ZKBUILDERTYPES_TD_

include "ZKBuilderDialect.td"
include "mlir/IR/AttrTypeBase.td"

def ZKBuilderState : TypeDef<ZKBuilder_Dialect, "ZKBuilderState"> {
  let summary = "State of the ZKBuilder";
  let mnemonic = "zkbuilderstate";
}

#endif  // LIB_DIALECT_ZKBUILDER_ZKBUILDERTYPES_TD_
