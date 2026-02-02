//===-- Types.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_SHARED_TYPES
#define LLZK_SHARED_TYPES

include "mlir/IR/AttrTypeBase.td"

def AnyLLZKType : Type<CPred<"::llzk::isValidType($_self)">,
                       "a valid LLZK type", "::mlir::Type">;

def GlobalDefType : Type<CPred<"::llzk::isValidGlobalType($_self)">,
                         "any LLZK type except non-constant types">;

def EmitEqType
    : Type<CPred<"::llzk::isValidEmitEqType($_self)">,
           "any LLZK type, excluding non-Signal struct and string types">;

def ConstReadType : Type<CPred<"::llzk::isValidConstReadType($_self)">,
                         "integral, felt, or typevar type">;

def AnyLLZKIntType : AnyTypeOf<[I1, Index]>;

#endif // LLZK_SHARED_TYPES
