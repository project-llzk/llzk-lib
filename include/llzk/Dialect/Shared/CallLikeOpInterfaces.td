//===-- CallLikeOpInterfaces.td ----------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_CALL_LIKE_OP_INTERFACES
#define LLZK_CALL_LIKE_OP_INTERFACES

include "llzk/Dialect/Shared/OpTraits.td"

include "mlir/IR/Interfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"

def LLZKCallLikeOpInterface
    : OpInterface<
          "LLZKCallLikeOpInterface",
          [DeclareOpInterfaceMethods<
               CallOpInterface, ["resolveCallable", "resolveCallableInTable"]>,
           DeclareOpInterfaceMethods<
               SymbolUserOpInterface, ["verifySymbolUses"]>]> {
  let cppNamespace = "::llzk";
  let methods =
      [InterfaceMethod<
           [{Returns the "self" value from the callee (or nullptr if it's not a struct target).}],
           "::mlir::Value", "getSelfValue">,
       InterfaceMethod<
           [{Returns the function signature inferred from this call site.}],
           "::mlir::FunctionType", "getTypeSignature", (ins), /*methodBody=*/"",
           /*defaultImplementation=*/[{ return ::mlir::FunctionType::get(
                                $_op.getContext(),
                                $_op.getArgOperands().getTypes(),
                                $_op->getResultTypes());
                            }]>,
       InterfaceMethod<
           [{Unifies this call site's inferred signature with `other`.}],
           "::mlir::FailureOr<::llzk::UnificationMap>", "unifyTypeSignature",
           (ins "::mlir::FunctionType":$other), /*methodBody=*/"",
           /*defaultImplementation=*/[{
                ::llzk::UnificationMap unifications;
                if (::llzk::functionTypesUnify($_op.getTypeSignature(), other, {}, &unifications)) {
                    return unifications;
                } else {
                    return ::mlir::failure();
                }
           }]>,
       StaticInterfaceMethod<
           [{Allocates consecutive storage for operand groups so they can be passed to builders
             as an `ArrayRef<ValueRange>`.}],
           "::llvm::SmallVector<::mlir::ValueRange>", "toVectorOfValueRange",
           (ins "::mlir::OperandRangeRange":$input), /*methodBody=*/"",
           /*defaultImplementation=*/[{
                ::llvm::SmallVector<::mlir::ValueRange, 4> output;
                output.reserve(input.size());
                output.insert(output.end(), input.begin(), input.end());
                return output;
           }]>,
       InterfaceMethod<
           [{Checks a template parameter value against the declared type of its target parameter.}],
           "::mlir::LogicalResult", "verifyTemplateParamValueCompatibility",
           (ins "::mlir::Attribute":$paramFromCallOp,
               "::llzk::polymorphic::TemplateParamOp":$targetParam),
           /*methodBody=*/"",
           /*defaultImplementation=*/[{
                return ::llzk::verifyTemplateParamValueCompatibility(
                    $_op.getOperation(), paramFromCallOp, targetParam);
           }]>,
       InterfaceMethod<
           [{Checks each explicit template parameter against its corresponding declared parameter.

             Pre-condition assertions:
               - `!isNullOrEmpty(getTemplateParamsAttr())`
               - `getTemplateParamsAttr().size() == llvm::range_size(targetParamDefs)`}],
           "::mlir::LogicalResult", "verifyTemplateParamValuesCompatibility",
           (ins "::llvm::iterator_range<::mlir::Region::op_iterator<::llzk::"
                "polymorphic::TemplateParamOp>>":$targetParamDefs),
           /*methodBody=*/"",
           /*defaultImplementation=*/[{
                return ::llzk::verifyTemplateParamValuesCompatibility(
                    $_op.getOperation(), $_op.getTemplateParamsAttr(), targetParamDefs);
           }]>,
       InterfaceMethod<
           [{Checks that explicit template parameters agree with their inferred values.

             Pre-condition assertions:
               - `!isNullOrEmpty(getTemplateParamsAttr())`
               - `getTemplateParamsAttr().size() == llvm::range_size(targetParamDefs)`}],
           "::mlir::LogicalResult", "verifyTemplateParamsMatchInferred",
           (ins "::llvm::iterator_range<::mlir::Region::op_iterator<::llzk::"
                "polymorphic::TemplateParamOp>>":$targetParamDefs,
               "const ::llzk::UnificationMap &":$unifications),
           /*methodBody=*/"",
           /*defaultImplementation=*/[{
                return ::llzk::verifyTemplateParamsMatchInferred(
                    $_op.getOperation(), $_op.getTemplateParamsAttr(), targetParamDefs, unifications);
           }]>];
}

#endif // LLZK_CALL_LIKE_OP_INTERFACES
