//===-- OpInterfaces.td ------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_STRUCT_OP_INTERFACES
#define LLZK_STRUCT_OP_INTERFACES

include "llzk/Dialect/Struct/IR/Types.td"

def MemberRefOpInterface : OpInterface<"MemberRefOpInterface"> {
  let description = [{
    Common interface for operations that read or write from a struct member.
  }];
  let cppNamespace = "::llzk::component";

  let methods = [
      // Requires implementors to have a '$val' argument.
      InterfaceMethod<
          [{Gets the SSA Value that holds the read/write data for the MemberRefOp.}],
          "::mlir::Value", "getVal", (ins)>,
      // Requires implementors to have a '$component' argument.
      InterfaceMethod<
          [{Gets the SSA value with the target component from the MemberRefOp.}],
          "::mlir::TypedValue<::llzk::component::StructType>", "getComponent",
          (ins)>,
      InterfaceMethod<
          [{Gets the SSA value with the target component from the MemberRefOp.}],
          "::mlir::OpOperand &", "getComponentMutable", (ins)>,
      // Requires implementors to have a '$member_name' argument.
      InterfaceMethod<
          [{Gets the member name attribute value from the MemberRefOp.}],
          "::llvm::StringRef", "getMemberName", (ins)>,
      InterfaceMethod<[{Gets the member name attribute from the MemberRefOp.}],
                      "::mlir::FlatSymbolRefAttr", "getMemberNameAttr", (ins)>,
      InterfaceMethod<
          [{Sets the member name attribute value in the MemberRefOp.}], "void",
          "setMemberName", (ins "::llvm::StringRef":$attrValue)>,
      InterfaceMethod<[{Sets the member name attribute in the MemberRefOp.}],
                      "void", "setMemberNameAttr",
                      (ins "::mlir::FlatSymbolRefAttr":$attr)>,
      // Requires the '$component' argument type to be LLZK_StructType.
      InterfaceMethod<
          [{Gets the struct type of the target component.}],
          "::llzk::component::StructType", "getStructType", (ins),
          /*methodBody=*/[{ return $_op.getComponent().getType(); }]>,
      InterfaceMethod<
          [{Return `true` if the op is a read, `false` if it's a write.}],
          "bool", "isRead", (ins)>,
  ];

  let extraClassDeclaration = [{
    /// Gets the definition for the `member` referenced in this op.
    ::mlir::FailureOr<SymbolLookupResult<MemberDefOp>> getMemberDefOp(::mlir::SymbolTableCollection &tables);
  }];
}

#endif // LLZK_STRUCT_OP_INTERFACES
