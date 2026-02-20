//===-- Types.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_POD_TYPES
#define LLZK_POD_TYPES

include "llzk/Dialect/Shared/Types.td"
include "llzk/Dialect/POD/IR/Dialect.td"
include "llzk/Dialect/POD/IR/Attrs.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class PODDialectType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<PODDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def LLZK_PODType : PODDialectType<"Pod", "type"> {
  let summary = "plain-old-data struct";
  let description = [{
    Contains a list of `RecordAttr` defining structure of a pod. 
    The order of records in the pod matters and each record must have a unique name.
    It is possible to have a pod with no records.

    Two `PODType` types unify if they have the same record names, defined in the 
    same order, and if the types of corresponding records unify.

    This type can be used in the `struct.member` op, and it can be declared as a column,
    in which case that annotation is also implied on the inner records (except those that
    cannot be column such as `string.type` records).

    ```llzk
    // Type with no records.
    !pod.type

    // Type with two records of felt type.
    !pod.type<[@a: !felt.type, @b: !felt.type]>
    ```
  }];

  let parameters =
      (ins ArrayRefParameter<"::llzk::pod::RecordAttr",
                             "List of records in the pod">:$records);

  let genVerifyDecl = 1;
  let assemblyFormat = "`<` custom<PodType>($records) `>`";

  let extraClassDeclaration = [{
    /// Creates a new type from a set of initialized records. 
    /// 
    /// The records are derived from the names and the types of the values.
    static PodType fromInitialValues(::mlir::MLIRContext *ctx, InitializedRecords init);

    /// Searches a record by name.
    ///
    /// If you plan to make several queries to this type, it's better to generate the map 
    /// once via `getRecordMap()` and query it instead of repeatedly calling `getRecord()`.
    ::llvm::FailureOr<::mlir::Type> getRecord(::llvm::StringRef name, ::llvm::function_ref<::mlir::InFlightDiagnostic()>) const;

    /// Returns the records in map form.
    /// 
    /// If you plan to make several queries to this type, it's better to generate the map 
    /// once via `getRecordMap()` and query it instead of repeatedly calling `getRecord()`.
    ::llvm::StringMap<::mlir::Type> getRecordMap() const;
  }];
}

#endif // LLZK_POD_TYPES
