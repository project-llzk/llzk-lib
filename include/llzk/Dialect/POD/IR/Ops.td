//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_POD_OPS
#define LLZK_POD_OPS

include "llzk/Dialect/POD/IR/Dialect.td"
include "llzk/Dialect/POD/IR/Types.td"
include "llzk/Dialect/Shared/OpTraits.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class PODDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<PODDialect, mnemonic, traits>;

def LLZK_NewPodOp
    : PODDialectOp<"new", [Pure, AttrSizedOperandSegments,
                           VerifySizesForMultiAffineOps<1>,
                           DeclareOpInterfaceMethods<
                               OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "create a new plain-old-data struct";
  let description = [{
    Creates a new, uninitialized, pod instance. Optionally, the user can pass a list of record names and values 
    that initialize the records of the pod. Partial initialization is allowed. All records without 
    an explicit initialization are initialized with nondeterministic values.

    If the types of the pod records have affine map parameters the user can pass values for them similar 
    to how `array.new` does it. 

    This operation returns one value of type `PODType` and, if present, the records passed for initialization
    must form a subset of the records in the type.

    Examples:
    ```llzk 
    // Empty pod instance
    %0 = pod.new : !pod.type<[]>

    // Uninitialized/nondeterministic pod instance
    %0 = pod.new : !pod.type<[@n: !felt.type]>

    // Initialized pod instance
    %0 = felt.const_felt 1 
    %1 = pod.new { @n = %0 } : !pod.type<[@n: !felt.type]>

    // Another one, but with 2 fields
    %0 = felt.const_felt 1
    %1 = felt.inv %0
    %2 = pod.new { @n = %0, @inv = %1 } :	!pod.type<[@n: !felt.type, @inv: !felt.type]>

    // Partial init
    %0 = felt.const_felt 1
    %1 = pod.new { @n = %0 } : !pod.type<[@n: !felt.type, @inv: !felt.type]>

    // Affine map args on uninitialized pod instance
    %0 = arith.constant 1 : index
    %c = arith.constant 2 : index
    %1 = pod.new(%0)[%c] : !pod.type<[@a: !array.type<#map, !felt.type>]>

    // Affine map with initialized records
    %0 = arith.constant 1 : index
    %c = arith.constant 2 : index
    %1 = felt.const_felt 5 
    %2 = pod.new { @f = %1 }(%0)[%c] : !pod.type<[@f: !felt.type, @a: !array.type<#map, !felt.type>]>
    ```
  }];

  let arguments = (ins
      // Initialization values
      Variadic<AnyLLZKType>:$initialValues,
      DefaultValuedAttr<StrArrayAttr, "{}">:$initializedRecords,
      // Affine map arguments
      VariadicOfVariadic<Index, "mapOpGroupSizes">:$mapOperands,
      DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$numDimsPerMap,
      DenseI32ArrayAttr:$mapOpGroupSizes);
  let results = (outs LLZK_PODType:$result);
  let skipDefaultBuilders = 1;
  let builders =
      [OpBuilder<
           (ins CArg<"::llzk::pod::InitializedRecords", "{}">:$initialValues),
           [{
        auto resultType = ::llzk::pod::PodType::fromInitialValues($_builder.getContext(), initialValues);
        build($_builder, $_state, resultType, initialValues);
      }]>,
       OpBuilder<(ins "::llzk::pod::PodType":$resultType,
           CArg<"::llzk::pod::InitializedRecords", "{}">:$initialValues)>,
       OpBuilder<(ins "::llzk::pod::PodType":$resultType,
           "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
           "::mlir::DenseI32ArrayAttr":$numDimsPerMap,
           CArg<"::llzk::pod::InitializedRecords", "{}">:$initialValues)>,
       OpBuilder<
           (ins "::llzk::pod::PodType":$resultType,
               "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
               "::llvm::ArrayRef<int32_t>":$numDimsPerMap,
               CArg<"::llzk::pod::InitializedRecords", "{}">:$initialValues),
           [{
      build($_builder, $_state, resultType, mapOperands,
         $_builder.getDenseI32ArrayAttr(numDimsPerMap), initialValues);
    }]>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::SmallVector<::llzk::pod::RecordValue> getInitializedRecordValues();
  }];
}

def LLZK_ReadPodOp : PODDialectOp<"read"> {
  let summary = "reads the contents of a plain-old-data struct record";
  let description = [{
    Reads the current value of a named record from a pod instance. 
    Returns one value of the type of the record.

    The name of the record must be a valid record name for the pod type and the result 
    type must match the type of the record.

    Example:
    ```llzk
    %1 = pod.read %0[@sym] : !pod.type<[@sym: !type, ...]>, !type
    ```
  }];

  let arguments = (ins Arg<LLZK_PODType, "the pod to read from">:$pod_ref,
      FlatSymbolRefAttr:$record_name);
  let results = (outs AnyLLZKType:$result);
  let assemblyFormat = [{
    $pod_ref `[` custom<RecordName>($record_name) `]` `:` type($pod_ref) `,` type($result) attr-dict
  }];
  let hasVerifier = 1;
}

def LLZK_WritePodOp : PODDialectOp<"write"> {
  let summary = "writes content into a plain-old-data struct record";
  let description = [{
    Writes a value into a named record of a pod instance. This operation has no result values.

    The name of the record must be a valid record name for the pod type and the source 
    value type must match the type of the record.

    Writing a record that was written or initialized earlier overwrites the previous value.

    Example:
    ```llzk
    pod.write %0[@sym] = %1 : !pod.type<[@sym: !type, ...]>, !type
    ```
  }];

  let arguments = (ins Arg<LLZK_PODType, "the pod to write into">:$pod_ref,
      FlatSymbolRefAttr:$record_name, AnyLLZKType:$value);

  let assemblyFormat = [{
    $pod_ref `[` custom<RecordName>($record_name) `]` `=` $value `:` type($pod_ref) `,` type($value) attr-dict
  }];
  let hasVerifier = 1;
}

#endif // LLZK_POD_OPS
