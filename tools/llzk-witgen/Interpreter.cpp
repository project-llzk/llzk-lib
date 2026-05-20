//===-- Interpreter.cpp - llzk-witgen compute interpreter -------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"

#include "Errors.h"
#include "WitgenUtils.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/SymbolLookup.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include <limits>
#include <random>

using namespace mlir;

namespace llzk::witgen {

namespace {

/// Return whether the loop should compare its bounds as unsigned integers.
static bool usesUnsignedCmp(scf::ForOp forOp) {
  if (auto boolAttr = forOp->getAttrOfType<BoolAttr>("unsignedCmp")) {
    return boolAttr.getValue();
  }
  return forOp->hasAttr("unsignedCmp");
}

/// Convert an unsigned intermediate back to `int64_t`, rejecting underflow.
static llvm::Expected<int64_t> toCheckedInt64(uint64_t value) {
  if (value > llzk::checkedCast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return makeError("unsigned value does not fit in signed int64_t");
  }
  return llzk::checkedCast<int64_t>(value);
}

/// Represent the values yielded by a block or region along with termination state.
struct BlockResult {
  bool terminated = false;
  llvm::SmallVector<WitnessVal> values;
};

/// Validate array indices and flatten them with MLIR's row-major helper.
llvm::Expected<size_t> checkedLinearize(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> indices, llvm::StringRef context
) {
  if (shape.size() != indices.size()) {
    return makeError("wrong number of array indices");
  }
  for (auto [idx, dim] : llvm::zip_equal(indices, shape)) {
    if (idx < 0 || dim < 0 || idx >= dim) {
      return makeError(context);
    }
  }
  auto strides = mlir::computeStrides(shape);
  return llzk::checkedCast<size_t>(mlir::linearize(indices, strides));
}

} // namespace

/// Build an interpreter for a specific module and field.
FunctionInterpreter::FunctionInterpreter(
    ModuleOp module, SymbolTableCollection &symbolTables, const Field &moduleField,
    UninitializedBehavior behavior, std::mt19937_64 r
)
    : moduleOp(module), tables(symbolTables), field(moduleField), uninitializedBehavior(behavior),
      rng(r) {}

namespace {

/// Execute one function invocation over a mutable SSA environment.
class InvocationInterpreter {
public:
  /// Create an invocation interpreter that shares module-level state.
  InvocationInterpreter(
      ModuleOp module, SymbolTableCollection &symbolTables, const Field &moduleField,
      UninitializedBehavior behavior, std::mt19937_64 &r
  )
      : moduleOp(module), tables(symbolTables), field(moduleField), uninitializedBehavior(behavior),
        rng(r) {}

  /// Execute a function body with the provided arguments.
  llvm::Expected<llvm::SmallVector<WitnessVal>>
  run(function::FuncDefOp funcOp, ArrayRef<WitnessVal> args) {
    if (funcOp.isExternal()) {
      return makeError("extern functions are not supported in llzk-witgen");
    }
    if (!funcOp.getBody().hasOneBlock()) {
      return makeError("multi-block functions are not supported in llzk-witgen");
    }
    if (funcOp.getNumArguments() != args.size()) {
      return makeError("wrong number of arguments passed to function");
    }

    llvm::DenseMap<mlir::Value, WitnessVal> scope;
    Block &entry = funcOp.getBody().front();
    for (auto [arg, value] : llvm::zip(entry.getArguments(), args)) {
      scope[arg] = value;
    }

    auto result = runBlock(entry, scope);
    if (!result) {
      return result.takeError();
    }
    return result->values;
  }

private:
  ModuleOp moduleOp;
  SymbolTableCollection &tables;
  const Field &field;
  UninitializedBehavior uninitializedBehavior;
  std::mt19937_64 &rng;

  /// Execute every operation in a block until termination or fallthrough.
  llvm::Expected<BlockResult>
  runBlock(Block &block, llvm::DenseMap<mlir::Value, WitnessVal> &scope) {
    for (Operation &op : block) {
      auto handled = runOperation(op, scope);
      if (!handled) {
        return handled.takeError();
      }
      if (handled->terminated) {
        return *handled;
      }
    }
    return BlockResult {};
  }

  /// Execute a single-block region with explicit block arguments.
  llvm::Expected<BlockResult> runRegion(
      Region &region, ArrayRef<WitnessVal> args, llvm::DenseMap<mlir::Value, WitnessVal> scope
  ) {
    if (!region.hasOneBlock()) {
      return makeError("multi-block regions are not supported in llzk-witgen");
    }
    Block &block = region.front();
    if (block.getNumArguments() != args.size()) {
      return makeError("region argument count mismatch");
    }
    for (auto [arg, value] : llvm::zip(block.getArguments(), args)) {
      scope[arg] = value;
    }
    return runBlock(block, scope);
  }

  /// Look up the runtime value bound to an SSA value.
  llvm::Expected<WitnessVal>
  lookup(mlir::Value value, llvm::DenseMap<mlir::Value, WitnessVal> &scope) {
    auto it = scope.find(value);
    if (it == scope.end()) {
      return makeError("failed to find SSA value during interpretation");
    }
    return it->second;
  }

  /// Materialize operand values for an operation in source order.
  llvm::Expected<llvm::SmallVector<WitnessVal>>
  collectOperands(OperandRange operands, llvm::DenseMap<mlir::Value, WitnessVal> &scope) {
    llvm::SmallVector<WitnessVal> values;
    values.reserve(operands.size());
    for (mlir::Value operand : operands) {
      auto value = lookup(operand, scope);
      if (!value) {
        return value.takeError();
      }
      values.push_back(*value);
    }
    return values;
  }

  /// Execute one supported operation and bind its result values.
  llvm::Expected<BlockResult>
  runOperation(Operation &op, llvm::DenseMap<mlir::Value, WitnessVal> &scope) {
    if (auto returnOp = dyn_cast<function::ReturnOp>(op)) {
      auto values = collectOperands(returnOp.getOperands(), scope);
      if (!values) {
        return values.takeError();
      }
      return BlockResult {true, std::move(*values)};
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      auto values = collectOperands(yieldOp.getOperands(), scope);
      if (!values) {
        return values.takeError();
      }
      return BlockResult {true, std::move(*values)};
    }
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(op)) {
      auto values = collectOperands(conditionOp.getOperands(), scope);
      if (!values) {
        return values.takeError();
      }
      return BlockResult {true, std::move(*values)};
    }

    auto bind = [&](ArrayRef<WitnessVal> results) -> llvm::Expected<BlockResult> {
      if (results.size() != op.getNumResults()) {
        return makeError("internal result count mismatch");
      }
      for (auto [result, value] : llvm::zip(op.getResults(), results)) {
        scope[result] = value;
      }
      return BlockResult {};
    };

    if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
      Attribute valueAttr = constantOp.getValue();
      if (auto integerAttr = dyn_cast<IntegerAttr>(valueAttr)) {
        if (integerAttr.getType().isInteger(1)) {
          return bind({WitnessVal(integerAttr.getValue().getBoolValue())});
        }
        return bind({WitnessVal(integerAttr.getValue().getSExtValue())});
      }
      return makeError("unsupported arith.constant value");
    }

    if (auto nondetOp = dyn_cast<llzk::NonDetOp>(op)) {
      auto value = defaultValue(
          nondetOp.getType(), tables, nondetOp.getOperation(), field, uninitializedBehavior, &rng
      );
      if (!value) {
        return value.takeError();
      }
      return bind({*value});
    }

    if (auto assertOp = dyn_cast<boolean::AssertOp>(op)) {
      auto condition = lookup(assertOp.getCondition(), scope);
      if (!condition) {
        return condition.takeError();
      }
      auto boolValue = asBool(*condition);
      if (!boolValue) {
        return boolValue.takeError();
      }
      if (!*boolValue) {
        std::string msg = "bool.assert failed";
        if (auto attr = assertOp.getMsg()) {
          msg = attr->str();
        }
        return makeError(msg);
      }
      return BlockResult {};
    }

    if (auto andOp = dyn_cast<boolean::AndBoolOp>(op)) {
      auto lhsValue = lookup(andOp.getLhs(), scope);
      auto rhsValue = lookup(andOp.getRhs(), scope);
      if (!lhsValue) {
        return lhsValue.takeError();
      }
      if (!rhsValue) {
        return rhsValue.takeError();
      }
      auto lhs = asBool(*lhsValue);
      if (!lhs) {
        return lhs.takeError();
      }
      auto rhs = asBool(*rhsValue);
      if (!rhs) {
        return rhs.takeError();
      }
      return bind({WitnessVal(*lhs && *rhs)});
    }
    if (auto orOp = dyn_cast<boolean::OrBoolOp>(op)) {
      auto lhsValue = lookup(orOp.getLhs(), scope);
      auto rhsValue = lookup(orOp.getRhs(), scope);
      if (!lhsValue) {
        return lhsValue.takeError();
      }
      if (!rhsValue) {
        return rhsValue.takeError();
      }
      auto lhs = asBool(*lhsValue);
      if (!lhs) {
        return lhs.takeError();
      }
      auto rhs = asBool(*rhsValue);
      if (!rhs) {
        return rhs.takeError();
      }
      return bind({WitnessVal(*lhs || *rhs)});
    }
    if (auto xorOp = dyn_cast<boolean::XorBoolOp>(op)) {
      auto lhsValue = lookup(xorOp.getLhs(), scope);
      auto rhsValue = lookup(xorOp.getRhs(), scope);
      if (!lhsValue) {
        return lhsValue.takeError();
      }
      if (!rhsValue) {
        return rhsValue.takeError();
      }
      auto lhs = asBool(*lhsValue);
      if (!lhs) {
        return lhs.takeError();
      }
      auto rhs = asBool(*rhsValue);
      if (!rhs) {
        return rhs.takeError();
      }
      return bind({WitnessVal(*lhs != *rhs)});
    }
    if (auto notOp = dyn_cast<boolean::NotBoolOp>(op)) {
      auto operand = lookup(notOp.getOperand(), scope);
      if (!operand) {
        return operand.takeError();
      }
      auto boolValue = asBool(*operand);
      if (!boolValue) {
        return boolValue.takeError();
      }
      return bind({WitnessVal(!*boolValue)});
    }
    if (auto cmpOp = dyn_cast<boolean::CmpOp>(op)) {
      auto lhs = lookup(cmpOp.getLhs(), scope);
      auto rhs = lookup(cmpOp.getRhs(), scope);
      if (!lhs) {
        return lhs.takeError();
      }
      if (!rhs) {
        return rhs.takeError();
      }
      auto lhsValue = asFelt(*lhs);
      if (!lhsValue) {
        return lhsValue.takeError();
      }
      auto rhsValue = asFelt(*rhs);
      if (!rhsValue) {
        return rhsValue.takeError();
      }
      bool result = false;
      switch (cmpOp.getPredicate()) {
      case boolean::FeltCmpPredicate::EQ:
        result = *lhsValue == *rhsValue;
        break;
      case boolean::FeltCmpPredicate::NE:
        result = *lhsValue != *rhsValue;
        break;
      case boolean::FeltCmpPredicate::LT:
        result = *lhsValue < *rhsValue;
        break;
      case boolean::FeltCmpPredicate::LE:
        result = *lhsValue <= *rhsValue;
        break;
      case boolean::FeltCmpPredicate::GT:
        result = *lhsValue > *rhsValue;
        break;
      case boolean::FeltCmpPredicate::GE:
        result = *lhsValue >= *rhsValue;
        break;
      }
      return bind({WitnessVal(result)});
    }

    if (auto feltConst = dyn_cast<felt::FeltConstantOp>(op)) {
      return bind({WitnessVal(field.reduce(feltConst.getValue().getValue()))});
    }

    auto handleBinaryFelt = [&](auto feltOp, auto fn) -> llvm::Expected<BlockResult> {
      auto lhsValue = lookup(feltOp.getLhs(), scope);
      auto rhsValue = lookup(feltOp.getRhs(), scope);
      if (!lhsValue) {
        return lhsValue.takeError();
      }
      if (!rhsValue) {
        return rhsValue.takeError();
      }
      auto lhs = asFelt(*lhsValue);
      if (!lhs) {
        return lhs.takeError();
      }
      auto rhs = asFelt(*rhsValue);
      if (!rhs) {
        return rhs.takeError();
      }
      return bind({WitnessVal(field.reduce(fn(*lhs, *rhs)))});
    };

    if (auto addOp = dyn_cast<felt::AddFeltOp>(op)) {
      return handleBinaryFelt(addOp, [](const auto &lhs, const auto &rhs) { return lhs + rhs; });
    }
    if (auto subOp = dyn_cast<felt::SubFeltOp>(op)) {
      return handleBinaryFelt(subOp, [](const auto &lhs, const auto &rhs) { return lhs - rhs; });
    }
    if (auto mulOp = dyn_cast<felt::MulFeltOp>(op)) {
      return handleBinaryFelt(mulOp, [](const auto &lhs, const auto &rhs) { return lhs * rhs; });
    }
    if (auto divOp = dyn_cast<felt::DivFeltOp>(op)) {
      return handleBinaryFelt(divOp, [&](const auto &lhs, const auto &rhs) {
        return lhs * field.inv(rhs);
      });
    }
    if (auto negOp = dyn_cast<felt::NegFeltOp>(op)) {
      auto operand = lookup(negOp.getOperand(), scope);
      if (!operand) {
        return operand.takeError();
      }
      auto feltValue = asFelt(*operand);
      if (!feltValue) {
        return feltValue.takeError();
      }
      return bind({WitnessVal(field.reduce(-*feltValue))});
    }
    if (auto invOp = dyn_cast<felt::InvFeltOp>(op)) {
      auto operand = lookup(invOp.getOperand(), scope);
      if (!operand) {
        return operand.takeError();
      }
      auto feltValue = asFelt(*operand);
      if (!feltValue) {
        return feltValue.takeError();
      }
      return bind({WitnessVal(field.inv(*feltValue))});
    }
    // Reduces signed integers to unsigned field elements using Field::reduce.
    // Negative results are reduced by subtracting from the prime (e.g., -1 -> p - 1).
    if (auto intToFeltOp = dyn_cast<cast::IntToFeltOp>(op)) {
      auto operand = lookup(intToFeltOp.getValue(), scope);
      if (!operand) {
        return operand.takeError();
      }
      if (std::holds_alternative<bool>(*operand)) {
        return bind({WitnessVal(field.reduce(std::get<bool>(*operand) ? 1 : 0))});
      }
      auto integer = asIndex(*operand);
      if (!integer) {
        return integer.takeError();
      }
      return bind({WitnessVal(field.reduce(*integer))});
    }
    // Field elements are unsigned. If the field element would overflow the 64-bit
    // index, an error is reported.
    if (auto feltToIndexOp = dyn_cast<cast::FeltToIndexOp>(op)) {
      auto operand = lookup(feltToIndexOp.getValue(), scope);
      if (!operand) {
        return operand.takeError();
      }
      auto feltValue = asFelt(*operand);
      if (!feltValue) {
        return feltValue.takeError();
      }
      auto &felt = *feltValue;
      if (felt < 0 || felt > std::numeric_limits<int64_t>::max()) {
        return makeError("felt value does not fit in index");
      }
      return bind({WitnessVal(int64_t(felt))});
    }

    if (auto structNewOp = dyn_cast<component::CreateStructOp>(op)) {
      auto value = defaultValue(
          structNewOp.getType(), tables, structNewOp.getOperation(), field, uninitializedBehavior,
          &rng
      );
      if (!value) {
        return value.takeError();
      }
      return bind({*value});
    }
    if (auto readMemberOp = dyn_cast<component::MemberReadOp>(op)) {
      auto componentValue = lookup(readMemberOp.getComponent(), scope);
      if (!componentValue) {
        return componentValue.takeError();
      }
      auto structValue = asStruct(*componentValue);
      if (!structValue) {
        return structValue.takeError();
      }
      auto it = (*structValue)->members.find(readMemberOp.getMemberName());
      if (it == (*structValue)->members.end()) {
        return makeError("missing struct member");
      }
      return bind({it->second});
    }
    if (auto writeMemberOp = dyn_cast<component::MemberWriteOp>(op)) {
      auto componentValue = lookup(writeMemberOp.getComponent(), scope);
      auto memberValue = lookup(writeMemberOp.getVal(), scope);
      if (!componentValue) {
        return componentValue.takeError();
      }
      if (!memberValue) {
        return memberValue.takeError();
      }
      auto structValue = asStruct(*componentValue);
      if (!structValue) {
        return structValue.takeError();
      }
      (*structValue)->members[writeMemberOp.getMemberName()] = *memberValue;
      return BlockResult {};
    }

    if (auto newPodOp = dyn_cast<pod::NewPodOp>(op)) {
      auto podValue = defaultValue(
          newPodOp.getType(), tables, newPodOp.getOperation(), field, uninitializedBehavior, &rng
      );
      if (!podValue) {
        return podValue.takeError();
      }
      auto podRef = asPod(*podValue);
      if (!podRef) {
        return podRef.takeError();
      }
      auto initValues = newPodOp.getInitializedRecordValues();
      for (pod::RecordValue init : initValues) {
        auto value = lookup(init.value, scope);
        if (!value) {
          return value.takeError();
        }
        (*podRef)->records[init.name] = *value;
      }
      return bind({*podRef});
    }
    if (auto readPodOp = dyn_cast<pod::ReadPodOp>(op)) {
      auto podValue = lookup(readPodOp.getPodRef(), scope);
      if (!podValue) {
        return podValue.takeError();
      }
      auto podRef = asPod(*podValue);
      if (!podRef) {
        return podRef.takeError();
      }
      auto it = (*podRef)->records.find(readPodOp.getRecordName());
      if (it == (*podRef)->records.end()) {
        return makeError("missing pod record");
      }
      return bind({it->second});
    }
    if (auto writePodOp = dyn_cast<pod::WritePodOp>(op)) {
      auto podValue = lookup(writePodOp.getPodRef(), scope);
      auto recordValue = lookup(writePodOp.getValue(), scope);
      if (!podValue) {
        return podValue.takeError();
      }
      if (!recordValue) {
        return recordValue.takeError();
      }
      auto podRef = asPod(*podValue);
      if (!podRef) {
        return podRef.takeError();
      }
      (*podRef)->records[writePodOp.getRecordName()] = *recordValue;
      return BlockResult {};
    }

    if (auto arrayNewOp = dyn_cast<array::CreateArrayOp>(op)) {
      auto arrayValue = std::make_shared<ArrayValue>();
      arrayValue->type = arrayNewOp.getType();
      if (arrayNewOp.getElements().empty()) {
        auto elementCount =
            getStaticShapeElementCount(arrayValue->type.getShape(), "array.create default value");
        if (!elementCount) {
          return elementCount.takeError();
        }
        arrayValue->elements.reserve(*elementCount);
        for (size_t i = 0; i < *elementCount; ++i) {
          auto elem = defaultValue(
              arrayValue->type.getElementType(), tables, arrayNewOp.getOperation(), field,
              uninitializedBehavior, &rng
          );
          if (!elem) {
            return elem.takeError();
          }
          arrayValue->elements.push_back(*elem);
        }
      } else {
        auto values = collectOperands(arrayNewOp.getElements(), scope);
        if (!values) {
          return values.takeError();
        }
        arrayValue->elements.assign(values->begin(), values->end());
      }
      return bind({arrayValue});
    }
    if (auto readArrayOp = dyn_cast<array::ReadArrayOp>(op)) {
      auto arrayValue = lookup(readArrayOp.getArrRef(), scope);
      if (!arrayValue) {
        return arrayValue.takeError();
      }
      auto arrayRef = asArray(*arrayValue);
      if (!arrayRef) {
        return arrayRef.takeError();
      }
      llvm::SmallVector<int64_t> indices;
      for (mlir::Value indexVal : readArrayOp.getIndices()) {
        auto value = lookup(indexVal, scope);
        if (!value) {
          return value.takeError();
        }
        auto index = asIndex(*value);
        if (!index) {
          return index.takeError();
        }
        indices.push_back(*index);
      }
      auto offset =
          checkedLinearize((*arrayRef)->type.getShape(), indices, "array index out of bounds");
      if (!offset) {
        return offset.takeError();
      }
      return bind({(*arrayRef)->elements[*offset]});
    }
    if (auto writeArrayOp = dyn_cast<array::WriteArrayOp>(op)) {
      auto arrayValue = lookup(writeArrayOp.getArrRef(), scope);
      auto rvalue = lookup(writeArrayOp.getRvalue(), scope);
      if (!arrayValue) {
        return arrayValue.takeError();
      }
      if (!rvalue) {
        return rvalue.takeError();
      }
      auto arrayRef = asArray(*arrayValue);
      if (!arrayRef) {
        return arrayRef.takeError();
      }
      llvm::SmallVector<int64_t> indices;
      for (mlir::Value indexVal : writeArrayOp.getIndices()) {
        auto value = lookup(indexVal, scope);
        if (!value) {
          return value.takeError();
        }
        auto index = asIndex(*value);
        if (!index) {
          return index.takeError();
        }
        indices.push_back(*index);
      }
      auto offset =
          checkedLinearize((*arrayRef)->type.getShape(), indices, "array index out of bounds");
      if (!offset) {
        return offset.takeError();
      }
      (*arrayRef)->elements[*offset] = *rvalue;
      return BlockResult {};
    }
    if (auto extractArrayOp = dyn_cast<array::ExtractArrayOp>(op)) {
      auto arrayValue = lookup(extractArrayOp.getArrRef(), scope);
      if (!arrayValue) {
        return arrayValue.takeError();
      }
      auto arrayRef = asArray(*arrayValue);
      if (!arrayRef) {
        return arrayRef.takeError();
      }
      llvm::SmallVector<int64_t> indices;
      for (mlir::Value indexVal : extractArrayOp.getIndices()) {
        auto value = lookup(indexVal, scope);
        if (!value) {
          return value.takeError();
        }
        auto index = asIndex(*value);
        if (!index) {
          return index.takeError();
        }
        indices.push_back(*index);
      }
      llvm::ArrayRef<int64_t> shape = (*arrayRef)->type.getShape();
      if (indices.size() >= shape.size()) {
        return makeError("array.extract indices exceed array rank");
      }
      auto subArraySize =
          getStaticShapeElementCount(shape.drop_front(indices.size()), "array.extract shape");
      if (!subArraySize) {
        return subArraySize.takeError();
      }
      auto prefixOffset =
          checkedLinearize(shape.take_front(indices.size()), indices, "array index out of bounds");
      if (!prefixOffset) {
        return prefixOffset.takeError();
      }
      bool baseOverflow = false;
      size_t base = llvm::SaturatingMultiply(*prefixOffset, *subArraySize, &baseOverflow);
      if (baseOverflow) {
        return makeError("array.extract element offset would overflow size_t");
      }
      auto subArray = std::make_shared<ArrayValue>();
      subArray->type = extractArrayOp.getType();
      subArray->elements.reserve(*subArraySize);
      for (size_t i = 0; i < *subArraySize; ++i) {
        bool overflow = false;
        size_t elementOffset = llvm::SaturatingAdd(base, i, &overflow);
        if (overflow) {
          return makeError("array.extract element offset would overflow size_t");
        }
        subArray->elements.push_back((*arrayRef)->elements[elementOffset]);
      }
      return bind({subArray});
    }
    if (auto insertArrayOp = dyn_cast<array::InsertArrayOp>(op)) {
      auto arrayValue = lookup(insertArrayOp.getArrRef(), scope);
      auto subArrayValue = lookup(insertArrayOp.getRvalue(), scope);
      if (!arrayValue) {
        return arrayValue.takeError();
      }
      if (!subArrayValue) {
        return subArrayValue.takeError();
      }
      auto arrayRef = asArray(*arrayValue);
      auto subArrayRef = asArray(*subArrayValue);
      if (!arrayRef) {
        return arrayRef.takeError();
      }
      if (!subArrayRef) {
        return subArrayRef.takeError();
      }
      llvm::SmallVector<int64_t> indices;
      for (mlir::Value indexVal : insertArrayOp.getIndices()) {
        auto value = lookup(indexVal, scope);
        if (!value) {
          return value.takeError();
        }
        auto index = asIndex(*value);
        if (!index) {
          return index.takeError();
        }
        indices.push_back(*index);
      }
      llvm::ArrayRef<int64_t> shape = (*arrayRef)->type.getShape();
      size_t subArraySize = (*subArrayRef)->elements.size();
      auto prefixOffset =
          checkedLinearize(shape.take_front(indices.size()), indices, "array index out of bounds");
      if (!prefixOffset) {
        return prefixOffset.takeError();
      }
      bool baseOverflow = false;
      size_t base = llvm::SaturatingMultiply(*prefixOffset, subArraySize, &baseOverflow);
      if (baseOverflow) {
        return makeError("array.insert element offset would overflow size_t");
      }
      for (size_t i = 0; i < subArraySize; ++i) {
        bool overflow = false;
        size_t elementOffset = llvm::SaturatingAdd(base, i, &overflow);
        if (overflow) {
          return makeError("array.insert element offset would overflow size_t");
        }
        (*arrayRef)->elements[elementOffset] = (*subArrayRef)->elements[i];
      }
      return BlockResult {};
    }
    if (auto arrayLenOp = dyn_cast<array::ArrayLengthOp>(op)) {
      auto dimValue = lookup(arrayLenOp.getDim(), scope);
      if (!dimValue) {
        return dimValue.takeError();
      }
      auto dim = asIndex(*dimValue);
      if (!dim) {
        return dim.takeError();
      }
      llvm::ArrayRef<int64_t> shape = arrayLenOp.getArrRefType().getShape();
      auto dimIndex = checkedShapeDimToSize(*dim, "array.len dimension");
      if (!dimIndex) {
        return dimIndex.takeError();
      }
      if (*dimIndex >= shape.size()) {
        return makeError("array.len dimension out of bounds");
      }
      return bind({WitnessVal(shape[*dimIndex])});
    }

    if (auto callOp = dyn_cast<function::CallOp>(op)) {
      if (callOp.getTemplateParams() || !callOp.getMapOperands().empty()) {
        return makeError("templated or affine-instantiated calls are not supported in llzk-witgen");
      }
      auto callee = lookupTopLevelSymbol<function::FuncDefOp>(tables, callOp.getCalleeAttr(), &op);
      if (failed(callee)) {
        return makeError("could not resolve called function");
      }
      auto args = collectOperands(callOp.getArgOperands(), scope);
      if (!args) {
        return args.takeError();
      }
      auto results = run(callee->get(), *args);
      if (!results) {
        return results.takeError();
      }
      return bind(*results);
    }

    auto handleBinaryIndex = [&](auto arithOp, auto fn) -> llvm::Expected<BlockResult> {
      auto lhs = lookup(arithOp.getLhs(), scope);
      auto rhs = lookup(arithOp.getRhs(), scope);
      if (!lhs) {
        return lhs.takeError();
      }
      if (!rhs) {
        return rhs.takeError();
      }
      auto lhsValue = asIndex(*lhs);
      if (!lhsValue) {
        return lhsValue.takeError();
      }
      auto rhsValue = asIndex(*rhs);
      if (!rhsValue) {
        return rhsValue.takeError();
      }
      return bind({WitnessVal(fn(*lhsValue, *rhsValue))});
    };

    if (auto addIOp = dyn_cast<arith::AddIOp>(op)) {
      return handleBinaryIndex(addIOp, [](int64_t lhs, int64_t rhs) { return lhs + rhs; });
    }
    if (auto subIOp = dyn_cast<arith::SubIOp>(op)) {
      return handleBinaryIndex(subIOp, [](int64_t lhs, int64_t rhs) { return lhs - rhs; });
    }
    if (auto mulIOp = dyn_cast<arith::MulIOp>(op)) {
      return handleBinaryIndex(mulIOp, [](int64_t lhs, int64_t rhs) { return lhs * rhs; });
    }
    if (auto divUIOp = dyn_cast<arith::DivUIOp>(op)) {
      return handleBinaryIndex(divUIOp, [](int64_t lhs, int64_t rhs) {
        auto divRes = static_cast<uint64_t>(lhs) / static_cast<uint64_t>(rhs);
        return static_cast<int64_t>(divRes);
      });
    }
    if (auto cmpIOp = dyn_cast<arith::CmpIOp>(op)) {
      return handleBinaryIndex(cmpIOp, [&cmpIOp](int64_t lhs, int64_t rhs) -> bool {
        bool result = false;
        switch (cmpIOp.getPredicate()) {
        case arith::CmpIPredicate::eq:
          result = lhs == rhs;
          break;
        case arith::CmpIPredicate::ne:
          result = lhs != rhs;
          break;
        case arith::CmpIPredicate::slt:
          result = lhs < rhs;
          break;
        case arith::CmpIPredicate::sle:
          result = lhs <= rhs;
          break;
        case arith::CmpIPredicate::sgt:
          result = lhs > rhs;
          break;
        case arith::CmpIPredicate::sge:
          result = lhs >= rhs;
          break;
        case arith::CmpIPredicate::ult: {
          result = static_cast<uint64_t>(lhs) < static_cast<uint64_t>(rhs);
          break;
        }
        case arith::CmpIPredicate::ule: {
          result = static_cast<uint64_t>(lhs) <= static_cast<uint64_t>(rhs);
          break;
        }
        case arith::CmpIPredicate::ugt: {
          result = static_cast<uint64_t>(lhs) > static_cast<uint64_t>(rhs);
          break;
        }
        case arith::CmpIPredicate::uge: {
          result = static_cast<uint64_t>(lhs) >= static_cast<uint64_t>(rhs);
          break;
        }
        }
        return result;
      });
    }

    if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      auto cond = lookup(selectOp.getCondition(), scope);
      auto trueValue = lookup(selectOp.getTrueValue(), scope);
      auto falseValue = lookup(selectOp.getFalseValue(), scope);
      if (!cond) {
        return cond.takeError();
      }
      if (!trueValue) {
        return trueValue.takeError();
      }
      if (!falseValue) {
        return falseValue.takeError();
      }
      auto condition = asBool(*cond);
      if (!condition) {
        return condition.takeError();
      }
      return bind({*condition ? *trueValue : *falseValue});
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      auto cond = lookup(ifOp.getCondition(), scope);
      if (!cond) {
        return cond.takeError();
      }
      auto condition = asBool(*cond);
      if (!condition) {
        return condition.takeError();
      }
      if (!*condition && ifOp.getNumResults() == 0 && ifOp.getElseRegion().empty()) {
        return bind({});
      }
      Region &region = *condition ? ifOp.getThenRegion() : ifOp.getElseRegion();
      auto result = runRegion(region, {}, scope);
      if (!result) {
        return result.takeError();
      }
      return bind(result->values);
    }

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto lowerBoundValue = lookup(forOp.getLowerBound(), scope);
      auto upperBoundValue = lookup(forOp.getUpperBound(), scope);
      auto stepValue = lookup(forOp.getStep(), scope);
      if (!lowerBoundValue) {
        return lowerBoundValue.takeError();
      }
      if (!upperBoundValue) {
        return upperBoundValue.takeError();
      }
      if (!stepValue) {
        return stepValue.takeError();
      }
      auto lowerBound = asIndex(*lowerBoundValue);
      auto upperBound = asIndex(*upperBoundValue);
      auto step = asIndex(*stepValue);
      if (!lowerBound) {
        return lowerBound.takeError();
      }
      if (!upperBound) {
        return upperBound.takeError();
      }
      if (!step) {
        return step.takeError();
      }
      auto iterValuesOrErr = collectOperands(forOp.getInitArgs(), scope);
      if (!iterValuesOrErr) {
        return iterValuesOrErr.takeError();
      }
      llvm::SmallVector<WitnessVal> iterValues = std::move(*iterValuesOrErr);

      if (usesUnsignedCmp(forOp)) {
        auto lowerBoundUIntValue = static_cast<uint64_t>(*lowerBound);
        auto upperBoundUIntValue = static_cast<uint64_t>(*upperBound);
        auto stepUInt = static_cast<uint64_t>(*step);
        for (uint64_t iv = lowerBoundUIntValue, ub = upperBoundUIntValue, unsignedStep = stepUInt;
             iv < ub; iv += unsignedStep) {
          auto signedIV = toCheckedInt64(iv);
          if (!signedIV) {
            return signedIV.takeError();
          }
          llvm::SmallVector<WitnessVal> regionArgs;
          regionArgs.push_back(WitnessVal(*signedIV));
          regionArgs.append(iterValues.begin(), iterValues.end());
          auto result = runRegion(forOp.getRegion(), regionArgs, scope);
          if (!result) {
            return result.takeError();
          }
          iterValues = std::move(result->values);
        }
      } else {
        for (int64_t iv = *lowerBound; iv < *upperBound; iv += *step) {
          llvm::SmallVector<WitnessVal> regionArgs;
          regionArgs.push_back(WitnessVal(iv));
          regionArgs.append(iterValues.begin(), iterValues.end());
          auto result = runRegion(forOp.getRegion(), regionArgs, scope);
          if (!result) {
            return result.takeError();
          }
          iterValues = std::move(result->values);
        }
      }
      return bind(iterValues);
    }

    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      auto iterValuesOrErr = collectOperands(whileOp.getInits(), scope);
      if (!iterValuesOrErr) {
        return iterValuesOrErr.takeError();
      }
      llvm::SmallVector<WitnessVal> iterValues = std::move(*iterValuesOrErr);
      while (true) {
        auto beforeResult = runRegion(whileOp.getBefore(), iterValues, scope);
        if (!beforeResult) {
          return beforeResult.takeError();
        }
        if (!beforeResult->terminated) {
          return makeError("scf.while before region must terminate with scf.condition");
        }
        if (beforeResult->values.empty()) {
          return makeError("scf.while before region did not produce a condition");
        }

        auto condition = asBool(beforeResult->values.front());
        if (!condition) {
          return condition.takeError();
        }

        llvm::SmallVector<WitnessVal> nextValues;
        nextValues.append(beforeResult->values.begin() + 1, beforeResult->values.end());
        if (!*condition) {
          return bind(nextValues);
        }

        auto afterResult = runRegion(whileOp.getAfter(), nextValues, scope);
        if (!afterResult) {
          return afterResult.takeError();
        }
        if (!afterResult->terminated) {
          return makeError("scf.while after region must terminate with scf.yield");
        }
        iterValues = std::move(afterResult->values);
      }
    }

    return makeError(llvm::Twine("unsupported op in llzk-witgen: ") + op.getName().getStringRef());
  }
};

} // namespace

/// Execute a function body with concrete runtime values.
llvm::Expected<llvm::SmallVector<WitnessVal>>
FunctionInterpreter::run(function::FuncDefOp funcOp, ArrayRef<WitnessVal> args) {
  return InvocationInterpreter(moduleOp, tables, field, uninitializedBehavior, rng)
      .run(funcOp, args);
}

} // namespace llzk::witgen
