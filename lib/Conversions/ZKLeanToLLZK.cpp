//===-- ZKLeanToLLZK.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Conversions/Passes.h"

#include "llzk/Dialect/Bool/IR/Attrs.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.h"
#include "llzk/Dialect/ZKExpr/IR/ZKExprOps.h"
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanOps.h"
#include "llzk/Util/Constants.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>

#include <optional>

// ZKLean -> LLZK conversion overview:
// - `createConvertZKLeanToLLZKPass` constructs `ConvertZKLeanToLLZKPass`.
// - `ConvertZKLeanToLLZKPass::runOnOperation` creates the LLZK module and calls
//   `convertLeanModule` for the actual conversion.
// - `convertLeanModule` emits structs with `emitStructDefsFromZKLean`, converts each
//   function via `convertFunction`, and then fills missing pieces with
//   `ensureComputeStub`/`ensureConstrainStub`.
// - `convertFunction` gathers ops with `collectOpsAndWitnesses`, resolves the target
//   struct via `resolveStructState`, creates the LLZK function with
//   `createTargetFunction`, and delegates per-op lowering to
//   `FunctionConverter::convertOperation` before finalize emits the return.
using namespace mlir;

namespace llzk {
#define GEN_PASS_DECL_CONVERTZKLEANTOLLZKPASS
#define GEN_PASS_DEF_CONVERTZKLEANTOLLZKPASS
#include "llzk/Conversions/LLZKConversionPasses.h.inc"
} // namespace llzk

namespace {

// Parse a `bool.cmp_*` function suffix into a predicate enum.
// Returns `std::nullopt` for unknown names so the caller can emit diagnostics.
static std::optional<llzk::boolean::FeltCmpPredicate>
parseCmpPredicate(StringRef name) {
  if (name == "eq")
    return llzk::boolean::FeltCmpPredicate::EQ;
  if (name == "ne")
    return llzk::boolean::FeltCmpPredicate::NE;
  if (name == "lt")
    return llzk::boolean::FeltCmpPredicate::LT;
  if (name == "le")
    return llzk::boolean::FeltCmpPredicate::LE;
  if (name == "gt")
    return llzk::boolean::FeltCmpPredicate::GT;
  if (name == "ge")
    return llzk::boolean::FeltCmpPredicate::GE;
  return std::nullopt;
}

// Track conversion status for a single `llzk::component::StructDefOp`.
// Used to avoid duplicate `@compute`/`@constrain` emission.
struct StructState {
  llzk::component::StructDefOp def;
  bool hasCompute = false;
  bool hasConstrain = false;
};

// Shared conversion state for ZKLean -> LLZK lowering.
// Carries the destination module, builder, felt type, and error flags.
struct ZKLeanToLLZKState {
  ModuleOp dest;
  OpBuilder &builder;
  llzk::felt::FeltType feltType;
  bool &hadError;
  llvm::StringMap<StructState> &structStates;
};

// Map `llzk::zkleanlean::StructType` into `llzk::component::StructType`.
// Leaves non-struct types untouched for downstream lowering.
static Type mapStructType(Type type) {
  if (auto structType = dyn_cast<llzk::zkleanlean::StructType>(type))
    return llzk::component::StructType::get(structType.getNameRef());
  return type;
}

// Materialize LLZK `struct.def` ops from ZKLeanLean struct definitions.
// This pre-pass ensures struct symbols exist before function conversion.
static void emitStructDefsFromZKLean(ModuleOp source,
                                     ZKLeanToLLZKState &state) {
  // Pre-pass: materialize LLZK struct.defs from ZKLeanLean.structure ops first so
  // struct.type<@Name> symbols exist before function conversion.
  llvm::StringSet<> seenStructs;
  for (auto def : source.getBody()->getOps<llzk::zkleanlean::StructDefOp>()) {
    StringRef name = def.getSymName();
    if (!seenStructs.insert(name).second)
      continue;
    OpBuilder::InsertionGuard guard(state.builder);
    state.builder.setInsertionPointToEnd(state.dest.getBody());
    auto structDef = state.builder.create<llzk::component::StructDefOp>(
        def.getLoc(), def.getSymNameAttr(), ArrayAttr());
    structDef.getBodyRegion().emplaceBlock();
    auto &body = structDef.getBodyRegion().front();
    OpBuilder memberBuilder(&body, body.begin());
    for (auto member : def.getBody()->getOps<llzk::zkleanlean::MemberDefOp>()) {
      memberBuilder.create<llzk::component::MemberDefOp>(
          member.getLoc(), member.getSymName(), state.feltType);
    }
    StructState structState;
    structState.def = structDef;
    structState.hasCompute = structDef.getComputeFuncOp() != nullptr;
    structState.hasConstrain = structDef.getConstrainFuncOp() != nullptr;
    state.structStates.try_emplace(name, structState);
  }
}

// Ensure a struct has a minimal `@compute` function body.
// The stub returns an empty struct instance when ZKLean provides no compute.
static void ensureComputeStub(StructState &state,
                              ArrayRef<Type> baseInputTypes, Location loc,
                              ZKLeanToLLZKState &ctx) {
  if (state.hasCompute)
    return;
  SmallVector<Type> computeInputs;
  if (!baseInputTypes.empty() &&
      mlir::isa<llzk::component::StructType>(baseInputTypes.front())) {
    computeInputs.append(baseInputTypes.begin() + 1, baseInputTypes.end());
  } else {
    computeInputs.append(baseInputTypes.begin(), baseInputTypes.end());
  }
  auto structType = state.def.getType();
  auto computeType = FunctionType::get(ctx.dest.getContext(), computeInputs,
                                       ArrayRef<Type>{structType});
  OpBuilder::InsertionGuard guard(ctx.builder);
  ctx.builder.setInsertionPointToEnd(&state.def.getBodyRegion().front());
  // Stub compute: return an empty struct instance when ZKLean has no compute.
  auto computeFunc = ctx.builder.create<llzk::function::FuncDefOp>(
      loc, ctx.builder.getStringAttr(llzk::FUNC_NAME_COMPUTE), computeType);
  computeFunc.setAllowWitnessAttr(true);
  Block *computeBlock = computeFunc.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(computeBlock);
  auto selfVal =
      bodyBuilder.create<llzk::component::CreateStructOp>(loc, structType);
  bodyBuilder.create<llzk::function::ReturnOp>(loc, selfVal.getResult());
  state.hasCompute = true;
}

// Ensure a struct has a minimal `@constrain` function body.
// The stub is a no-op constrain function to satisfy LLZK invariants.
static void ensureConstrainStub(StructState &state, Location loc,
                                ZKLeanToLLZKState &ctx) {
  if (state.hasConstrain)
    return;
  auto structType = state.def.getType();
  auto constrainType =
      FunctionType::get(ctx.dest.getContext(), ArrayRef<Type>{structType}, {});
  OpBuilder::InsertionGuard guard(ctx.builder);
  ctx.builder.setInsertionPointToEnd(&state.def.getBodyRegion().front());
  auto constrainFunc = ctx.builder.create<llzk::function::FuncDefOp>(
      loc, ctx.builder.getStringAttr(llzk::FUNC_NAME_CONSTRAIN), constrainType);
  constrainFunc.setAllowConstraintAttr(true);
  Block *constrainBlock = constrainFunc.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(constrainBlock);
  bodyBuilder.create<llzk::function::ReturnOp>(loc);
  state.hasConstrain = true;
}

// Collect operations and ZKExpr witnesses from a single block.
// The captured list is used to avoid iterator invalidation during rewriting.
static void collectOpsAndWitnesses(
    Block &block, SmallVector<Operation *, 16> &ops,
    SmallVector<llzk::zkexpr::WitnessOp, 8> &witnesses) {
  for (Operation &op : block) {
    ops.push_back(&op);
    if (auto witness = dyn_cast<llzk::zkexpr::WitnessOp>(&op))
      witnesses.push_back(witness);
  }
}

// Resolve the struct context for a constrain-style function, if any.
// Validates name/type consistency and records errors through `hadError`.
template <typename FuncOpTy>
static StructState *resolveStructState(
    FuncOpTy func, ArrayRef<Type> baseInputTypes,
    llvm::StringMap<StructState> &structStates, bool &hadError) {
  StringRef funcName = func.getSymName();
  std::optional<StringRef> nameStruct;
  std::optional<StringRef> typeStruct;
  StructState *structState = nullptr;

  if (funcName.ends_with("__constrain")) {
    auto splitPos = funcName.rfind("__");
    if (splitPos != StringRef::npos)
      nameStruct = funcName.take_front(splitPos);
  }
  if (!baseInputTypes.empty()) {
    if (auto structType =
            dyn_cast<llzk::component::StructType>(baseInputTypes.front())) {
      typeStruct = structType.getNameRef().getRootReference().getValue();
      auto it = structStates.find(*typeStruct);
      if (it != structStates.end()) {
        structState = &it->second;
      } else {
        func.emitError("missing struct.def for function self parameter");
        hadError = true;
        return nullptr;
      }
    }
  }
  if (nameStruct && typeStruct && *nameStruct != *typeStruct) {
    func.emitError("struct name mismatch between function name and type");
    hadError = true;
    return nullptr;
  }
  if (nameStruct && !typeStruct) {
    func.emitError("constrain name requires a struct-typed first argument");
    hadError = true;
    return nullptr;
  }
  return structState;
}

// Per-function converter from ZKLean/ZKExpr ops into LLZK ops.
// Owns SSA maps for felt values, ZKExpr values, and witness arguments.
struct FunctionConverter {
  ZKLeanToLLZKState &state;
  Block &oldBlock;
  Block *newBlock;
  DenseMap<Value, Value> feltValueMap;
  DenseMap<Value, Value> zkToFeltMap;
  DenseMap<Value, Value> leanValueMap;
  DenseMap<Value, Value> argMap;
  DenseMap<Operation *, Value> witnessArgs;

  // Initialize conversion state for a single function body.
  // The new block is populated by subsequent mapping calls.
  FunctionConverter(ZKLeanToLLZKState &stateRef, Block &srcBlock,
                    Block *destBlock)
      : state(stateRef), oldBlock(srcBlock), newBlock(destBlock) {}

  // Add block arguments for the new function signature.
  // Tracks original arguments and pre-maps felt-typed values.
  void mapBlockArguments(ArrayRef<Type> inputTypes) {
    for (auto [idx, oldArg] : llvm::enumerate(oldBlock.getArguments())) {
      auto newArg = newBlock->addArgument(inputTypes[idx], oldArg.getLoc());
      argMap[oldArg] = newArg;
      if (mlir::isa<llzk::felt::FeltType>(oldArg.getType()))
        feltValueMap[oldArg] = newArg;
    }
  }

  // Append extra arguments for each `llzk::zkexpr::WitnessOp`.
  // Each witness becomes a felt-typed argument on the target block.
  void mapWitnessArgs(ArrayRef<llzk::zkexpr::WitnessOp> witnesses) {
    for (auto witness : witnesses) {
      auto newArg = newBlock->addArgument(state.feltType, witness.getLoc());
      witnessArgs[witness.getOperation()] = newArg;
    }
  }

  // Resolve a felt value mapped into the new function.
  // Falls back to argument mapping when possible.
  Value mapFelt(Value v) {
    if (auto it = feltValueMap.find(v); it != feltValueMap.end())
      return it->second;
    if (auto blockArg = mlir::dyn_cast<BlockArgument>(v))
      return argMap.lookup(blockArg);
    return Value();
  }

  // Resolve a ZKExpr value that was lowered into a felt value.
  // Returns an empty `Value` if the mapping is missing.
  Value mapZK(Value v) {
    if (auto it = zkToFeltMap.find(v); it != zkToFeltMap.end())
      return it->second;
    return Value();
  }

  // Map a non-ZK value (e.g., `zkleanlean.call` operand) into LLZK.
  // Emits diagnostics when the value cannot be resolved.
  Value mapLeanValue(Value v, Operation *userOp) {
    if (auto it = leanValueMap.find(v); it != leanValueMap.end())
      return it->second;
    if (auto blockArg = mlir::dyn_cast<BlockArgument>(v))
      return argMap.lookup(blockArg);
    userOp->emitError("unsupported value producer in ZKLean conversion");
    state.hadError = true;
    return Value();
  }

  // Lower a single operation into its LLZK equivalent.
  // Handles ZKExpr arithmetic, `zkbuilder.constrain_eq`, and Lean calls.
  void convertOperation(Operation *op) {
    // Felt constants to felt constants
    if (auto constOp = dyn_cast<llzk::felt::FeltConstantOp>(op)) {
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      auto cloned = state.builder.create<llzk::felt::FeltConstantOp>(
          constOp.getLoc(), constOp.getResult().getType(),
          constOp.getValueAttr());
      feltValueMap[constOp.getResult()] = cloned.getResult();
      return;
    }

    // ZKExpr literal maps to underlying felt value
    if (auto literal = dyn_cast<llzk::zkexpr::LiteralOp>(op)) {
      Value mapped = mapFelt(literal.getLiteral());
      if (!mapped) {
        literal.emitError("unsupported literal source in ZKLean conversion");
        state.hadError = true;
        return;
      }
      zkToFeltMap[literal.getOutput()] = mapped;
      return;
    }

    // ZKExpr witness maps to dedicated witness argument
    if (auto witness = dyn_cast<llzk::zkexpr::WitnessOp>(op)) {
      Value arg = witnessArgs.lookup(witness.getOperation());
      zkToFeltMap[witness.getOutput()] = arg;
      return;
    }

    // ZKLeanLean.call to LLZK operations
    if (auto call = dyn_cast<llzk::zkleanlean::CallOp>(op)) {
      StringRef calleeName =
          call.getCallee().getRootReference().getValue();
      if (calleeName.consume_front("bool.cmp_")) {
        auto predicate = parseCmpPredicate(calleeName);
        if (!predicate) {
          call.emitError("unsupported bool.cmp predicate in ZKLean conversion");
          state.hadError = true;
          return;
        }
        if (call.getNumOperands() != 2 || call.getNumResults() != 1) {
          call.emitError("bool.cmp expects two operands and one result");
          state.hadError = true;
          return;
        }
        Value lhs = mapZK(call.getOperand(0));
        Value rhs = mapZK(call.getOperand(1));
        if (!lhs || !rhs) {
          call.emitError("unsupported bool.cmp operands in ZKLean conversion");
          state.hadError = true;
          return;
        }
        OpBuilder::InsertionGuard guard(state.builder);
        state.builder.setInsertionPointToEnd(newBlock);
        auto predAttr = llzk::boolean::FeltCmpPredicateAttr::get(
            state.dest.getContext(), *predicate);
        auto cmpOp = state.builder.create<llzk::boolean::CmpOp>(
            call.getLoc(), predAttr, lhs, rhs);
        leanValueMap[call.getResult(0)] = cmpOp.getResult();
        return;
      }
      if (calleeName == "cast.tofelt") {
        if (call.getNumOperands() != 1 || call.getNumResults() != 1) {
          call.emitError("cast.tofelt expects one operand and one result");
          state.hadError = true;
          return;
        }
        Value value = mapLeanValue(call.getOperand(0), call.getOperation());
        if (!value)
          return;
        OpBuilder::InsertionGuard guard(state.builder);
        state.builder.setInsertionPointToEnd(newBlock);
        auto castOp = state.builder.create<llzk::cast::IntToFeltOp>(
            call.getLoc(), state.feltType, value);
        zkToFeltMap[call.getResult(0)] = castOp.getResult();
        return;
      }
      call.emitError("unsupported ZKLeanLean call in ZKLean conversion");
      state.hadError = true;
      return;
    }

    // ZKLeanLean.accessor to struct.readf
    if (auto accessor = dyn_cast<llzk::zkleanlean::AccessorOp>(op)) {
      Value component = argMap.lookup(accessor.getComponent());
      if (!component) {
        accessor.emitError("unsupported struct source in ZKLean conversion");
        state.hadError = true;
        return;
      }
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      auto memberAttr =
          state.builder.getStringAttr(accessor.getMemberNameAttr().getValue());
      auto newRead = state.builder.create<llzk::component::MemberReadOp>(
          accessor.getLoc(), state.feltType, component, memberAttr);
      zkToFeltMap[accessor.getValue()] = newRead.getVal();
      return;
    }

    // ZKExpr.Add to felt.add
    if (auto add = dyn_cast<llzk::zkexpr::AddOp>(op)) {
      Value lhs = mapZK(add.getLhs());
      Value rhs = mapZK(add.getRhs());
      if (!lhs || !rhs) {
        add.emitError("missing ZKExpr operands in ZKLean conversion");
        state.hadError = true;
        return;
      }
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      auto feltAdd =
          state.builder.create<llzk::felt::AddFeltOp>(add.getLoc(), lhs, rhs);
      zkToFeltMap[add.getOutput()] = feltAdd.getResult();
      return;
    }

    // ZKExpr.Sub to felt.sub
    if (auto sub = dyn_cast<llzk::zkexpr::SubOp>(op)) {
      Value lhs = mapZK(sub.getLhs());
      Value rhs = mapZK(sub.getRhs());
      if (!lhs || !rhs) {
        sub.emitError("missing ZKExpr operands in ZKLean conversion");
        state.hadError = true;
        return;
      }
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      auto feltSub =
          state.builder.create<llzk::felt::SubFeltOp>(sub.getLoc(), lhs, rhs);
      zkToFeltMap[sub.getOutput()] = feltSub.getResult();
      return;
    }

    // ZKExpr.Mul to felt.mul
    if (auto mul = dyn_cast<llzk::zkexpr::MulOp>(op)) {
      Value lhs = mapZK(mul.getLhs());
      Value rhs = mapZK(mul.getRhs());
      if (!lhs || !rhs) {
        mul.emitError("missing ZKExpr operands in ZKLean conversion");
        state.hadError = true;
        return;
      }
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      auto feltMul =
          state.builder.create<llzk::felt::MulFeltOp>(mul.getLoc(), lhs, rhs);
      zkToFeltMap[mul.getOutput()] = feltMul.getResult();
      return;
    }

    // ZKExpr.Neg to felt.Neg
    if (auto neg = dyn_cast<llzk::zkexpr::NegOp>(op)) {
      Value operand = mapZK(neg.getValue());
      if (!operand) {
        neg.emitError("missing ZKExpr operand in ZKLean conversion");
        state.hadError = true;
        return;
      }
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      auto feltNeg =
          state.builder.create<llzk::felt::NegFeltOp>(neg.getLoc(), operand);
      zkToFeltMap[neg.getOutput()] = feltNeg.getResult();
      return;
    }

    // ZKBuilder.ConstrainEq to constrain.eq
    if (auto constraint = dyn_cast<llzk::zkbuilder::ConstrainEqOp>(op)) {
      Value lhs = mapZK(constraint.getLhs());
      Value rhs = mapZK(constraint.getRhs());
      if (!lhs || !rhs) {
        constraint.emitError("missing ZKExpr operands in ZKLean conversion");
        state.hadError = true;
        return;
      }
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      state.builder.create<llzk::constrain::EmitEqualityOp>(
          constraint.getLoc(), lhs, rhs);
      return;
    }

    if (op->getNumResults() != 0) {
      op->emitError("unsupported ZKLean operation in conversion");
      state.hadError = true;
    }
  }

  // Emit the terminal `llzk::function::ReturnOp` for the new block.
  // Called after all operations have been converted.
  void finalize(Location loc) {
    OpBuilder::InsertionGuard guard(state.builder);
    state.builder.setInsertionPointToEnd(newBlock);
    state.builder.create<llzk::function::ReturnOp>(loc);
  }
};

// Create the target LLZK function, optionally nested under a struct.
// Ensures struct constraints are unique and marks witness/constraint attrs.
template <typename FuncOpTy>
static llzk::function::FuncDefOp createTargetFunction(
    FuncOpTy func, FunctionType newFuncType, StructState *structState,
    ArrayRef<Type> baseInputTypes, bool allowWitness,
    ZKLeanToLLZKState &state) {
  llzk::function::FuncDefOp newFunc;
  if (structState) {
    if (structState->hasConstrain) {
      func.emitError("duplicate constrain function for struct");
      state.hadError = true;
      return llzk::function::FuncDefOp();
    }
    // Ensure the struct has a compute body so it validates as a component.
    ensureComputeStub(*structState, baseInputTypes, func.getLoc(), state);
    OpBuilder::InsertionGuard guard(state.builder);
    state.builder.setInsertionPointToEnd(&structState->def.getBodyRegion().front());
    newFunc = state.builder.create<llzk::function::FuncDefOp>(
        func.getLoc(), state.builder.getStringAttr(llzk::FUNC_NAME_CONSTRAIN),
        newFuncType);
    structState->hasConstrain = true;
  } else {
    state.builder.setInsertionPointToEnd(state.dest.getBody());
    newFunc = state.builder.create<llzk::function::FuncDefOp>(
        func.getLoc(), func.getSymName(), newFuncType);
  }
  newFunc.setAllowConstraintAttr(true);
  if (allowWitness)
    newFunc.setAllowWitnessAttr(true);
  return newFunc;
}

// Convert a ZKLean function into an LLZK function body.
// Adds witness arguments, resolves struct context, and rewrites each op.
template <typename FuncOpTy>
static void convertFunction(FuncOpTy func, bool allowWitness,
                            ZKLeanToLLZKState &state) {
  if (func.getBody().empty())
    return;
  if (state.hadError)
    return;

  Block &oldBlock = func.getBody().front();
  SmallVector<Operation *, 16> ops;
  SmallVector<llzk::zkexpr::WitnessOp, 8> witnesses;
  collectOpsAndWitnesses(oldBlock, ops, witnesses);

  // Collect input types of original ZKLean function.
  SmallVector<Type> baseInputTypes;
  auto funcType = func.getFunctionType();
  for (Type type : funcType.getInputs())
    baseInputTypes.push_back(mapStructType(type));
  SmallVector<Type> inputTypes = baseInputTypes;
  // Add an input of felt type for each ZKExpr witness.
  inputTypes.append(witnesses.size(), state.feltType);
  auto newFuncType = FunctionType::get(state.dest.getContext(), inputTypes,
                                       funcType.getResults());

  StructState *structState =
      resolveStructState(func, baseInputTypes, state.structStates, state.hadError);
  if (state.hadError)
    return;

  auto newFunc =
      createTargetFunction(func, newFuncType, structState, baseInputTypes,
                           allowWitness, state);
  if (!newFunc || state.hadError)
    return;

  auto *newBlock = new Block();
  newFunc.getBody().push_front(newBlock);

  FunctionConverter converter(state, oldBlock, newBlock);
  converter.mapBlockArguments(inputTypes);
  converter.mapWitnessArgs(witnesses);
  for (Operation *op : ops)
    converter.convertOperation(op);
  if (state.hadError)
    return;
  converter.finalize(func.getLoc());
}

// Convert a ZKLean module into a nested LLZK module.
// Emits structs first, then converts functions and fills missing stubs.
static LogicalResult convertLeanModule(ModuleOp source, ModuleOp dest) {
  OpBuilder builder(dest.getContext());
  bool hadError = false;
  llvm::StringMap<StructState> structStates;
  auto feltType = llzk::felt::FeltType::get(dest.getContext());

  ZKLeanToLLZKState state{dest, builder, feltType, hadError, structStates};
  emitStructDefsFromZKLean(source, state);

  source.walk([&](mlir::func::FuncOp func) {
    convertFunction(func, func->hasAttr("function.allow_witness"), state);
  });
  source.walk([&](llzk::function::FuncDefOp func) {
    convertFunction(func, func.hasAllowWitnessAttr(), state);
  });

  // Any struct without a converted constrain function gets stubs.
  for (auto &entry : structStates) {
    StructState &structState = entry.getValue();
    if (!structState.hasConstrain) {
      ensureComputeStub(structState, {}, structState.def.getLoc(), state);
      ensureConstrainStub(structState, structState.def.getLoc(), state);
    }
  }

  if (hadError)
    return failure();
  return success();
}

// Pass wrapper that appends a converted LLZK module to the source.
// Delegates conversion details to `convertLeanModule`.
class ConvertZKLeanToLLZKPass
    : public llzk::impl::ConvertZKLeanToLLZKPassBase<
          ConvertZKLeanToLLZKPass> {
public:
  // Register dialects required for the lowered LLZK module.
  // Keeps the pass self-contained for `mlir::PassManager`.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<llzk::constrain::ConstrainDialect,
                    llzk::boolean::BoolDialect,
                    llzk::cast::CastDialect,
                    llzk::felt::FeltDialect,
                    llzk::function::FunctionDialect,
                    llzk::component::StructDialect,
                    mlir::func::FuncDialect,
                    llzk::zkexpr::ZKExprDialect,
                    llzk::zkleanlean::ZKLeanLeanDialect>();
  }

  // Create the LLZK module, run conversion, and replace the source module.
  // Emits a diagnostic and signals failure when conversion fails.
  void runOnOperation() override {
    ModuleOp source = getOperation();
    ModuleOp llzkModule = ModuleOp::create(source.getLoc());
    auto symName = StringAttr::get(&getContext(), "LLZK");
    llzkModule->setAttr(SymbolTable::getSymbolAttrName(), symName);
    if (auto lang = source->getAttr(llzk::LANG_ATTR_NAME)) {
      llzkModule->setAttr(llzk::LANG_ATTR_NAME, lang);
    }

    if (failed(convertLeanModule(source, llzkModule))) {
      source.emitError("failed to convert ZKLean module");
      signalPassFailure();
      return;
    }

    if (Block *parent = source->getBlock()) {
      parent->getOperations().insert(source->getIterator(),
                                     llzkModule.getOperation());
      source.erase();
      return;
    }

    source->setAttrs(llzkModule->getAttrs());
    source.getBodyRegion().takeBody(llzkModule.getBodyRegion());
  }
};

} // namespace

namespace llzk {

// Pass factory for `ConvertZKLeanToLLZKPass`.
// Used by pass registration and external callers.
std::unique_ptr<Pass> createConvertZKLeanToLLZKPass() {
  return std::make_unique<ConvertZKLeanToLLZKPass>();
}

} // namespace llzk
