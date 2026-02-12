//===-- LLZKToZKLean.cpp ----------------------------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "llzk/Conversions/Passes.h"

#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.h"
#include "llzk/Dialect/ZKExpr/IR/ZKExprOps.h"
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanOps.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>

// LLZK -> ZKLean conversion overview:
// - `createConvertLLZKToZKLeanPass` constructs `ConvertLLZKToZKLeanPass`.
// - `ConvertLLZKToZKLeanPass::runOnOperation` creates the `ZKLean` module,
//   copies relevant attrs, calls `convertModule`, and replaces the module body.
// - `convertModule` builds `LLZKToZKLeanState`, emits structs via
//   `emitZKLeanStructDefs`, then walks `llzk::function::FuncDefOp` and delegates
//   each to `convertFunction`.
// - `convertFunction` filters with `shouldConvertFunc`, instantiates
//   `FunctionConverter`, runs `collectBlockOps`, and relies on
//   `FunctionConverter::convertOperation` before `FunctionConverter::finalize`
//   closes the new function.
// - `mapStructType`, `mapValue`, and `mapLeanValue` handle type and SSA mapping
//   between LLZK values and ZKLean/ZKExpr values.
using namespace mlir;

namespace llzk {
#define GEN_PASS_DEF_CONVERTLLZKTOZKLEANPASS
#include "llzk/Conversions/LLZKConversionPasses.h.inc"
} // namespace llzk

namespace {

// Build a Lean-friendly name for a `llzk::function::FuncDefOp`.
// Flattens nested symbol references into a single identifier using "__".
static std::string buildLeanFunctionName(llzk::function::FuncDefOp func) {
  std::string name;
  auto fq = func.getFullyQualifiedName(false);
  if (!fq)
    return func.getSymName().str();

  name = fq.getRootReference().str();
  for (SymbolRefAttr nested : fq.getNestedReferences()) {
    name.append("__");
    name.append(nested.getLeafReference().str());
  }
  return name;
}

// Map a `llzk::boolean::FeltCmpPredicate` to a Lean call suffix.
// Keeps the conversion stable for `bool.cmp_*` function names.
static const char *cmpPredicateSuffix(llzk::boolean::FeltCmpPredicate predicate) {
  switch (predicate) {
  case llzk::boolean::FeltCmpPredicate::EQ:
    return "eq";
  case llzk::boolean::FeltCmpPredicate::NE:
    return "ne";
  case llzk::boolean::FeltCmpPredicate::LT:
    return "lt";
  case llzk::boolean::FeltCmpPredicate::LE:
    return "le";
  case llzk::boolean::FeltCmpPredicate::GT:
    return "gt";
  case llzk::boolean::FeltCmpPredicate::GE:
    return "ge";
  }
  return "eq";
}

// Shared conversion state for module-level LLZK -> ZKLean lowering.
// Bundles the destination `ModuleOp`, `OpBuilder`, and error state.
struct LLZKToZKLeanState {
  ModuleOp dest;
  OpBuilder &builder;
  Type zkType;
  bool &hadError;
};

// Convert `llzk::component::StructType` to `mlir::zkleanlean::StructType`.
// Leaves other types unchanged.
static Type mapStructType(Type type, MLIRContext *context) {
  if (auto structType = dyn_cast<llzk::component::StructType>(type))
    return mlir::zkleanlean::StructType::get(context, structType.getNameRef());
  return type;
}

// Decide if a `llzk::function::FuncDefOp` should be lowered.
// Requires a body and the `function.allow_constraint` attribute.
static bool shouldConvertFunc(llzk::function::FuncDefOp func) {
  if (func.getBody().empty())
    return false;
  return func->hasAttr("function.allow_constraint");
}

// Snapshot the operations in a `Block` to allow safe rewrites.
// This avoids iterator invalidation while emitting new ops.
static SmallVector<Operation *, 16> collectBlockOps(Block &block) {
  SmallVector<Operation *, 16> ops;
  for (Operation &op : block)
    ops.push_back(&op);
  return ops;
}

// Emit `mlir::zkleanlean::StructDefOp` symbols for LLZK structs.
// Only felt-typed fields are supported; others set `state.hadError`.
static void emitZKLeanStructDefs(ModuleOp source, LLZKToZKLeanState &state) {
  // Emit ZKLeanLean.structure equivalents so downstream functions can reference
  // ZKLean struct types by symbol.
  llvm::DenseSet<StringRef> seenStructs;
  for (auto def : source.getBody()->getOps<llzk::component::StructDefOp>()) {
    StringRef name = def.getSymName();
    if (!seenStructs.insert(name).second)
      continue;
    OpBuilder::InsertionGuard guard(state.builder);
    state.builder.setInsertionPointToEnd(state.dest.getBody());
    auto zkStruct = state.builder.create<mlir::zkleanlean::StructDefOp>(
        def.getLoc(), def.getSymNameAttr());
    auto *body = new Block();
    zkStruct.getBodyRegion().push_back(body);
    OpBuilder fieldBuilder(body, body->begin());
    for (auto field : def.getBody()->getOps<llzk::component::FieldDefOp>()) {
      if (!mlir::isa<llzk::felt::FeltType>(field.getType())) {
        field.emitError("unsupported field type for ZKLean struct conversion");
        state.hadError = true;
        continue;
      }
      fieldBuilder.create<mlir::zkleanlean::FieldDefOp>(
          field.getLoc(), field.getSymNameAttr(),
          TypeAttr::get(state.zkType));
    }
  }
}

// Per-function converter for LLZK constraint bodies into ZKLean IR.
// Owns the new function, blocks, and SSA maps for felt/ZK/Lean values.
struct FunctionConverter {
  LLZKToZKLeanState &state;
  llzk::function::FuncDefOp func;
  Block &oldBlock;
  mlir::func::FuncOp leanFunc;
  Block *newBlock = nullptr;
  DenseMap<Value, Value> zkValues;
  DenseMap<Value, Value> leanValues;
  DenseMap<Value, Value> argMapping;

  // Initialize the target `mlir::func::FuncOp` and argument mapping.
  // Converts struct-typed args with `mapStructType` and creates the entry block.
  FunctionConverter(LLZKToZKLeanState &stateRef,
                    llzk::function::FuncDefOp funcOp)
      : state(stateRef), func(funcOp), oldBlock(funcOp.getBody().front()) {
    SmallVector<Type> newInputTypes;
    for (BlockArgument arg : oldBlock.getArguments())
      newInputTypes.push_back(
          mapStructType(arg.getType(), state.dest.getContext()));
    SmallVector<Type> newResultTypes;
    auto funcType = func.getFunctionType();
    for (Type resultType : funcType.getResults())
      newResultTypes.push_back(
          mapStructType(resultType, state.dest.getContext()));

    auto newFuncType =
        FunctionType::get(state.dest.getContext(), newInputTypes, newResultTypes);

    OpBuilder::InsertionGuard guard(state.builder);
    state.builder.setInsertionPointToEnd(state.dest.getBody());
    leanFunc = state.builder.create<mlir::func::FuncOp>(
        func.getLoc(), buildLeanFunctionName(func), newFuncType);

    newBlock = leanFunc.addEntryBlock();
    unsigned newIdx = 0;
    for (BlockArgument oldArg : oldBlock.getArguments()) {
      auto newArg = newBlock->getArgument(newIdx++);
      argMapping[oldArg] = newArg;
    }
  }

  // Look up a converted argument and validate it for ZKLean use.
  // Emits errors for missing mappings or struct values that require accessors.
  Value lookupArg(Value v, Operation *userOp) {
    auto newArg = argMapping.lookup(v);
    if (!newArg) {
      if (auto *def = v.getDefiningOp())
        def->emitError("unsupported value producer for ZKLean conversion");
      else
        userOp->emitError("unsupported block argument for ZKLean conversion");
      state.hadError = true;
      return Value();
    }
    if (mlir::isa<mlir::zkleanlean::StructType>(newArg.getType())) {
      userOp->emitError("struct values must be accessed via zkleanlean.accessor");
      state.hadError = true;
      return Value();
    }
    return newArg;
  }

  // Map a felt-producing value into a `mlir::zkexpr::LiteralOp`.
  // Caches the result so multiple uses share the same literal.
  Value mapValue(Value v, Operation *userOp) {
    if (auto it = zkValues.find(v); it != zkValues.end())
      return it->second;
    Value newArg = lookupArg(v, userOp);
    if (!newArg)
      return Value();
    OpBuilder::InsertionGuard guard(state.builder);
    state.builder.setInsertionPointToEnd(newBlock);
    auto literal = state.builder.create<mlir::zkexpr::LiteralOp>(
        v.getLoc(), state.zkType, newArg);
    zkValues[v] = literal.getOutput();
    return literal.getOutput();
  }

  // Map a Lean-side operand into the new function's arguments.
  // Reuses `lookupArg` so error handling is consistent.
  Value mapLeanValue(Value v, Operation *userOp) {
    if (auto it = leanValues.find(v); it != leanValues.end())
      return it->second;
    return lookupArg(v, userOp);
  }

  // Lower a single LLZK operation into ZKLean/ZKExpr/ZKBuilder IR.
  // Uses `mapValue`/`mapLeanValue` for operands and updates SSA maps.
  void convertOperation(Operation *op) {
    // Convert felt.const to ZKExpr.Literal
    if (auto constOp = dyn_cast<llzk::felt::FeltConstantOp>(op)) {
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      auto newConst = state.builder.create<llzk::felt::FeltConstantOp>(
          constOp.getLoc(), constOp.getResult().getType(),
          constOp.getValueAttr());
      auto literal = state.builder.create<mlir::zkexpr::LiteralOp>(
          constOp.getLoc(), state.zkType, newConst.getResult());
      zkValues[constOp.getResult()] = literal.getOutput();
      return;
    }
    // Convert felt.add to ZKExpr.Add
    if (auto add = dyn_cast<llzk::felt::AddFeltOp>(op)) {
      Value lhs = mapValue(add.getLhs(), op);
      Value rhs = mapValue(add.getRhs(), op);
      if (!lhs || !rhs)
        return;
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      auto zkAdd =
          state.builder.create<mlir::zkexpr::AddOp>(add.getLoc(), lhs, rhs);
      zkValues[add.getResult()] = zkAdd.getOutput();
      return;
    }
    // Convert felt.sub to ZKExpr.Sub
    if (auto sub = dyn_cast<llzk::felt::SubFeltOp>(op)) {
      Value lhs = mapValue(sub.getLhs(), op);
      Value rhs = mapValue(sub.getRhs(), op);
      if (!lhs || !rhs)
        return;
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      auto zkSub =
          state.builder.create<mlir::zkexpr::SubOp>(sub.getLoc(), lhs, rhs);
      zkValues[sub.getResult()] = zkSub.getOutput();
      return;
    }
    // Convert felt.mul to ZKExpr.Mul
    if (auto mul = dyn_cast<llzk::felt::MulFeltOp>(op)) {
      Value lhs = mapValue(mul.getLhs(), op);
      Value rhs = mapValue(mul.getRhs(), op);
      if (!lhs || !rhs)
        return;
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      auto zkMul =
          state.builder.create<mlir::zkexpr::MulOp>(mul.getLoc(), lhs, rhs);
      zkValues[mul.getResult()] = zkMul.getOutput();
      return;
    }
    // Convert felt.neg to ZKExpr.Neg
    if (auto neg = dyn_cast<llzk::felt::NegFeltOp>(op)) {
      Value operand = mapValue(neg.getOperand(), op);
      if (!operand)
        return;
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      auto zkNeg =
          state.builder.create<mlir::zkexpr::NegOp>(neg.getLoc(), operand);
      zkValues[neg.getResult()] = zkNeg.getOutput();
      return;
    }
    // Convert bool.cmp to ZKLeanLean.call
    if (auto cmp = dyn_cast<llzk::boolean::CmpOp>(op)) {
      Value lhs = mapValue(cmp.getLhs(), op);
      Value rhs = mapValue(cmp.getRhs(), op);
      if (!lhs || !rhs)
        return;
      std::string callee = "bool.cmp_";
      callee.append(cmpPredicateSuffix(cmp.getPredicate()));
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      auto call = state.builder.create<mlir::zkleanlean::CallOp>(
          cmp.getLoc(), state.builder.getI1Type(),
          SymbolRefAttr::get(state.dest.getContext(), callee),
          ValueRange{lhs, rhs});
      leanValues[cmp.getResult()] = call.getResult(0);
      return;
    }
    // Convert cast.tofelt to ZKLeanLean.call
    if (auto cast = dyn_cast<llzk::cast::IntToFeltOp>(op)) {
      Value value = mapLeanValue(cast.getValue(), op);
      if (!value)
        return;
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      auto call = state.builder.create<mlir::zkleanlean::CallOp>(
          cast.getLoc(), state.zkType,
          SymbolRefAttr::get(state.dest.getContext(), "cast.tofelt"),
          ValueRange{value});
      zkValues[cast.getResult()] = call.getResult(0);
      return;
    }
    // Convert struct.readf to ZKLeanLean.accessor
    if (auto read = dyn_cast<llzk::component::FieldReadOp>(op)) {
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      Value component = argMapping.lookup(read.getComponent());
      if (!component) {
        read.emitError("unsupported struct source in ZKLean conversion");
        state.hadError = true;
        return;
      }
      auto accessorOp = state.builder.create<mlir::zkleanlean::AccessorOp>(
          read.getLoc(), state.zkType, component, read.getFieldNameAttr());
      zkValues[read.getResult()] = accessorOp.getValue();
      return;
    }
    // Convert constrain.eq to ZKBuilder.ConstrainEq
    if (auto eq = dyn_cast<llzk::constrain::EmitEqualityOp>(op)) {
      Value lhs = mapValue(eq.getLhs(), op);
      Value rhs = mapValue(eq.getRhs(), op);
      if (!lhs || !rhs)
        return;
      auto stateType =
          mlir::zkbuilder::ZKBuilderStateType::get(state.dest.getContext());
      OpBuilder::InsertionGuard guard(state.builder);
      state.builder.setInsertionPointToEnd(newBlock);
      state.builder.create<mlir::zkbuilder::ConstrainEqOp>(eq.getLoc(),
                                                          stateType, lhs, rhs);
      return;
    }
    if (op->getNumResults() != 0) {
      op->emitError("unsupported op in ZKLean conversion");
      state.hadError = true;
    }
  }

  // Emit the final `mlir::func::ReturnOp` for the new function.
  // Assumes conversion errors were already reported.
  void finalize() {
    OpBuilder::InsertionGuard guard(state.builder);
    state.builder.setInsertionPointToEnd(newBlock);
    state.builder.create<mlir::func::ReturnOp>(func.getLoc());
  }
};

// Convert a single `llzk::function::FuncDefOp` into ZKLean form.
// Skips non-constrain functions and stops early on errors.
static bool convertFunction(llzk::function::FuncDefOp func,
                            LLZKToZKLeanState &state) {
  if (state.hadError)
    return false;
  if (!shouldConvertFunc(func))
    return false;

  FunctionConverter converter(state, func);
  auto ops = collectBlockOps(converter.oldBlock);
  for (Operation *op : ops)
    converter.convertOperation(op);
  if (state.hadError)
    return false;
  converter.finalize();
  return true;
}

// Convert all eligible LLZK functions into a new ZKLean module.
// Emits struct defs first, then lowers each function via `convertFunction`.
static LogicalResult convertModule(ModuleOp source, ModuleOp dest) {
  OpBuilder builder(dest.getContext());
  auto zkType = mlir::zkexpr::ZKExprType::get(dest.getContext());
  bool createdAny = false;
  bool hadError = false;

  LLZKToZKLeanState state{dest, builder, zkType, hadError};
  emitZKLeanStructDefs(source, state);

  source.walk([&](llzk::function::FuncDefOp func) {
    if (convertFunction(func, state))
      createdAny = true;
  });

  if (hadError)
    return failure();
  return success(createdAny);
}

// Pass wrapper that appends a converted ZKLean module to the source.
// Delegates all conversion logic to `convertModule`.
class ConvertLLZKToZKLeanPass
    : public llzk::impl::ConvertLLZKToZKLeanPassBase<
          ConvertLLZKToZKLeanPass> {
public:
  // Register dialects needed by the generated ZKLean IR.
  // Keeps the pass self-contained for `mlir::PassManager`.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<llzk::boolean::BoolDialect,
                    llzk::cast::CastDialect,
                    mlir::func::FuncDialect,
                    mlir::zkbuilder::ZKBuilderDialect,
                    mlir::zkexpr::ZKExprDialect,
                    mlir::zkleanlean::ZKLeanLeanDialect>();
  }

  // Replace the current `ModuleOp` with the ZKLean module.
  // Propagates language attributes and signals pass failure on errors.
  void runOnOperation() override {
    ModuleOp original = getOperation();
    ModuleOp zkLeanModule = ModuleOp::create(original.getLoc());
    auto symName = StringAttr::get(&getContext(), "ZKLean");
    zkLeanModule->setAttr(SymbolTable::getSymbolAttrName(), symName);
    if (auto lang = original->getAttr("veridise.lang"))
      zkLeanModule->setAttr("veridise.lang", lang);

    if (failed(convertModule(original, zkLeanModule))) {
      original.emitError("failed to produce ZKLean module");
      signalPassFailure();
      return;
    }
    original->setAttrs(zkLeanModule->getAttrs());
    original.getBodyRegion().takeBody(zkLeanModule.getBodyRegion());
  }
};

} // namespace

namespace llzk {

// Pass factory for `ConvertLLZKToZKLeanPass`.
// Used by pass registration and external callers.
std::unique_ptr<Pass> createConvertLLZKToZKLeanPass() {
  return std::make_unique<ConvertLLZKToZKLeanPass>();
}

// Register conversion passes with MLIR's global pass registry.
// Includes both `createConvertLLZKToZKLeanPass` and `createConvertZKLeanToLLZKPass`.
void registerConversionPasses() {
  ::mlir::registerPass([] { return createConvertLLZKToZKLeanPass(); });
  ::mlir::registerPass([] { return createConvertZKLeanToLLZKPass(); });
}

} // namespace llzk
