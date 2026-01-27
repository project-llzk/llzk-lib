//===-- ZKLeanToLLZK.cpp ---------------------------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "llzk/Conversions/Passes.h"

#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.h"
#include "llzk/Dialect/ZKExpr/IR/ZKExprOps.h"
#include "llzk/Dialect/ZKLeanStruct/IR/ZKLeanStructOps.h"
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

using namespace mlir;

namespace llzk {
#define GEN_PASS_DEF_CONVERTZKLEANTOLLZKPASS
#include "llzk/Conversions/LLZKConversionPasses.h.inc"
} // namespace llzk

namespace {

// Create new @LLZK module from ZKLean dialects
static LogicalResult convertLeanModule(ModuleOp source, ModuleOp dest) {
  OpBuilder builder(dest.getContext());
  bool hadError = false;
  auto mapType = [&](Type type) -> Type {
    if (auto structType = dyn_cast<mlir::zkleanstruct::StructType>(type)) {
      return llzk::component::StructType::get(structType.getNameRef());
    }
    return type;
  };

  // Track per-struct conversion state to decide where to place functions.
  struct StructState {
    llzk::component::StructDefOp def;
    bool hasCompute = false;
    bool hasConstrain = false;
  };

  // Pre-pass: materialize LLZK struct.defs from ZKLeanStruct.defs first so
  // struct.type<@Name> symbols exist before function conversion.
  llvm::StringSet<> seenStructs;
  llvm::StringMap<StructState> structStates;
  auto feltType = llzk::felt::FeltType::get(dest.getContext());
  for (auto def : source.getBody()->getOps<mlir::zkleanstruct::StructDefOp>()) {
    StringRef name = def.getSymName();
    if (!seenStructs.insert(name).second)
      continue;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(dest.getBody());
    auto structDef = builder.create<llzk::component::StructDefOp>(
        def.getLoc(), def.getSymNameAttr(), ArrayAttr());
    structDef.getBodyRegion().emplaceBlock();
    auto &body = structDef.getBodyRegion().front();
    OpBuilder fieldBuilder(&body, body.begin());
    for (auto field :
         def.getBody()->getOps<mlir::zkleanstruct::FieldDefOp>()) {
      fieldBuilder.create<llzk::component::FieldDefOp>(
          field.getLoc(), field.getSymName(), feltType);
    }
    StructState state;
    state.def = structDef;
    state.hasCompute = structDef.getComputeFuncOp() != nullptr;
    state.hasConstrain = structDef.getConstrainFuncOp() != nullptr;
    structStates.try_emplace(name, state);
  }

  // Synthesize a compute stub when ZKLean has no witness-generation body.
  auto ensureComputeStub = [&](StructState &state,
                               ArrayRef<Type> baseInputTypes, Location loc) {
    if (state.hasCompute)
      return;
    SmallVector<Type> computeInputs;
    if (!baseInputTypes.empty() &&
        mlir::isa<llzk::component::StructType>(baseInputTypes.front())) {
      computeInputs.append(baseInputTypes.begin() + 1,
                           baseInputTypes.end());
    } else {
      computeInputs.append(baseInputTypes.begin(), baseInputTypes.end());
    }
    auto structType = state.def.getType();
    auto computeType = FunctionType::get(dest.getContext(), computeInputs,
                                         ArrayRef<Type>{structType});
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&state.def.getBodyRegion().front());
    // Stub compute: return an empty struct instance when ZKLean has no compute.
    auto computeFunc = builder.create<llzk::function::FuncDefOp>(
        loc, builder.getStringAttr(llzk::FUNC_NAME_COMPUTE), computeType);
    computeFunc.setAllowWitnessAttr(true);
    Block *computeBlock = computeFunc.addEntryBlock();
    OpBuilder bodyBuilder = OpBuilder::atBlockEnd(computeBlock);
    auto selfVal =
        bodyBuilder.create<llzk::component::CreateStructOp>(loc, structType);
    bodyBuilder.create<llzk::function::ReturnOp>(loc, selfVal.getResult());
    state.hasCompute = true;
  };

  // Synthesize a constrain stub to satisfy LLZK struct verification rules.
  auto ensureConstrainStub = [&](StructState &state, Location loc) {
    if (state.hasConstrain)
      return;
    auto structType = state.def.getType();
    auto constrainType =
        FunctionType::get(dest.getContext(), ArrayRef<Type>{structType}, {});
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&state.def.getBodyRegion().front());
    auto constrainFunc = builder.create<llzk::function::FuncDefOp>(
        loc, builder.getStringAttr(llzk::FUNC_NAME_CONSTRAIN), constrainType);
    constrainFunc.setAllowConstraintAttr(true);
    Block *constrainBlock = constrainFunc.addEntryBlock();
    OpBuilder bodyBuilder = OpBuilder::atBlockEnd(constrainBlock);
    bodyBuilder.create<llzk::function::ReturnOp>(loc);
    state.hasConstrain = true;
  };

  auto convertFunc = [&](auto func, bool allowWitness) {
    if (func.getBody().empty())
      return;

    if (hadError)
      return;

    Block &oldBlock = func.getBody().front();
    // Collect ZKLean ops into ops
    SmallVector<Operation *, 16> ops;
    // Collect ZKExpr.Witnessable.witness into witnesses
    SmallVector<mlir::zkexpr::WitnessOp, 8> witnesses;
    for (Operation &op : oldBlock) {
      ops.push_back(&op);
      if (auto witness = dyn_cast<mlir::zkexpr::WitnessOp>(&op))
        witnesses.push_back(witness);
    }

    // Collect input types of original ZKLean function
    SmallVector<Type> baseInputTypes;
    auto funcType = func.getFunctionType();
    for (Type type : funcType.getInputs())
      baseInputTypes.push_back(mapType(type));
    SmallVector<Type> inputTypes = baseInputTypes;
    // Add an input of felt type for each ZKExpr witness
    inputTypes.append(witnesses.size(), feltType);
    // Create new function type with input types from above
    auto newFuncType =
        FunctionType::get(dest.getContext(), inputTypes,
                          funcType.getResults());
    llzk::function::FuncDefOp newFunc;
    StructState *structState = nullptr;
    StringRef funcName = func.getSymName();
    // Prefer the self parameter type to bind a function to its struct; use
    // the name only as a consistency check if it follows Struct__constrain.
    std::optional<StringRef> nameStruct;
    std::optional<StringRef> typeStruct;
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
          return;
        }
      }
    }
    if (nameStruct && typeStruct && *nameStruct != *typeStruct) {
      func.emitError("struct name mismatch between function name and type");
      hadError = true;
      return;
    }
    if (nameStruct && !typeStruct) {
      func.emitError("constrain name requires a struct-typed first argument");
      hadError = true;
      return;
    }

    if (structState) {
      if (structState->hasConstrain) {
        func.emitError("duplicate constrain function for struct");
        hadError = true;
        return;
      }
      // Ensure the struct has a compute body so it validates as a component.
      ensureComputeStub(*structState, baseInputTypes, func.getLoc());
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&structState->def.getBodyRegion().front());
      newFunc = builder.create<llzk::function::FuncDefOp>(
          func.getLoc(), builder.getStringAttr(llzk::FUNC_NAME_CONSTRAIN),
          newFuncType);
      structState->hasConstrain = true;
    } else {
      builder.setInsertionPointToEnd(dest.getBody());
      newFunc = builder.create<llzk::function::FuncDefOp>(
          func.getLoc(), func.getSymName(), newFuncType);
    }
    // Allow constraints 
    newFunc.setAllowConstraintAttr(true);
    // Allow witnesses if original function permitted witnesses
    if (allowWitness)
      newFunc.setAllowWitnessAttr(true);

    auto *newBlock = new Block();
    newFunc.getBody().push_front(newBlock);

    DenseMap<Value, Value> feltValueMap;
    DenseMap<Value, Value> zkToFeltMap;
    DenseMap<Value, Value> argMap;
    DenseMap<Operation *, Value> witnessArgs;

    // Map original block arguments to the new block arguments (for non-witness inputs)
    for (auto [idx, oldArg] : llvm::enumerate(oldBlock.getArguments())) {
      auto newArg = newBlock->addArgument(inputTypes[idx], oldArg.getLoc());
      argMap[oldArg] = newArg;
      if (mlir::isa<llzk::felt::FeltType>(oldArg.getType()))
        feltValueMap[oldArg] = newArg;
    }

    // Assign each ZKExpr witness to a new felt argument
    for (auto witness : witnesses) {
      auto newArg = newBlock->addArgument(feltType, witness.getLoc());
      witnessArgs[witness.getOperation()] = newArg;
    }

    // Map felts to felts
    auto mapFelt = [&](Value v) -> Value {
      if (auto it = feltValueMap.find(v); it != feltValueMap.end())
        return it->second;
      if (auto blockArg = mlir::dyn_cast<BlockArgument>(v))
        return argMap.lookup(blockArg);
      return Value();
    };

    // Map felt value to corresponding ZKExpr result
    auto mapZK = [&](Value v) -> Value {
      if (auto it = zkToFeltMap.find(v); it != zkToFeltMap.end())
        return it->second;
      return Value();
    };

    for (Operation *op : ops) {
      // Felt constants to felt constants
      if (auto constOp = dyn_cast<llzk::felt::FeltConstantOp>(op)) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto cloned = builder.create<llzk::felt::FeltConstantOp>(
            constOp.getLoc(), constOp.getResult().getType(),
            constOp.getValueAttr());
        feltValueMap[constOp.getResult()] = cloned.getResult();
        continue;
      }

      // ZKExpr literal maps to underlying felt value
      if (auto literal = dyn_cast<mlir::zkexpr::LiteralOp>(op)) {
        Value mapped = mapFelt(literal.getLiteral());
        zkToFeltMap[literal.getOutput()] = mapped;
        continue;
      }

      // ZKExpr witness maps to dedicated witness argument
      if (auto witness = dyn_cast<mlir::zkexpr::WitnessOp>(op)) {
        Value arg = witnessArgs.lookup(witness.getOperation());
        zkToFeltMap[witness.getOutput()] = arg;
        continue;
      }

      // ZKLeanStruct.readf to struct.readf
      if (auto read = dyn_cast<mlir::zkleanstruct::ReadOp>(op)) {
        Value component = argMap.lookup(read.getComponent());
        if (!component) {
          read.emitError("unsupported struct source in ZKLean conversion");
          continue;
        }
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto fieldAttr =
            builder.getStringAttr(read.getFieldNameAttr().getValue());
        auto newRead = builder.create<llzk::component::FieldReadOp>(
            read.getLoc(), feltType, component, fieldAttr);
        zkToFeltMap[read.getValue()] = newRead.getVal();
        continue;
      }

      // ZKExpr.Add to felt.add
      if (auto add = dyn_cast<mlir::zkexpr::AddOp>(op)) {
        Value lhs = mapZK(add.getLhs());
        Value rhs = mapZK(add.getRhs());
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto feltAdd =
            builder.create<llzk::felt::AddFeltOp>(add.getLoc(), lhs, rhs);
        zkToFeltMap[add.getOutput()] = feltAdd.getResult();
        continue;
      }

      // ZKExpr.Mul to felt.mul
      if (auto mul = dyn_cast<mlir::zkexpr::MulOp>(op)) {
        Value lhs = mapZK(mul.getLhs());
        Value rhs = mapZK(mul.getRhs());
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto feltMul =
            builder.create<llzk::felt::MulFeltOp>(mul.getLoc(), lhs, rhs);
        zkToFeltMap[mul.getOutput()] = feltMul.getResult();
        continue;
      }

      // ZKExpr.Neg to felt.Neg
      if (auto neg = dyn_cast<mlir::zkexpr::NegOp>(op)) {
        Value operand = mapZK(neg.getValue());
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto feltNeg =
            builder.create<llzk::felt::NegFeltOp>(neg.getLoc(), operand);
        zkToFeltMap[neg.getOutput()] = feltNeg.getResult();
        continue;
      }

      // ZKBuilder.ConstrainEq to constrain.eq
      if (auto constraint = dyn_cast<mlir::zkbuilder::ConstrainEqOp>(op)) {
        Value lhs = mapZK(constraint.getLhs());
        Value rhs = mapZK(constraint.getRhs());
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        builder.create<llzk::constrain::EmitEqualityOp>(constraint.getLoc(),
                                                        lhs, rhs);
        continue;
      }
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(newBlock);
    builder.create<llzk::function::ReturnOp>(func.getLoc());
  };

  source.walk([&](mlir::func::FuncOp func) {
    convertFunc(func, func->hasAttr("function.allow_witness"));
  });
  source.walk([&](llzk::function::FuncDefOp func) {
    convertFunc(func, func.hasAllowWitnessAttr());
  });

  // Any struct without a converted constrain function gets stubs.
  for (auto &entry : structStates) {
    StructState &state = entry.getValue();
    if (!state.hasConstrain) {
      ensureComputeStub(state, {}, state.def.getLoc());
      ensureConstrainStub(state, state.def.getLoc());
    }
  }

  if (hadError)
    return failure();
  return success();
}

class ConvertZKLeanToLLZKPass
    : public llzk::impl::ConvertZKLeanToLLZKPassBase<
          ConvertZKLeanToLLZKPass> {
public:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<llzk::constrain::ConstrainDialect,
                    llzk::felt::FeltDialect,
                    llzk::function::FunctionDialect,
                    llzk::component::StructDialect,
                    mlir::func::FuncDialect,
                    mlir::zkexpr::ZKExprDialect,
                    mlir::zkleanstruct::ZKLeanStructDialect>();
  }

  void runOnOperation() override {
    ModuleOp source = getOperation();
    ModuleOp llzkModule = ModuleOp::create(source.getLoc());
    auto symName = StringAttr::get(&getContext(), "LLZK");
    llzkModule->setAttr(SymbolTable::getSymbolAttrName(), symName);
    if (auto lang = source->getAttr("veridise.lang"))
      llzkModule->setAttr("veridise.lang", lang);

    if (failed(convertLeanModule(source, llzkModule))) {
      source.emitError("failed to convert ZKLean module");
      signalPassFailure();
      return;
    }

    source.getBody()->push_back(llzkModule.getOperation());
  }
};

} // namespace

namespace llzk {

std::unique_ptr<Pass> createConvertZKLeanToLLZKPass() {
  return std::make_unique<ConvertZKLeanToLLZKPass>();
}

} // namespace llzk
