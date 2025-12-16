//===-- ZKLeanToLLZK.cpp ---------------------------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "llzk/Conversions/Passes.h"

#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.h"
#include "llzk/Dialect/ZKExpr/IR/ZKExprOps.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/DenseMap.h>

using namespace mlir;

namespace llzk {
#define GEN_PASS_DEF_CONVERTZKLEANTOLLZKPASS
#include "llzk/Conversions/LLZKConversionPasses.h.inc"
} // namespace llzk

namespace {

// Create new @LLZK module from ZKLean dialects
static LogicalResult convertLeanModule(ModuleOp source, ModuleOp dest) {
  OpBuilder builder(dest.getContext());

  source.walk([&](llzk::function::FuncDefOp func) {
    if (func.getBody().empty())
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
    SmallVector<Type> inputTypes(func.getFunctionType().getInputs().begin(),
                                 func.getFunctionType().getInputs().end());
    auto feltType = llzk::felt::FeltType::get(dest.getContext());
    // Add an input of felt type for each ZKExpr witness
    inputTypes.append(witnesses.size(), feltType);
    // Create new function type with input types from above
    auto funcType =
        FunctionType::get(dest.getContext(), inputTypes,
                          func.getFunctionType().getResults());
    builder.setInsertionPointToEnd(dest.getBody());
    // Create new LLZK function.def
    auto newFunc =
        builder.create<llzk::function::FuncDefOp>(func.getLoc(),
                                                  func.getSymName(), funcType);
    // Allow constraints / witnesses if original function did
    if (func.hasAllowConstraintAttr())
      newFunc.setAllowConstraintAttr(true);
    if (func.hasAllowWitnessAttr())
      newFunc.setAllowWitnessAttr(true);

    auto *newBlock = new Block();
    newFunc.getBody().push_front(newBlock);

    DenseMap<Value, Value> feltValueMap;
    DenseMap<Value, Value> zkToFeltMap;
    DenseMap<Value, Value> argMap;
    DenseMap<Operation *, Value> witnessArgs;

    // Map original block arguments to the new block arguments (for non-witness inputs)
    for (auto [idx, oldArg] : llvm::enumerate(oldBlock.getArguments())) {
      auto newArg = newBlock->addArgument(oldArg.getType(), oldArg.getLoc());
      argMap[oldArg] = newArg;
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
  });

  return success();
}

class ConvertZKLeanToLLZKPass
    : public llzk::impl::ConvertZKLeanToLLZKPassBase<
          ConvertZKLeanToLLZKPass> {
public:
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
