//===-- LLZKToZKLean.cpp ----------------------------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "llzk/Conversions/Passes.h"

#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.h"
#include "llzk/Dialect/ZKExpr/IR/ZKExprOps.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/DenseMap.h>

using namespace mlir;

namespace llzk {
#define GEN_PASS_DEF_CONVERTLLZKTOZKLEANPASS
#include "llzk/Conversions/LLZKConversionPasses.h.inc"
} // namespace llzk

namespace {

// Create name for Lean function from llzk function full namepath, e.g.
// "struct.def @IsZero { ... function.def @constrain ..." -> "@IsZero__constrain ..."
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

// Build ZKLean IR versions of constraint functions 
// from `source` module and insert into `dest` module.
static LogicalResult convertModule(ModuleOp source, ModuleOp dest) {
  OpBuilder builder(dest.getContext());
  auto zkType = mlir::zkexpr::ZKExprType::get(dest.getContext());
  bool createdAny = false;
  bool hadError = false;

  source.walk([&](llzk::function::FuncDefOp func) {
    if (hadError)
      return;
    if (func.getBody().empty())
      return;

    // ZKLean only interested in constrains so LLZK module must permit
    // constraints
    if (!func->hasAttr("function.allow_constraint")) 
      return;

    // Snapshot the original block so we can iterate without mutating in place.
    Block &oldBlock = func.getBody().front();

    // Copy ops from original block into ops
    SmallVector<Operation *, 16> ops;
    for (Operation &op : oldBlock)
      ops.push_back(&op);

    // Copy types of arguments from original block
    SmallVector<Type> newInputTypes;
    for (BlockArgument arg : oldBlock.getArguments()) {
      if (mlir::isa<llzk::component::StructType>(arg.getType()))
        continue;
      newInputTypes.push_back(arg.getType());
    }

    auto funcType = FunctionType::get(dest.getContext(), newInputTypes,
                                      func.getFunctionType().getResults());

    // Create new ZKLean IR function in a place after original block
    builder.setInsertionPointToEnd(dest.getBody());
    auto leanFunc =
        builder.create<llzk::function::FuncDefOp>(func.getLoc(),
                                                  buildLeanFunctionName(func),
                                                  funcType);

    // Enable constraints for the new ZKLean IR function 
    leanFunc.setAllowConstraintAttr(true);

    // Create new block 
    auto *newBlock = new Block();
    leanFunc.getBody().push_front(newBlock);

    DenseMap<Value, Value> zkValues;
    DenseMap<Value, Value> argMapping;

    unsigned newIdx = 0;

    // Drop struct-typed arguments from the Lean signature.
    for (BlockArgument oldArg : oldBlock.getArguments()) {
      if (mlir::isa<llzk::component::StructType>(oldArg.getType()))
        continue;
      auto newArg =
          newBlock->addArgument(newInputTypes[newIdx++], oldArg.getLoc());
      argMapping[oldArg] = newArg;
    }

    // Map a felt SSA value to its Lean/ZKExpr equivalent.
    auto mapValue = [&](Value v, Operation *userOp) -> Value {
      if (auto it = zkValues.find(v); it != zkValues.end())
        return it->second;

      auto newArg = argMapping.lookup(v);
      if (!newArg)
        {
          if (auto *def = v.getDefiningOp())
            def->emitError("unsupported value producer for ZKLean conversion");
          else
            userOp->emitError("unsupported block argument for ZKLean conversion");
          hadError = true;
          return Value();
        }

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(newBlock);
      auto literal =
          builder.create<mlir::zkexpr::LiteralOp>(v.getLoc(), zkType, newArg);
      zkValues[v] = literal.getOutput();
      return literal.getOutput();
    };

    for (Operation *op : ops) {
      // Convert felt.const to ZKExpr.Literal
      if (auto constOp = dyn_cast<llzk::felt::FeltConstantOp>(op)) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto newConst = builder.create<llzk::felt::FeltConstantOp>(
            constOp.getLoc(), constOp.getResult().getType(),
            constOp.getValueAttr());
        auto literal =
            builder.create<mlir::zkexpr::LiteralOp>(constOp.getLoc(), zkType,
                                                    newConst.getResult());
        zkValues[constOp.getResult()] = literal.getOutput();
        continue;
      }
      // Convert felt.add to ZKExpr.Add
      if (auto add = dyn_cast<llzk::felt::AddFeltOp>(op)) {
        Value lhs = mapValue(add.getLhs(), op);
        Value rhs = mapValue(add.getRhs(), op);
        if (!lhs || !rhs)
          continue;
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto zkAdd =
            builder.create<mlir::zkexpr::AddOp>(add.getLoc(), lhs, rhs);
        zkValues[add.getResult()] = zkAdd.getOutput();
        continue;
      }
      // Convert felt.sub to ZKExpr.Sub
      if (auto sub = dyn_cast<llzk::felt::SubFeltOp>(op)) {
        Value lhs = mapValue(sub.getLhs(), op);
        Value rhs = mapValue(sub.getRhs(), op);
        if (!lhs || !rhs)
          continue;
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto zkSub =
            builder.create<mlir::zkexpr::SubOp>(sub.getLoc(), lhs, rhs);
        zkValues[sub.getResult()] = zkSub.getOutput();
        continue;
      }
      // Convert felt.mul to ZKExpr.Mul
      if (auto mul = dyn_cast<llzk::felt::MulFeltOp>(op)) {
        Value lhs = mapValue(mul.getLhs(), op);
        Value rhs = mapValue(mul.getRhs(), op);
        if (!lhs || !rhs)
          continue;
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto zkMul =
            builder.create<mlir::zkexpr::MulOp>(mul.getLoc(), lhs, rhs);
        zkValues[mul.getResult()] = zkMul.getOutput();
        continue;
      }
      // Convert felt.neg to ZKExpr.Neg
      if (auto neg = dyn_cast<llzk::felt::NegFeltOp>(op)) {
        Value operand = mapValue(neg.getOperand(), op);
        if (!operand)
          continue;
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto zkNeg =
            builder.create<mlir::zkexpr::NegOp>(neg.getLoc(), operand);
        zkValues[neg.getResult()] = zkNeg.getOutput();
        continue;
      }
      // Convert struct.readf to ZKExpr.Witnessable.witness
      if (auto read = dyn_cast<llzk::component::FieldReadOp>(op)) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto witness = builder.create<mlir::zkexpr::WitnessOp>(read.getLoc());
        zkValues[read.getResult()] = witness.getOutput();
        continue;
      }
      // Convert constrain.eq to ZKBuilder.ConstrainEq
      if (auto eq = dyn_cast<llzk::constrain::EmitEqualityOp>(op)) {
        Value lhs = mapValue(eq.getLhs(), op);
        Value rhs = mapValue(eq.getRhs(), op);
        if (!lhs || !rhs)
          continue;
        auto stateType =
            mlir::zkbuilder::ZKBuilderStateType::get(dest.getContext());
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        builder.create<mlir::zkbuilder::ConstrainEqOp>(eq.getLoc(), stateType,
                                                       lhs, rhs);
        continue;
      }
      if (op->getNumResults() != 0) {
        op->emitError("unsupported op in ZKLean conversion");
        hadError = true;
        continue;
      }
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(newBlock);

    // TODO: Consider more nuanced handling of return
    // Return Op at the end of function 
    builder.create<llzk::function::ReturnOp>(func.getLoc());
    createdAny = true;
  });

  if (hadError)
    return failure();
  return success(createdAny);
}

class ConvertLLZKToZKLeanPass
    : public llzk::impl::ConvertLLZKToZKLeanPassBase<
          ConvertLLZKToZKLeanPass> {
public:
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
    original.getBody()->push_back(zkLeanModule.getOperation());
  }
};

} // namespace

namespace llzk {

std::unique_ptr<Pass> createConvertLLZKToZKLeanPass() {
  return std::make_unique<ConvertLLZKToZKLeanPass>();
}

void registerConversionPasses() {
  ::mlir::registerPass([] { return createConvertLLZKToZKLeanPass(); });
  ::mlir::registerPass([] { return createConvertZKLeanToLLZKPass(); });
}

} // namespace llzk
