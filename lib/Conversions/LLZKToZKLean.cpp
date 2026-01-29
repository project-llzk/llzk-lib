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

// Build ZKLean IR versions of constraint functions 
// from `source` module and insert into `dest` module.
static LogicalResult convertModule(ModuleOp source, ModuleOp dest) {
  OpBuilder builder(dest.getContext());
  auto zkType = mlir::zkexpr::ZKExprType::get(dest.getContext());
  bool createdAny = false;
  bool hadError = false;

  // Map LLZK struct types to ZKLean struct types for ZKLean signatures.
  auto mapType = [&](Type type) -> Type {
    if (auto structType = dyn_cast<llzk::component::StructType>(type)) {
      return mlir::zkleanlean::StructType::get(dest.getContext(),
                                                 structType.getNameRef());
    }
    return type;
  };

  // Emit ZKLeanLean.def equivalents so downstream functions can reference
  // ZKLean struct types by symbol.
  llvm::DenseSet<StringRef> seenStructs;
  for (auto def : source.getBody()->getOps<llzk::component::StructDefOp>()) {
    StringRef name = def.getSymName();
    if (!seenStructs.insert(name).second)
      continue;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(dest.getBody());
    auto zkStruct =
        builder.create<mlir::zkleanlean::StructDefOp>(def.getLoc(),
                                                        def.getSymNameAttr());
    auto *body = new Block();
    zkStruct.getBodyRegion().push_back(body);
    OpBuilder fieldBuilder(body, body->begin());
    for (auto field : def.getBody()->getOps<llzk::component::FieldDefOp>()) {
      if (!mlir::isa<llzk::felt::FeltType>(field.getType())) {
        field.emitError("unsupported field type for ZKLean struct conversion");
        hadError = true;
        continue;
      }
      fieldBuilder.create<mlir::zkleanlean::FieldDefOp>(
          field.getLoc(), field.getSymNameAttr(), TypeAttr::get(zkType));
    }
  }

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

    // Copy types of arguments from original block, mapping LLZK struct types to
    // ZKLean struct types.
    SmallVector<Type> newInputTypes;
    for (BlockArgument arg : oldBlock.getArguments()) {
      newInputTypes.push_back(mapType(arg.getType()));
    }
    SmallVector<Type> newResultTypes;
    auto funcType = func.getFunctionType();
    for (Type resultType : funcType.getResults()) {
      newResultTypes.push_back(mapType(resultType));
    }

    auto newFuncType =
        FunctionType::get(dest.getContext(), newInputTypes, newResultTypes);

    // Create new ZKLean IR function in a place after original block
    builder.setInsertionPointToEnd(dest.getBody());
    // Use func.func for ZKLean to allow non-LLZK types (e.g. ZKLeanLean).
    auto leanFunc = builder.create<mlir::func::FuncOp>(
        func.getLoc(), buildLeanFunctionName(func), newFuncType);

    // Create new block.
    Block *newBlock = leanFunc.addEntryBlock();

    DenseMap<Value, Value> zkValues;
    DenseMap<Value, Value> leanValues;
    DenseMap<Value, Value> argMapping;

    unsigned newIdx = 0;

    // Preserve original arguments in the ZKLean signature.
    for (BlockArgument oldArg : oldBlock.getArguments()) {
      auto newArg = newBlock->getArgument(newIdx++);
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
      if (mlir::isa<mlir::zkleanlean::StructType>(newArg.getType())) {
        userOp->emitError("struct values must be accessed via zkleanlean.accessor");
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

    auto mapLeanValue = [&](Value v, Operation *userOp) -> Value {
      if (auto it = leanValues.find(v); it != leanValues.end())
        return it->second;

      auto newArg = argMapping.lookup(v);
      if (!newArg) {
        if (auto *def = v.getDefiningOp())
          def->emitError("unsupported value producer for ZKLean conversion");
        else
          userOp->emitError("unsupported block argument for ZKLean conversion");
        hadError = true;
        return Value();
      }
      if (mlir::isa<mlir::zkleanlean::StructType>(newArg.getType())) {
        userOp->emitError("struct values must be accessed via zkleanlean.accessor");
        hadError = true;
        return Value();
      }
      return newArg;
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
      // Convert bool.cmp to ZKLeanLean.call
      if (auto cmp = dyn_cast<llzk::boolean::CmpOp>(op)) {
        Value lhs = mapValue(cmp.getLhs(), op);
        Value rhs = mapValue(cmp.getRhs(), op);
        if (!lhs || !rhs)
          continue;
        std::string callee = "bool.cmp_";
        callee.append(cmpPredicateSuffix(cmp.getPredicate()));
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto call = builder.create<mlir::zkleanlean::CallOp>(
            cmp.getLoc(), builder.getI1Type(),
            SymbolRefAttr::get(dest.getContext(), callee),
            ValueRange{lhs, rhs});
        leanValues[cmp.getResult()] = call.getResult(0);
        continue;
      }
      // Convert cast.tofelt to ZKLeanLean.call
      if (auto cast = dyn_cast<llzk::cast::IntToFeltOp>(op)) {
        Value value = mapLeanValue(cast.getValue(), op);
        if (!value)
          continue;
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        auto call = builder.create<mlir::zkleanlean::CallOp>(
            cast.getLoc(), zkType,
            SymbolRefAttr::get(dest.getContext(), "cast.tofelt"),
            ValueRange{value});
        zkValues[cast.getResult()] = call.getResult(0);
        continue;
      }
      // Convert struct.readf to ZKLeanLean.accessor
      if (auto read = dyn_cast<llzk::component::FieldReadOp>(op)) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        Value component = argMapping.lookup(read.getComponent());
        if (!component) {
          read.emitError("unsupported struct source in ZKLean conversion");
          hadError = true;
          continue;
        }
        auto accessorOp = builder.create<mlir::zkleanlean::AccessorOp>(
            read.getLoc(), zkType, component, read.getFieldNameAttr());
        zkValues[read.getResult()] = accessorOp.getValue();
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
    builder.create<mlir::func::ReturnOp>(func.getLoc());
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
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<llzk::boolean::BoolDialect,
                    llzk::cast::CastDialect,
                    mlir::func::FuncDialect,
                    mlir::zkbuilder::ZKBuilderDialect,
                    mlir::zkexpr::ZKExprDialect,
                    mlir::zkleanlean::ZKLeanLeanDialect>();
  }

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
