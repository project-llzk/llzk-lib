//===-- SymbolicConstraintEvaluation.cpp ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/ConstraintEvaluation.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Field.h"
#include "llzk/Util/SymbolLookup.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <chrono>
#include <list>
#include <map>
#include <memory>

namespace llzk::polymorphic {
#define GEN_PASS_DEF_SYMBOLICCONSTRAINTEVALUATIONPASS
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h.inc"
} // namespace llzk::polymorphic
using namespace mlir;
using namespace llzk;
using namespace llzk::component;
using namespace llzk::function;
using namespace llzk::felt;

namespace {
/// Conservatively prove that a call cannot constrain, assert, or mutate storage.
/// Cycles and unavailable definitions are deliberately treated as effectful.
class EvaluationCleanup {
  SymbolTableCollection tables;
  DenseMap<Operation *, bool> pure;

  bool isPure(FuncDefOp function) {
    if (!function || function.getBody().empty()) {
      return false;
    }
    auto [it, inserted] = pure.try_emplace(function, false);
    if (!inserted) {
      return it->second;
    }
    bool safe = true;
    function.walk([this, &function, &safe](Operation *op) {
      if (op == function.getOperation() || isa<ReturnOp, scf::YieldOp, scf::ConditionOp>(op)) {
        return;
      }
      if (auto call = dyn_cast<CallOp>(op)) {
        auto found = lookupTopLevelSymbol<FuncDefOp>(tables, call.getCalleeAttr(), call);
        safe &= !call.calleeIsConstrain() && succeeded(found) && isPure(found->get());
      } else if (isa<NonDetOp>(op)) {
        // Fresh uninitialized values are local; all uses are checked separately.
      } else if (isa<scf::IfOp, scf::ForOp, scf::WhileOp, scf::ExecuteRegionOp>(op)) {
        // Nested operations are checked independently by the walk.
      } else {
        safe &= isMemoryEffectFree(op);
      }
    });
    pure[function] = safe;
    return safe;
  }

public:
  /// Fold and remove dead computations to a fixed point without inlining/unrolling.
  LogicalResult run(ModuleOp module) {
    RewritePatternSet patterns(module.getContext());
    scf::IfOp::getCanonicalizationPatterns(patterns, module.getContext());
    NonDetOp::getCanonicalizationPatterns(patterns, module.getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    bool changed;
    do {
      if (failed(applyPatternsGreedily(module, frozen))) {
        return failure();
      }
      changed = false;
      pure.clear();
      module.walk([this, &changed](CallOp call) {
        if (!call->use_empty() || call.calleeIsConstrain()) {
          return;
        }
        auto found = lookupTopLevelSymbol<FuncDefOp>(tables, call.getCalleeAttr(), call);
        if (succeeded(found) && isPure(found->get())) {
          call.erase();
          changed = true;
        }
      });
    } while (changed);
    return success();
  }
};

/// Interpreter value. Aggregate contents are interpreter storage, never emitted IR.
/// Lazy witness locations carry a structural path, independent of emitted SSA names.
struct AbstractValue {
  Type type;
  Attribute constant;
  Value scalar;
  std::map<std::string, std::shared_ptr<AbstractValue>> children;
  SmallVector<Attribute> path;
  ArrayAttr family;
  SmallVector<int64_t> indices;
  StructDefOp definition;
  polymorphic::InstanceId instance {0};
  bool witness = false;
  bool publicPath = false;
};
using AV = std::shared_ptr<AbstractValue>;
using Env = DenseMap<Value, AV>;

/// Executes specialized bodies with independent call frames and instance storage.
class Evaluator {
  ModuleOp module;
  OpBuilder builder;
  SymbolTableCollection tables;
  FuncDefOp output;
  std::map<uint64_t, StructDefOp> definitions;
  DenseMap<Attribute, AV> signals;
  DenseMap<Attribute, Value> storage;
  DenseMap<Attribute, Type> storageTypes;
  DenseMap<Attribute, uint64_t> instanceIds;
  SmallVector<Attribute> instances, bindings, stack, loops;
  DenseMap<Attribute, Value> constants;
  std::map<std::string, Value> expressions;
  // Values are numeric: aliases of the same prime can share inverses without
  // leaking the source field name into the folded result's type.
  using InverseKey = std::pair<APInt, APInt>;
  std::list<InverseKey> inverseOrder;
  struct CachedInverse {
    APInt value;
    std::list<InverseKey>::iterator position;
  };
  DenseMap<InverseKey, CachedInverse> inverses;
  uint64_t inverseHits = 0, inverseMisses = 0;
  uint64_t retainedOperations;
  uint64_t iterations = 0, calls = 0, equations = 0, steps = 0;
  uint64_t limit;
  bool invalid = false;

  AV value(Type type) {
    auto v = std::make_shared<AbstractValue>();
    v->type = type;
    return v;
  }
  AV known(Type type, Attribute attr) {
    auto v = value(type);
    v->constant = attr;
    return v;
  }
  AV index(int64_t n) { return known(builder.getIndexType(), builder.getIndexAttr(n)); }
  LogicalResult error(Operation *op, const Twine &message) {
    invalid = true;
    return op->emitError("symbolic evaluation: ") << message;
  }
  std::optional<int64_t> integer(AV v) {
    if (!v) {
      return std::nullopt;
    }
    if (auto a = dyn_cast_or_null<IntegerAttr>(v->constant)) {
      return a.getInt();
    }
    if (auto a = dyn_cast_or_null<FeltConstAttr>(v->constant)) {
      if (a.getValue().getActiveBits() < 64) {
        return a.getValue().getZExtValue();
      }
    }
    return std::nullopt;
  }
  StructDefOp resolve(Type type, Operation *origin) {
    auto st = dyn_cast<StructType>(type);
    if (!st) {
      return {};
    }
    auto found = lookupTopLevelSymbol<StructDefOp>(tables, st.getNameRef(), origin);
    return succeeded(found) ? found->get() : StructDefOp();
  }
  /// Intern a component occurrence separately from its specialized definition.
  void component(AV v, Operation *origin) {
    if (!isa<StructType>(v->type)) {
      return;
    }
    if (!v->definition) {
      v->definition = resolve(v->type, origin);
    }
    if (!v->definition) {
      invalid = true;
      return;
    }
    auto path = builder.getArrayAttr(v->path);
    if (auto it = instanceIds.find(path); it != instanceIds.end()) {
      v->instance = {it->second};
      return;
    }
    auto parentInstance = v->instance;
    v->instance = {instances.size()};
    instanceIds[path] = v->instance.value;
    NamedAttrList info;
    info.set("id", builder.getI64IntegerAttr(v->instance.value));
    if (v->path.size() > 1) {
      info.set("parent", builder.getI64IntegerAttr(parentInstance.value));
    }
    info.set("path", builder.getArrayAttr(v->path));
    if (auto id = v->definition->getAttr("poly.specialization_id")) {
      info.set("specialization", id);
    }
    instances.push_back(builder.getDictionaryAttr(info));
  }
  /// Resolve one member/index without materializing aggregate reads in the output.
  AV child(AV parent, std::string key, Attribute segment, Type type, Operation *op) {
    if (auto it = parent->children.find(key); it != parent->children.end()) {
      return it->second;
    }
    auto result = value(type);
    if (auto elements = dyn_cast_or_null<ArrayAttr>(parent->constant)) {
      if (auto position = dyn_cast<IntegerAttr>(segment)) {
        if (auto remaining = dyn_cast<array::ArrayType>(type)) {
          int64_t stride = remaining.getNumElements();
          result->constant =
              builder.getArrayAttr(elements.getValue().slice(position.getInt() * stride, stride));
        } else {
          result->constant = elements[position.getInt()];
        }
      }
    }
    result->witness = parent->witness;
    result->publicPath = parent->publicPath;
    result->instance = parent->instance;
    result->path = parent->path;
    result->path.push_back(segment);
    if (auto podType = dyn_cast<pod::PodType>(parent->type)) {
      auto records = podType.getRecordMap();
      if (auto recordType = records.lookup(key)) {
        result->type = recordType;
      }
    }
    result->family = parent->family;
    result->indices = parent->indices;
    if (auto i = dyn_cast<IntegerAttr>(segment)) {
      result->indices.push_back(i.getInt());
    }
    if (parent->definition) {
      auto member = parent->definition.getMemberDef(builder.getStringAttr(key));
      if (!member) {
        (void)error(op, "unknown component member");
        return result;
      }
      result->type = member.getType();
      result->publicPath &= member->hasAttr("llzk.pub");
      result->family = member->getAttrOfType<ArrayAttr>("poly.family");
      result->indices.clear();
    }
    if (isa<StructType>(result->type) && result->family) {
      for (auto item : result->family) {
        auto entry = llvm::cast<DictionaryAttr>(item);
        auto indices = llvm::cast<ArrayAttr>(entry.get("indices"));
        if (indices.size() != result->indices.size()) {
          continue;
        }
        bool matches = true;
        for (auto [a, b] : llvm::zip(indices, result->indices)) {
          matches &= llvm::cast<IntegerAttr>(a).getInt() == b;
        }
        if (matches) {
          result->definition =
              definitions[llvm::cast<IntegerAttr>(entry.get("specialization")).getInt()];
        }
      }
      if (!result->definition) {
        (void)error(op, "component family has no specialization for index");
      }
    }
    component(result, op);
    if (result->witness) {
      storageTypes[builder.getArrayAttr(result->path)] = result->type;
    }
    parent->children[key] = result;
    return result;
  }
  /// Read a logical witness path directly from the original constrain arguments.
  /// Cache prefixes so repeated nested accesses share their straight-line reads.
  Value readStorage(AV v, Location loc) {
    SmallVector<Attribute> prefix {v->path.front()};
    Value current = output.getArgument(llvm::cast<IntegerAttr>(v->path.front()).getInt());
    for (size_t i = 1; i < v->path.size();) {
      Type type = current.getType();
      if (auto array = dyn_cast<array::ArrayType>(type)) {
        SmallVector<int64_t> indexValues;
        for (unsigned d = 0; d < array.getRank(); ++d) {
          if (i == v->path.size() || !isa<IntegerAttr>(v->path[i])) {
            (void)error(output, "incomplete array storage path");
            return {};
          }
          auto index = llvm::cast<IntegerAttr>(v->path[i++]);
          prefix.push_back(index);
          indexValues.push_back(index.getInt());
        }
        auto key = builder.getArrayAttr(prefix);
        if (auto cached = storage.lookup(key)) {
          current = cached;
        } else {
          SmallVector<Value> indices;
          for (int64_t index : indexValues) {
            indices.push_back(builder.create<arith::ConstantIndexOp>(loc, index));
          }
          current =
              builder.create<array::ReadArrayOp>(loc, array.getElementType(), current, indices);
          storage[key] = current;
        }
      } else {
        auto name = llvm::cast<StringAttr>(v->path[i++]);
        prefix.push_back(name);
        auto key = builder.getArrayAttr(prefix);
        if (auto cached = storage.lookup(key)) {
          current = cached;
          continue;
        }
        if (auto def = resolve(type, output)) {
          auto member = def.getMemberDef(name);
          if (!member) {
            (void)error(output, "unknown storage member");
            return {};
          }
          Type resultType = storageTypes.lookup(key);
          current = builder.create<MemberReadOp>(
              loc, resultType ? resultType : member.getType(), current, name
          );
        } else if (auto pod = dyn_cast<pod::PodType>(type)) {
          current = builder.create<pod::ReadPodOp>(
              loc, pod.getRecordMap().lookup(name.getValue()), current, name
          );
        } else {
          (void)error(output, "unsupported witness storage path");
          return {};
        }
        storage[key] = current;
      }
    }
    return current;
  }
  /// Intern one scalar expression input or constant in the generated function.
  Value materialize(AV v, Location loc) {
    if (isa<StructType, array::ArrayType, pod::PodType>(v->type)) {
      invalid = true;
      emitError(loc) << "symbolic evaluation: aggregate used as a scalar expression";
      return {};
    }
    if (v->scalar) {
      return v->scalar;
    }
    if (v->constant) {
      if (auto c = constants.lookup(v->constant)) {
        return c;
      }
      Operation *op;
      if (isa<FeltConstAttr>(v->constant)) {
        OperationState state(loc, "felt.const");
        state.addTypes(v->type);
        state.addAttribute("value", v->constant);
        op = builder.create(state);
      } else {
        op = builder.create<arith::ConstantOp>(loc, v->type, llvm::cast<TypedAttr>(v->constant));
      }
      constants[v->constant] = op->getResult(0);
      return op->getResult(0);
    }
    if (!v->witness || isa<StructType, array::ArrayType, pod::PodType>(v->type)) {
      invalid = true;
      emitError(loc) << "symbolic evaluation: read of uninitialized or non-scalar value " << v->type
                     << " path=" << builder.getArrayAttr(v->path)
                     << " calls=" << builder.getArrayAttr(stack);
      return {};
    }
    auto it = signals.find(builder.getArrayAttr(v->path));
    if (it != signals.end()) {
      return it->second->scalar;
    }
    v->scalar = readStorage(v, loc);
    if (!v->scalar) {
      return {};
    }
    signals[builder.getArrayAttr(v->path)] = v;
    NamedAttrList info;
    info.set("instance", builder.getI64IntegerAttr(v->instance.value));
    info.set("path", builder.getArrayAttr(v->path));
    auto owner = llvm::cast<DictionaryAttr>(instances[v->instance.value]);
    size_t prefix = llvm::cast<ArrayAttr>(owner.get("path")).size();
    // Main inputs are rooted in an argument, not in the main self object.
    if (v->path.front() != builder.getI64IntegerAttr(0)) {
      prefix = 0;
    }
    info.set("member_path", builder.getArrayAttr(ArrayRef(v->path).drop_front(prefix)));
    info.set("public", builder.getBoolAttr(v->publicPath));
    bindings.push_back(builder.getDictionaryAttr(info));
    if (auto *read = v->scalar.getDefiningOp()) {
      read->setAttr(polymorphic::SIGNAL_BINDING_ATTR_NAME, builder.getDictionaryAttr(info));
    }
    return v->scalar;
  }
  /// Copy aggregate values at language copy boundaries (array/POD writes).
  AV copy(AV v) {
    auto result = std::make_shared<AbstractValue>(*v);
    for (auto &[key, element] : result->children) {
      element = copy(element);
    }
    return result;
  }
  /// Fold one detached scalar op using existing dialect semantics.
  Attribute
  foldScalar(StringRef name, Operation &source, ValueRange operands, ArrayRef<Attribute> attrs) {
    OperationState state(source.getLoc(), name);
    state.addOperands(operands);
    state.addTypes(source.getResultTypes());
    Operation *temporary = Operation::create(state);
    SmallVector<OpFoldResult> results;
    Attribute result;
    if (succeeded(temporary->fold(attrs, results)) && results.size() == 1) {
      result = dyn_cast<Attribute>(results.front());
    }
    temporary->destroy();
    return result;
  }

  /// Reuse constant inverses across divisions and explicit inversions. The cache
  /// belongs to this evaluation, is capped at 8192 entries, and excludes zero.
  /// Modulus and denominator widths are normalized for numeric key equality.
  Attribute foldCachedInverse(Operation &op, ArrayRef<Attribute> attrs) {
    bool division = isa<DivFeltOp>(op);
    if (!division && !isa<InvFeltOp>(op)) {
      return {};
    }
    auto denominator = dyn_cast_or_null<FeltConstAttr>(attrs.back());
    if (!denominator || !denominator.getFieldName()) {
      return {};
    }
    if (division) {
      auto numerator = dyn_cast_or_null<FeltConstAttr>(attrs.front());
      if (!numerator || numerator.getFieldName() != denominator.getFieldName()) {
        return {};
      }
    }
    auto fieldResult = Field::tryGetField(denominator.getFieldName().getValue());
    if (failed(fieldResult)) {
      return {};
    }
    const Field &field = fieldResult->get();
    unsigned width =
        std::max(denominator.getValue().getBitWidth(), field.primeAPInt().getBitWidth());
    APInt prime = field.primeAPInt().zext(width);
    APInt divisor = denominator.getValue().zext(width);
    if (divisor.uge(prime)) {
      divisor = divisor.urem(prime);
    }
    if (divisor.isZero()) {
      return {};
    }
    width = prime.getActiveBits();
    auto key = std::make_pair(prime.zextOrTrunc(width), divisor.zextOrTrunc(width));
    FeltConstAttr inverse;
    if (auto it = inverses.find(key); it != inverses.end()) {
      ++inverseHits;
      inverseOrder.splice(inverseOrder.end(), inverseOrder, it->second.position);
      inverse = FeltConstAttr::get(op.getContext(), it->second.value, denominator.getType());
    } else {
      ++inverseMisses;
      inverse = dyn_cast_or_null<FeltConstAttr>(
          foldScalar(InvFeltOp::getOperationName(), op, op.getOperands().take_back(), {denominator})
      );
      if (!inverse) {
        return {};
      }
      // Evict only the least recently used denominator, preserving the hot set.
      if (inverses.size() >= 8192) {
        inverses.erase(inverseOrder.front());
        inverseOrder.pop_front();
      }
      inverseOrder.push_back(key);
      inverses.try_emplace(
          std::move(key), CachedInverse {inverse.getValue(), std::prev(inverseOrder.end())}
      );
    }
    if (!division) {
      return inverse;
    }
    return foldScalar(
        MulFeltOp::getOperationName(), op, op.getOperands(), {attrs.front(), inverse}
    );
  }

  /// Execute a block, returning its terminator operands in the abstract domain.
  LogicalResult block(Block &body, Env &env, SmallVectorImpl<AV> &returned) {
    for (Operation &op : body) {
      if (++steps > limit) {
        return error(&op, "evaluation step limit exceeded");
      }
      StringRef name = op.getName().getStringRef();
      if (name == "function.return" || name == "scf.yield" || name == "scf.condition") {
        for (auto operand : op.getOperands()) {
          returned.push_back(env.lookup(operand));
        }
        return success();
      }
      auto bind = [&op, &env](ArrayRef<AV> values) {
        for (auto [a, b] : llvm::zip_equal(op.getResults(), values)) {
          env[a] = b;
        }
      };
      if (auto execute = dyn_cast<scf::ExecuteRegionOp>(op)) {
        if (!llvm::hasSingleElement(execute.getRegion())) {
          return error(&op, "execute_region must have one block");
        }
        SmallVector<AV> values;
        if (failed(block(execute.getRegion().front(), env, values))) {
          return failure();
        }
        bind(values);
        continue;
      }
      if (auto read = dyn_cast<global::GlobalReadOp>(op)) {
        auto global =
            lookupTopLevelSymbol<global::GlobalDefOp>(tables, read.getNameRefAttr(), read);
        if (failed(global) || !global->get().isConstant() || !global->get().getInitialValue()) {
          return error(&op, "global must be an initialized constant");
        }
        env[op.getResult(0)] = known(op.getResult(0).getType(), global->get().getInitialValue());
        continue;
      }
      if (name == "llzk.nondet") {
        env[op.getResult(0)] = value(op.getResult(0).getType());
        continue;
      }
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        auto lower = integer(env.lookup(loop.getLowerBound())),
             upper = integer(env.lookup(loop.getUpperBound())),
             step = integer(env.lookup(loop.getStep()));
        if (!lower || !upper || !step || *step <= 0) {
          return error(&op, "for bounds and positive step must be known integers");
        }
        SmallVector<AV> carried;
        for (Value v : loop.getInitArgs()) {
          carried.push_back(env.lookup(v));
        }
        for (int64_t i = *lower; i < *upper;) {
          ++iterations;
          loops.push_back(builder.getI64IntegerAttr(i));
          env[loop.getInductionVar()] = index(i);
          for (auto [arg, v] : llvm::zip_equal(loop.getRegionIterArgs(), carried)) {
            env[arg] = v;
          }
          SmallVector<AV> next;
          if (failed(block(*loop.getBody(), env, next))) {
            return failure();
          }
          carried = std::move(next);
          loops.pop_back();
          if (i > INT64_MAX - *step) {
            return error(&op, "loop induction overflow");
          }
          i += *step;
        }
        bind(carried);
        continue;
      }
      if (auto loop = dyn_cast<scf::WhileOp>(op)) {
        SmallVector<AV> carried;
        for (Value v : loop.getInits()) {
          carried.push_back(env.lookup(v));
        }
        uint64_t loopIteration = 0;
        while (true) {
          for (auto [arg, v] : llvm::zip_equal(loop.getBeforeArguments(), carried)) {
            env[arg] = v;
          }
          SmallVector<AV> condition;
          if (failed(block(loop.getBefore().front(), env, condition))) {
            return failure();
          }
          auto test = integer(condition.front());
          if (!test) {
            return error(&op, "while condition must be statically known");
          }
          carried.assign(condition.begin() + 1, condition.end());
          if (!*test) {
            break;
          }
          ++iterations;
          loops.push_back(builder.getI64IntegerAttr(loopIteration++));
          for (auto [arg, v] : llvm::zip_equal(loop.getAfterArguments(), carried)) {
            env[arg] = v;
          }
          SmallVector<AV> next;
          if (failed(block(loop.getAfter().front(), env, next))) {
            return failure();
          }
          carried = std::move(next);
          loops.pop_back();
        }
        bind(carried);
        continue;
      }
      if (auto branch = dyn_cast<scf::IfOp>(op)) {
        auto condition = integer(env.lookup(branch.getCondition()));
        if (!condition) {
          return error(&op, "dynamic conditional is not yet supported");
        }
        Region &region = *condition ? branch.getThenRegion() : branch.getElseRegion();
        SmallVector<AV> values;
        if (!region.empty() && failed(block(region.front(), env, values))) {
          return failure();
        }
        bind(values);
        continue;
      }
      if (auto call = dyn_cast<CallOp>(op)) {
        SmallVector<AV> args;
        for (Value arg : call.getArgOperands()) {
          args.push_back(env.lookup(arg));
        }
        FuncDefOp callee;
        if (call.calleeIsConstrain() && !args.empty() && args[0]->definition) {
          callee = dyn_cast_or_null<FuncDefOp>(
              SymbolTable::lookupSymbolIn(args[0]->definition, "constrain")
          );
        } else {
          auto found = lookupTopLevelSymbol<FuncDefOp>(tables, call.getCalleeAttr(), call);
          if (failed(found)) {
            return failure();
          }
          callee = found->get();
        }
        if (!callee || !llvm::hasSingleElement(callee.getBody()) || stack.size() >= 256) {
          return error(&op, "unavailable or recursive callee");
        }
        ++calls;
        NamedAttrList frameInfo;
        frameInfo.set("callee", callee.getFullyQualifiedName());
        if (!args.empty() && args[0]->definition) {
          frameInfo.set("instance", builder.getI64IntegerAttr(args[0]->instance.value));
        }
        stack.push_back(builder.getDictionaryAttr(frameInfo));
        Env frame;
        for (auto [arg, v] : llvm::zip_equal(callee.getArguments(), args)) {
          frame[arg] = copy(v);
        }
        SmallVector<AV> values;
        if (failed(block(callee.getBody().front(), frame, values))) {
          return failure();
        }
        stack.pop_back();
        bind(values);
        continue;
      }
      if (name == "array.new" || name == "pod.new") {
        auto aggregate = value(op.getResult(0).getType());
        if (auto arr = dyn_cast<array::CreateArrayOp>(op)) {
          auto arrayType = llvm::cast<array::ArrayType>(aggregate->type);
          int64_t linear = 0;
          for (Value v : arr.getElements()) {
            AV target = aggregate;
            int64_t remainder = linear++;
            auto shape = arrayType.getShape();
            for (unsigned dimension = 0; dimension < shape.size(); ++dimension) {
              int64_t stride = 1;
              for (int64_t size : shape.drop_front(dimension + 1)) {
                if (ShapedType::isDynamic(size)) {
                  return error(&op, "initialized array requires a static shape");
                }
                stride *= size;
              }
              auto position = remainder / stride;
              remainder %= stride;
              std::string key = std::to_string(position);
              if (dimension + 1 == shape.size()) {
                target->children[key] = copy(env.lookup(v));
              } else {
                target = child(
                    target, key, builder.getIndexAttr(position),
                    arrayType.getSelectionType(dimension + 1), &op
                );
              }
            }
          }
        } else {
          auto pod = llvm::cast<pod::NewPodOp>(op);
          for (auto [key, v] : llvm::zip(pod.getInitializedRecords(), pod.getInitialValues())) {
            aggregate->children[llvm::cast<StringAttr>(key).str()] = copy(env.lookup(v));
          }
        }
        env[op.getResult(0)] = aggregate;
        continue;
      }
      if (name == "struct.readm" || name == "pod.read" || name == "pod.write") {
        auto parent = env.lookup(op.getOperand(0));
        std::string key;
        if (auto read = dyn_cast<MemberReadOp>(op)) {
          if (read.getTableOffset()) {
            return error(&op, "column offsets are unsupported");
          }
          key = read.getMemberName().str();
        } else if (auto podRead = dyn_cast<pod::ReadPodOp>(op)) {
          key = podRead.getRecordName().str();
        } else {
          key = llvm::cast<pod::WritePodOp>(op).getRecordName().str();
        }
        if (name == "pod.write") {
          parent->children[key] = copy(env.lookup(op.getOperand(1)));
        } else {
          env[op.getResult(0)] =
              copy(child(parent, key, builder.getStringAttr(key), op.getResult(0).getType(), &op));
        }
        if (invalid) {
          return failure();
        }
        continue;
      }
      if (name == "array.read" || name == "array.extract" || name == "array.write" ||
          name == "array.insert") {
        bool write = name == "array.write" || name == "array.insert";
        AV parent = env.lookup(op.getOperand(0));
        unsigned count = op.getNumOperands() - 1 - unsigned(write);
        for (unsigned n = 0; n < count; ++n) {
          auto i = integer(env.lookup(op.getOperand(n + 1)));
          auto type = dyn_cast<array::ArrayType>(parent->type);
          if (!i || !type) {
            return error(&op, "array selection requires a known index and array");
          }
          auto shape = type.getShape();
          if (ShapedType::isDynamic(shape[0])) {
            return error(&op, "array shape must be statically known");
          }
          if (*i < 0 || *i >= shape[0]) {
            return error(&op, "array index out of bounds");
          }
          std::string key = std::to_string(*i);
          if (write && n + 1 == count) {
            parent->children[key] = copy(env.lookup(op.getOperands().back()));
          } else {
            parent = child(parent, key, builder.getIndexAttr(*i), type.getSelectionType(1), &op);
          }
        }
        if (!write) {
          env[op.getResult(0)] = copy(parent);
        }
        if (invalid) {
          return failure();
        }
        continue;
      }
      if (name == "poly.unifiable_cast") {
        env[op.getResult(0)] = env.lookup(op.getOperand(0));
        continue;
      }
      SmallVector<Attribute> attrs;
      for (Value operand : op.getOperands()) {
        auto v = env.lookup(operand);
        if (!v) {
          return error(&op, "missing SSA value");
        }
        attrs.push_back(v->constant);
      }
      if (auto castOp = dyn_cast<llzk::cast::IntToFeltOp>(op)) {
        if (auto a = dyn_cast_or_null<IntegerAttr>(attrs.front())) {
          env[op.getResult(0)] = known(
              castOp.getType(), FeltConstAttr::get(op.getContext(), a.getValue(), castOp.getType())
          );
          continue;
        }
      }
      if (auto castOp = dyn_cast<llzk::cast::FeltToIndexOp>(op)) {
        if (auto n = integer(env.lookup(op.getOperand(0)))) {
          env[op.getResult(0)] = index(*n);
          continue;
        }
      }
      if (Attribute cached = foldCachedInverse(op, attrs)) {
        env[op.getResult(0)] = known(op.getResult(0).getType(), cached);
        continue;
      }
      SmallVector<OpFoldResult> folded;
      Operation *temporary = op.clone();
      auto successFold = temporary->fold(attrs, folded);
      // SSA folds refer to original operands of the clone. Propagate their abstract
      // values, never a result owned by the temporary operation being destroyed.
      SmallVector<AV> foldedValues;
      if (succeeded(successFold) && folded.size() == op.getNumResults()) {
        for (auto [result, fold] : llvm::zip_equal(op.getResults(), folded)) {
          if (auto attr = dyn_cast<Attribute>(fold)) {
            foldedValues.push_back(known(result.getType(), attr));
          } else if (auto v = env.lookup(llvm::cast<Value>(fold))) {
            foldedValues.push_back(copy(v));
          } else {
            break;
          }
        }
      }
      temporary->destroy();
      if (!foldedValues.empty() && foldedValues.size() == op.getNumResults()) {
        for (auto [result, v] : llvm::zip_equal(op.getResults(), foldedValues)) {
          env[result] = v;
        }
        continue;
      }
      if (name == "bool.assert") {
        auto condition = integer(env.lookup(op.getOperand(0)));
        if (!condition || !*condition) {
          return error(&op, "assertion is not statically true");
        }
        continue;
      }
      bool equation = name == "constrain.eq";
      if (!equation && !(isMemoryEffectFree(&op) && !op.getNumRegions() &&
                         (name.starts_with("felt.") || name.starts_with("arith.") ||
                          name.starts_with("bool.") || name.starts_with("cast.")))) {
        return error(&op, "unsupported operation '" + name + "'");
      }
      IRMapping mapping;
      for (Value operand : op.getOperands()) {
        Value scalar = materialize(env.lookup(operand), op.getLoc());
        if (!scalar) {
          return failure();
        }
        mapping.map(operand, scalar);
      }
      std::string key;
      llvm::raw_string_ostream keyStream(key);
      keyStream << name << op.getAttrDictionary() << op.getPropertiesAsAttribute();
      for (Value operand : op.getOperands()) {
        keyStream << mapping.lookup(operand).getAsOpaquePointer();
      }
      for (Type type : op.getResultTypes()) {
        keyStream << type;
      }
      if (!equation && op.getNumResults() == 1) {
        if (auto it = expressions.find(key); it != expressions.end()) {
          auto v = value(op.getResult(0).getType());
          v->scalar = it->second;
          env[op.getResult(0)] = v;
          continue;
        }
      }
      Operation *emitted = builder.clone(op, mapping);
      if (!equation && op.getNumResults() == 1) {
        expressions[key] = emitted->getResult(0);
      }
      if (equation) {
        ++equations;
        emitted->setAttr("poly.source_operation", op.getAttr("poly.evaluation_source_operation"));
        emitted->setAttr("poly.call_stack", builder.getArrayAttr(stack));
        emitted->setAttr("poly.loop_indices", builder.getArrayAttr(loops));
      }
      for (auto [a, b] : llvm::zip_equal(op.getResults(), emitted->getResults())) {
        auto v = value(a.getType());
        v->scalar = b;
        env[a] = v;
      }
    }
    return success();
  }

public:
  Evaluator(ModuleOp root, uint64_t maximum, uint64_t retained)
      : module(root), builder(root.getContext()), retainedOperations(retained), limit(maximum) {}
  /// Construct a straight-line constrain method while preserving rolled compute.
  FailureOr<FuncDefOp> run(bool report) {
    auto started = std::chrono::steady_clock::now();
    auto main = getMainInstanceType(module);
    if (failed(main) || !*main) {
      return error(module, "requires llzk.main");
    }
    auto definition = resolve(*main, module);
    if (!definition || !definition->hasAttr("poly.specialization_id")) {
      return error(module, "run llzk-monomorphize first");
    }
    module.walk([this](StructDefOp def) {
      if (auto id = def->getAttrOfType<IntegerAttr>("poly.specialization_id")) {
        definitions[id.getInt()] = def;
      }
    });
    auto source = dyn_cast_or_null<FuncDefOp>(SymbolTable::lookupSymbolIn(definition, "constrain"));
    if (!source || !llvm::hasSingleElement(source.getBody())) {
      return error(module, "main must have a single-block constrain function");
    }
    if (source->hasAttr(polymorphic::EVALUATED_ATTR_NAME)) {
      return error(module, "generated function already exists");
    }
    builder.setInsertionPointAfter(source);
    output = builder.create<FuncDefOp>(
        source.getLoc(), "__llzk_flat_constrain", source.getFunctionType()
    );
    output->setAttr("function.allow_constraint", builder.getUnitAttr());
    output->setAttr("function.allow_non_native_field_ops", builder.getUnitAttr());
    output->setAttr(polymorphic::EVALUATED_ATTR_NAME, builder.getUnitAttr());
    if (auto attrs = source->getAttr("arg_attrs")) {
      output->setAttr("arg_attrs", attrs);
    }
    output.addEntryBlock();
    builder.setInsertionPointToEnd(&output.getBody().front());
    Env env;
    for (auto [i, arg] : llvm::enumerate(source.getArguments())) {
      auto v = value(arg.getType());
      v->witness = true;
      v->publicPath = i == 0 || source.getArgAttr(i, "llzk.pub");
      v->path.push_back(builder.getI64IntegerAttr(i));
      component(v, source);
      env[arg] = v;
    }
    SmallVector<AV> returned;
    if (failed(block(source.getBody().front(), env, returned)) || invalid) {
      output.erase();
      return failure();
    }
    builder.create<ReturnOp>(source.getLoc());
    output->setAttr(polymorphic::SIGNAL_BINDINGS_ATTR_NAME, builder.getArrayAttr(bindings));
    output->setAttr(polymorphic::INSTANCES_ATTR_NAME, builder.getArrayAttr(instances));
    output->setAttr("poly.source", source.getFullyQualifiedName());
    if (report) {
      llvm::errs() << "evaluation: instances=" << instances.size() << " signals=" << signals.size()
                   << " iterations=" << iterations << " calls=" << calls
                   << " inverse_cache_hits=" << inverseHits
                   << " inverse_cache_misses=" << inverseMisses << " constraints=" << equations
                   << " dag_nodes=" << expressions.size()
                   << " emitted_operations=" << output.getBody().front().getOperations().size()
                   << " retained_operations=" << retainedOperations << " evaluation_ms="
                   << std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - started
                      )
                          .count()
                   << '\n';
    }
    return output;
  }
};

/// Evaluate on a clone; commit the new method and nested access changes only on success.
class SymbolicConstraintEvaluationPassImpl
    : public polymorphic::impl::SymbolicConstraintEvaluationPassBase<
          SymbolicConstraintEvaluationPassImpl> {
  using Base = SymbolicConstraintEvaluationPassBase<SymbolicConstraintEvaluationPassImpl>;
  using Base::Base;
  void runOnOperation() override {
    OwningOpRef<ModuleOp> temporary = llvm::cast<ModuleOp>(getOperation()->clone());
    uint64_t ordinal = 0;
    temporary->walk([this, &ordinal](Operation *op) {
      if (op->getName().getStringRef() == "constrain.eq") {
        op->setAttr(
            "poly.evaluation_source_operation",
            IntegerAttr::get(IntegerType::get(&getContext(), 64), ordinal)
        );
      }
      ++ordinal;
    });
    if (failed(EvaluationCleanup().run(*temporary))) {
      signalPassFailure();
      return;
    }
    auto evaluated = Evaluator(*temporary, maxSteps, ordinal).run(report);
    if (failed(evaluated)) {
      signalPassFailure();
      return;
    }
    FuncDefOp output = *evaluated;
    auto target = llvm::cast<StructDefOp>(SymbolTable::lookupSymbolIn(
        getOperation(), output->getParentOfType<StructDefOp>().getSymName()
    ));
    auto source = target.getConstrainFuncOp();
    output.setSymName("constrain");
    auto origin = target->getAttrOfType<SymbolRefAttr>("poly.origin");
    if (origin) {
      SmallVector<FlatSymbolRefAttr> nested(origin.getNestedReferences());
      nested.push_back(FlatSymbolRefAttr::get(&getContext(), "constrain"));
      output->setAttr("poly.source", SymbolRefAttr::get(origin.getRootReference(), nested));
    }
    // llzk.pub grants generated cross-struct reads access; it no longer describes
    // circuit visibility on nested members. Export and witness consumers must use
    // isOriginallyPublic(), including if public aggregate outputs become legal.
    getOperation().walk([this](MemberDefOp member) {
      member->setAttr(
          polymorphic::ORIGINAL_PUBLIC_ATTR_NAME,
          BoolAttr::get(&getContext(), member.hasPublicAttr())
      );
      if (!member->getParentOfType<StructDefOp>().isMainComponent()) {
        member->setAttr("llzk.pub", UnitAttr::get(&getContext()));
      }
    });
    output->walk([](Operation *op) { op->removeAttr("poly.evaluation_source_operation"); });
    output->moveBefore(source);
    source.erase();
    getOperation()->setAttr(polymorphic::EVALUATED_MAIN_ATTR_NAME, target.getFullyQualifiedName());
  }
};
} // namespace
