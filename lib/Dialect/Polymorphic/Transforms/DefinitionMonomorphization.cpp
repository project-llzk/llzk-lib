//===-- DefinitionMonomorphization.cpp --------------------------*- C++ -*-===//
// Part of the LLZK Project, under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/AttrTypeSubElements.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>

namespace llzk::polymorphic {
#define GEN_PASS_DEF_DEFINITIONMONOMORPHIZATIONPASS
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h.inc"
} // namespace llzk::polymorphic

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::polymorphic;

namespace {

/// Attempt to evaluate the concrete result of a single `TemplateExprOp` expression given
/// the currently-known concrete param values in `paramNameToConcrete`. Returns the result
/// attribute if all referenced params are concrete and all operations in the body can be
/// constant-folded; otherwise returns `std::nullopt`.
static std::optional<Attribute>
evaluateExpr(TemplateExprOp exprOp, const DenseMap<Attribute, Attribute> &paramNameToConcrete) {
  // Map from SSA value in the expr body to its concrete Attribute.
  DenseMap<Value, Attribute> valueMap;
  for (Operation &bodyOp : exprOp.getInitializerRegion().front()) {
    if (auto yieldOp = llvm::dyn_cast<YieldOp>(bodyOp)) {
      auto it = valueMap.find(yieldOp.getVal());
      return it != valueMap.end() ? std::make_optional(it->second) : std::nullopt;
    }

    if (auto constReadOp = llvm::dyn_cast<ConstReadOp>(bodyOp)) {
      auto it = paramNameToConcrete.find(constReadOp.getConstNameAttr());
      if (it == paramNameToConcrete.end()) {
        return std::nullopt; // a referenced param is not concrete
      }
      // If the attribute type is `FeltType` but it's stored as an IntegerAttr, promote to
      // a `FeltConstAttr`.
      Attribute val = it->second;
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(val)) {
        if (auto feltTy = llvm::dyn_cast<FeltType>(constReadOp.getResult().getType())) {
          val = FeltConstAttr::get(bodyOp.getContext(), intAttr.getValue(), feltTy);
        }
      }
      valueMap[constReadOp.getResult()] = val;
      continue;
    }

    // Gather constant attributes for all operands.
    SmallVector<Attribute> operandAttrs;
    operandAttrs.reserve(bodyOp.getNumOperands());
    for (Value operand : bodyOp.getOperands()) {
      auto it = valueMap.find(operand);
      if (it == valueMap.end()) {
        return std::nullopt; // operand not known as a constant
      }
      operandAttrs.push_back(it->second);
    }

    // These casts canonicalize constants but do not implement the fold hook.
    if (auto castOp = dyn_cast<llzk::cast::IntToFeltOp>(bodyOp)) {
      auto integer = dyn_cast<IntegerAttr>(operandAttrs.front());
      if (!integer) {
        return std::nullopt;
      }
      valueMap[castOp.getResult()] =
          FeltConstAttr::get(bodyOp.getContext(), integer.getValue(), castOp.getType());
      continue;
    }
    if (auto castOp = dyn_cast<llzk::cast::FeltToIndexOp>(bodyOp)) {
      auto felt = dyn_cast<FeltConstAttr>(operandAttrs.front());
      if (!felt || felt.getValue().isNegative() || felt.getValue().getActiveBits() > 63) {
        return std::nullopt;
      }
      valueMap[castOp.getResult()] =
          IntegerAttr::get(castOp.getType(), felt.getValue().getZExtValue());
      continue;
    }
    if (isa<llzk::boolean::AssertOp>(bodyOp)) {
      auto condition = dyn_cast<IntegerAttr>(operandAttrs.front());
      if (!condition || condition.getValue().isZero()) {
        return std::nullopt;
      }
      continue;
    }

    // Try constant folding.
    SmallVector<OpFoldResult> foldResults;
    // Folding may mutate an operation; never fold the source template itself.
    auto *copy = bodyOp.clone();
    auto folded = copy->fold(operandAttrs, foldResults);
    copy->destroy();
    if (succeeded(folded) && foldResults.size() == bodyOp.getNumResults()) {
      for (auto [result, fr] : llvm::zip_equal(bodyOp.getResults(), foldResults)) {
        if (Attribute a = llvm::dyn_cast<Attribute>(fr)) {
          valueMap[result] = a;
        } else if (auto value = valueMap.lookup(llvm::cast<Value>(fr))) {
          valueMap[result] = value;
        } else {
          return std::nullopt;
        }
      }
    } else {
      return std::nullopt;
    }
  }
  return std::nullopt; // no YieldOp found (shouldn't happen in a valid expr)
}

/// Evaluate all `TemplateExprOp`s in `templateOp` that can be computed from the currently-known
/// concrete param values in `paramNameToConcrete`, and add their results to the map.
/// Exprs whose operands are not all concrete are silently skipped (partial instantiation).
static void evaluateTemplateExprs(TemplateOp templ, DenseMap<Attribute, Attribute> &bindings) {
  for (auto expr : templ.getConstOps<TemplateExprOp>()) {
    if (auto value = evaluateExpr(expr, bindings)) {
      bindings.try_emplace(FlatSymbolRefAttr::get(expr.getSymNameAttr()), *value);
    }
  }
}

/// Substitute template values throughout a definition, including nested POD records.
/// Only type-bearing attributes and explicit call arguments are rewritten as values;
/// symbol identities such as member names remain unchanged.
static LogicalResult substituteDefinition(
    Operation *definition, const DenseMap<Attribute, Attribute> &bindings, StructType oldSelf = {},
    StructType newSelf = {}
) {
  bool invalid = false;
  AttrTypeReplacer values;
  values.addReplacement([&](TypeVarType type) -> std::optional<Type> {
    if (auto value = dyn_cast_or_null<TypeAttr>(bindings.lookup(type.getNameRef()))) {
      return value.getValue();
    }
    return std::nullopt;
  });
  values.addReplacement([&](StructType type) -> std::optional<Type> {
    if (oldSelf && type == oldSelf) {
      return newSelf;
    }
    return std::nullopt;
  });
  values.addReplacement([&](ArrayType type) -> std::optional<std::pair<Type, WalkResult>> {
    SmallVector<Attribute> dimensions;
    for (Attribute dim : type.getDimensionSizes()) {
      if (Attribute value = bindings.lookup(dim)) {
        dim = value;
      }
      if (auto felt = dyn_cast<FeltConstAttr>(dim)) {
        auto integer = felt.getValue();
        if (integer.isNegative() || integer.getActiveBits() > 63) {
          definition->emitError("specialized array dimension does not fit a nonnegative index");
          invalid = true;
          return std::make_pair(Type(type), WalkResult::skip());
        }
        dim = IntegerAttr::get(IndexType::get(type.getContext()), integer.getZExtValue());
      }
      dimensions.push_back(dim);
    }
    return std::make_pair(
        Type(ArrayType::get(values.replace(type.getElementType()), dimensions)), WalkResult::skip()
    );
  });
  values.addReplacement([&](SymbolRefAttr name) -> std::optional<Attribute> {
    if (auto value = bindings.lookup(name)) {
      return value;
    }
    return std::nullopt;
  });

  // Resolve reads before rewriting types, retaining each read's requested scalar type.
  SmallVector<ConstReadOp> reads;
  definition->walk([&](ConstReadOp read) { reads.push_back(read); });
  for (auto read : reads) {
    Attribute value = bindings.lookup(read.getConstNameAttr());
    if (!value) {
      return read.emitError("cannot evaluate specialization constant ") << read.getConstNameAttr();
    }
    OpBuilder builder(read);
    Value replacement;
    auto integer = dyn_cast<IntegerAttr>(value);
    auto felt = dyn_cast<FeltConstAttr>(value);
    if (!integer && !felt) {
      return read.emitError("unsupported specialization constant ") << value;
    }
    APInt bits = integer ? integer.getValue() : felt.getValue();
    if (auto type = dyn_cast<FeltType>(read.getType())) {
      replacement = FeltConstantOp::create(
          builder, read.getLoc(), FeltConstAttr::get(read.getContext(), bits, type)
      );
    } else if (read.getType().isIndex()) {
      if (felt && (bits.isNegative() || bits.getActiveBits() > 63)) {
        return read.emitError("specialization constant does not fit a nonnegative index");
      }
      replacement = arith::ConstantOp::create(
          builder, read.getLoc(), builder.getIndexAttr(bits.getSExtValue())
      );
    } else if (auto integerType = dyn_cast<IntegerType>(read.getType())) {
      // Circom/LLZK boolean reads interpret every nonzero scalar as true.
      auto scalar = integerType.getWidth() == 1 ? APInt(1, !bits.isZero())
                                                : bits.zextOrTrunc(integerType.getWidth());
      replacement =
          arith::ConstantOp::create(builder, read.getLoc(), IntegerAttr::get(integerType, scalar));
    } else {
      return read.emitError("unsupported specialization constant type");
    }
    read.replaceAllUsesWith(replacement);
    read.erase();
  }
  AttrTypeReplacer types;
  types.addReplacement([&](Type type) -> std::optional<Type> { return values.replace(type); });
  types.recursivelyReplaceElementsIn(definition, true, false, true);
  definition->walk([&](CallOp call) {
    if (auto args = call.getTemplateParamsAttr()) {
      call.setTemplateParamsAttr(llvm::cast<ArrayAttr>(values.replace(args)));
    }
    auto callee = call.getCalleeAttr();
    auto binding = dyn_cast_or_null<TypeAttr>(
        bindings.lookup(FlatSymbolRefAttr::get(callee.getRootReference()))
    );
    if (binding) {
      if (auto type = dyn_cast<StructType>(values.replace(binding.getValue()))) {
        auto pieces = getPieces(type.getNameRef());
        llvm::append_range(pieces, callee.getNestedReferences());
        call.setCalleeAttr(asSymbolRefAttr(pieces));
      }
    }
  });
  return failure(invalid);
}

/// Definition-only worklist. This deliberately does not invoke the flattening driver,
/// greedy folding, inlining, or loop transformations. Substitution recursively
/// updates types while preserving all source control-flow structure.
class DefinitionRegistry {
  struct Entry {
    Operation *source;
    Operation *definition;
    ArrayAttr arguments;
    ArrayAttr constantArguments;
    std::optional<unsigned> parent;
  };
  ModuleOp root;
  unsigned limit;
  std::unique_ptr<SymbolTableCollection> tables = std::make_unique<SymbolTableCollection>();
  DenseMap<std::pair<std::pair<Operation *, ArrayAttr>, ArrayAttr>, unsigned> cache;
  SmallVector<Entry> entries;
  std::optional<unsigned> current;
  bool invalid = false;
  uint64_t familyIterations = 0;
  DenseMap<Operation *, SmallVector<Operation *>> callEdges;

  /// Affine parameters denote rolled families, not concrete specialization keys.
  static bool isFamily(Type type) {
    bool affine = false;
    type.walk([&](Attribute a) { affine |= isa<AffineMapAttr>(a); }, [](Type) {});
    return affine;
  }

  /// Enumerate a bounded member family by its array indices, retaining the rolled
  /// array type. Metadata maps index tuples to IDs for later instance elaboration.
  LogicalResult discoverMemberFamily(MemberDefOp member) {
    Type type = member.getType();
    if (!isFamily(type)) {
      return success();
    }
    auto array = dyn_cast<ArrayType>(type);
    if (!array || !isa<StructType>(array.getElementType())) {
      return member.emitError("unsupported affine member family: expected an array of structs");
    }
    SmallVector<int64_t> shape;
    for (Attribute size : array.getDimensionSizes()) {
      auto integer = dyn_cast<IntegerAttr>(size);
      if (!integer || integer.getInt() < 0) {
        return member.emitError("unsupported affine member family: expected concrete dimensions");
      }
      shape.push_back(integer.getInt());
    }
    Builder builder(root.getContext());
    SmallVector<Attribute> indices, records;
    std::function<LogicalResult(unsigned)> enumerate = [&](unsigned dimension) -> LogicalResult {
      if (dimension < shape.size()) {
        for (int64_t i = 0; i < shape[dimension]; ++i) {
          indices.push_back(builder.getIndexAttr(i));
          if (failed(enumerate(dimension + 1))) {
            return failure();
          }
          indices.pop_back();
        }
        return success();
      }
      if (++familyIterations > 1000000) {
        return member.emitError("affine specialization discovery iteration limit exceeded");
      }
      bool failedMap = false;
      AttrTypeReplacer evaluate;
      evaluate.addReplacement([&](AffineMapAttr attr) -> std::optional<Attribute> {
        auto map = attr.getAffineMap();
        SmallVector<Attribute> folded;
        bool poison = false;
        if (map.getNumInputs() != indices.size() ||
            failed(map.constantFold(indices, folded, &poison)) || poison || folded.size() != 1) {
          failedMap = true;
          return attr;
        }
        return folded.front();
      });
      auto concrete = llvm::cast<StructType>(evaluate.replace(array.getElementType()));
      if (failedMap || isFamily(concrete)) {
        return member.emitError(
            "unsupported affine member family map: expected one result and one input per array "
            "dimension"
        );
      }
      auto specialized = specialize(concrete);
      if (failed(specialized)) {
        return failure();
      }
      auto definition = specialized->getDefinition(*tables, root);
      if (failed(definition)) {
        return failure();
      }
      records.push_back(builder.getDictionaryAttr(
          {builder.getNamedAttr("indices", builder.getArrayAttr(indices)),
           builder.getNamedAttr(
               "specialization", definition->get()->getAttr("poly.specialization_id")
           )}
      ));
      return success();
    };
    if (failed(enumerate(0))) {
      return failure();
    }
    member->setAttr("poly.family", builder.getArrayAttr(records));
    return success();
  }

  /// Explore static induction values without cloning or changing loop bodies.
  LogicalResult discoverFamilies(Operation *op, DenseMap<Value, Attribute> &values) {
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      auto integer = [&](Value v) -> std::optional<int64_t> {
        if (auto a = dyn_cast_or_null<IntegerAttr>(values.lookup(v))) {
          return a.getInt();
        }
        return getConstantIntValue(v);
      };
      auto lo = integer(loop.getLowerBound()), hi = integer(loop.getUpperBound());
      auto step = integer(loop.getStep());
      bool needsEnumeration = false;
      loop.walk([&](CallOp c) {
        for (Type t : c.getResultTypes()) {
          needsEnumeration |= isFamily(t);
        }
      });
      if (!needsEnumeration) {
        return success();
      }
      if (!lo || !hi || !step || *step <= 0 || loop.getNumRegionIterArgs()) {
        return loop.emitError(
            "unsupported affine specialization loop: expected static bounds, positive step, no "
            "iter_args"
        );
      }
      for (int64_t i = *lo; i < *hi;) {
        if (++familyIterations > 1000000) {
          return loop.emitError("affine specialization discovery iteration limit exceeded");
        }
        auto local = values;
        local[loop.getInductionVar()] = IntegerAttr::get(IndexType::get(root.getContext()), i);
        for (Operation &child : *loop.getBody()) {
          if (failed(discoverFamilies(&child, local))) {
            return failure();
          }
        }
        if (i > INT64_MAX - *step) {
          break;
        }
        i += *step;
      }
      return success();
    }
    if (auto call = dyn_cast<CallOp>(op)) {
      auto type = getIfSingleton<StructType>(call.getResultTypes());
      if (type && isFamily(type)) {
        SmallVector<Attribute> params;
        unsigned group = 0;
        for (Attribute param : type.getParams()) {
          if (auto map = dyn_cast<AffineMapAttr>(param)) {
            if (group >= call.getMapOperands().size()) {
              return call.emitError("missing operands for affine specialization");
            }
            SmallVector<Attribute> operands, folded;
            for (Value v : call.getMapOperands()[group++]) {
              Attribute a = values.lookup(v);
              if (!a) {
                if (auto n = getConstantIntValue(v)) {
                  a = IntegerAttr::get(IndexType::get(root.getContext()), *n);
                }
              }
              if (!a) {
                return call.emitError("unsupported witness-dependent affine specialization");
              }
              operands.push_back(a);
            }
            bool poison = false;
            if (failed(map.getAffineMap().constantFold(operands, folded, &poison)) || poison ||
                folded.size() != 1) {
              return call.emitError("cannot evaluate affine specialization parameter");
            }
            params.push_back(folded.front());
          } else {
            params.push_back(param);
          }
        }
        auto instantiated =
            StructType::get(type.getNameRef(), ArrayAttr::get(root.getContext(), params));
        if (isFamily(instantiated)) {
          return call.emitError("unsupported nested affine type-argument specialization");
        }
        auto concrete = specialize(instantiated);
        if (failed(concrete)) {
          return failure();
        }
        auto target = concrete->getDefinition(*tables, root);
        if (failed(target)) {
          return failure();
        }
        auto id = target->get()->getAttr("poly.specialization_id");
        SmallVector<Attribute> ids;
        if (auto previous = call->getAttrOfType<ArrayAttr>("poly.family_specializations")) {
          llvm::append_range(ids, previous);
        }
        if (!llvm::is_contained(ids, id)) {
          ids.push_back(id);
        }
        call->setAttr("poly.family_specializations", ArrayAttr::get(root.getContext(), ids));
      }
    }
    SmallVector<Attribute> operands;
    for (Value v : op->getOperands()) {
      operands.push_back(values.lookup(v));
    }
    // Fold a temporary operation: discovery must never mutate source control flow.
    if (op->getNumRegions() == 0 && op->getNumResults() &&
        llvm::all_of(operands, [](Attribute a) { return bool(a); })) {
      Operation *copy = op->clone();
      SmallVector<OpFoldResult> results;
      if (succeeded(copy->fold(operands, results)) && results.size() == op->getNumResults()) {
        for (auto [v, r] : llvm::zip(op->getResults(), results)) {
          if (auto a = dyn_cast<Attribute>(r)) {
            values[v] = a;
          }
        }
      }
      copy->destroy();
    }
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        auto local = values;
        for (Operation &child : block) {
          if (failed(discoverFamilies(&child, local))) {
            return failure();
          }
        }
      }
    }
    return success();
  }

  /// Reject cycles even when every edge hits an already-populated cache entry.
  LogicalResult checkCallCycles() {
    DenseSet<Operation *> active, done;
    SmallVector<Operation *> chain;
    std::function<LogicalResult(Operation *)> visit = [&](Operation *op) {
      if (active.contains(op)) {
        auto diag = op->emitError("recursive specialization call cycle is unsupported");
        for (Operation *item : chain) {
          diag.attachNote(item->getLoc())
              << "specialization chain: "
              << getFullyQualifiedName(llvm::cast<SymbolOpInterface>(item));
        }
        return failure();
      }
      if (!done.insert(op).second) {
        return success();
      }
      active.insert(op);
      chain.push_back(op);
      for (Operation *child : callEdges.lookup(op)) {
        if (failed(visit(child))) {
          return failure();
        }
      }
      chain.pop_back();
      active.erase(op);
      return success();
    };
    SmallVector<Operation *> roots;
    for (const Entry &entry : entries) {
      entry.definition->walk([&](FuncDefOp function) { roots.push_back(function); });
    }
    for (Operation *op : roots) {
      if (failed(visit(op))) {
        return failure();
      }
    }
    return success();
  }

  ArrayAttr emptyArgs() { return ArrayAttr::get(root.getContext(), {}); }

  /// Return a scalar constant for `value`, folding only regionless producers.
  /// This deliberately leaves control flow, calls, aggregates, and unsupported
  /// conversions dynamic.
  std::optional<Attribute> getScalarConstant(
      Value value, DenseMap<Value, std::optional<Attribute>> &memo, DenseSet<Value> &visiting
  ) {
    if (auto it = memo.find(value); it != memo.end()) {
      return it->second;
    }
    if (!visiting.insert(value).second) {
      return std::nullopt;
    }
    auto finish = [&](std::optional<Attribute> result) {
      visiting.erase(value);
      memo[value] = result;
      return result;
    };
    Operation *def = value.getDefiningOp();
    if (!def) {
      return finish(std::nullopt);
    }
    if (auto constant = dyn_cast<FeltConstantOp>(def)) {
      return finish(constant.getValueAttr());
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(def)) {
      if (auto attr = dyn_cast<IntegerAttr>(constant.getValue())) {
        return finish(attr);
      }
      return finish(std::nullopt);
    }
    if (def->getNumRegions() != 0 || def->getNumResults() != 1) {
      return finish(std::nullopt);
    }
    SmallVector<Attribute> operands;
    for (Value operand : def->getOperands()) {
      auto attr = getScalarConstant(operand, memo, visiting);
      if (!attr) {
        return finish(std::nullopt);
      }
      operands.push_back(*attr);
    }
    Operation *copy = def->clone();
    SmallVector<OpFoldResult> results;
    auto folded = copy->fold(operands, results);
    copy->destroy();
    if (failed(folded) || results.size() != 1) {
      return finish(std::nullopt);
    }
    if (auto attr = dyn_cast<Attribute>(results.front())) {
      return finish(attr);
    }
    if (auto forwarded = dyn_cast<Value>(results.front())) {
      return finish(getScalarConstant(forwarded, memo, visiting));
    }
    return finish(std::nullopt);
  }

  /// Materialize scalar specializations at the clone entry and replace uses of
  /// their retained signature arguments. `UnitAttr` denotes a dynamic argument.
  LogicalResult substituteConstantArguments(FuncDefOp clone, ArrayAttr constants) {
    if (!constants || constants.empty()) {
      return success();
    }
    Block &entry = clone.getBody().front();
    OpBuilder builder(&entry, entry.begin());
    for (auto [index, attr] : llvm::enumerate(constants)) {
      if (isa<UnitAttr>(attr)) {
        continue;
      }
      Value argument = clone.getArgument(index);
      Value replacement;
      if (auto felt = dyn_cast<FeltConstAttr>(attr)) {
        if (felt.getType() != argument.getType()) {
          return clone.emitError("constant specialization type mismatch");
        }
        replacement = FeltConstantOp::create(builder, clone.getLoc(), felt);
      } else if (auto integer = dyn_cast<IntegerAttr>(attr)) {
        if (integer.getType() != argument.getType()) {
          return clone.emitError("constant specialization type mismatch");
        }
        replacement = arith::ConstantOp::create(builder, clone.getLoc(), integer);
      } else {
        return clone.emitError("unsupported constant specialization argument");
      }
      argument.replaceAllUsesWith(replacement);
    }
    return success();
  }

  /// Reserve an identity before cloning, so repeated requests share a work item.
  FailureOr<unsigned> reserve(Operation *source, ArrayAttr args, ArrayAttr constantArgs = {}) {
    if (!args) {
      args = emptyArgs();
    }
    if (!constantArgs) {
      constantArgs = emptyArgs();
    }
    auto key = std::make_pair(std::make_pair(source, args), constantArgs);
    if (auto it = cache.find(key); it != cache.end()) {
      return it->second;
    }
    for (auto ancestor = current; ancestor; ancestor = entries[*ancestor].parent) {
      if (entries[*ancestor].source == source) {
        auto diag = source->emitError("recursive specialization is unsupported");
        diag << ": requested " << args << " constants " << constantArgs;
        for (auto p = current; p; p = entries[*p].parent) {
          diag.attachNote(entries[*p].source->getLoc())
              << "specialization chain: "
              << getFullyQualifiedName(llvm::cast<SymbolOpInterface>(entries[*p].source))
              << entries[*p].arguments << " constants " << entries[*p].constantArguments;
        }
        return failure();
      }
    }
    if (entries.size() >= limit) {
      source->emitError("specialization limit exceeded");
      return failure();
    }
    unsigned id = entries.size();
    cache[key] = id;
    entries.push_back({source, nullptr, args, constantArgs, current});
    return id;
  }

  void publish(unsigned id, Operation *definition) {
    entries[id].definition = definition;
    Builder b(root.getContext());
    definition->setAttr("poly.specialization_id", b.getI64IntegerAttr(id));
    definition->setAttr(
        "poly.origin", getFullyQualifiedName(llvm::cast<SymbolOpInterface>(entries[id].source))
    );
    definition->setAttr("poly.arguments", entries[id].arguments);
    if (!entries[id].constantArguments.empty()) {
      definition->setAttr("poly.constant_arguments", entries[id].constantArguments);
    }
  }

  FailureOr<StructType> specialize(StructType type) {
    if (isFamily(type)) {
      return type;
    }
    auto found = type.getDefinition(*tables, root);
    if (failed(found)) {
      return failure();
    }
    if (found->viaInclude()) {
      root.emitError("inline includes before llzk-monomorphize");
      return failure();
    }
    StructDefOp source = found->get();
    if (source->hasAttr("poly.specialization_id")) {
      return type;
    }
    if (type.getParams() && !llvm::all_of(type.getParams(), [](Attribute a) {
      return isConcreteStructParamAttr(a, false);
    })) {
      source.emitError("unsupported non-concrete specialization arguments") << type;
      return failure();
    }
    auto id = reserve(source, type.getParams());
    if (failed(id)) {
      return failure();
    }
    if (entries[*id].definition) {
      return StructType::get(
          getFullyQualifiedName(llvm::cast<SymbolOpInterface>(entries[*id].definition))
      );
    }
    auto clone = source.clone();
    clone.setSymName(("__llzk_spec_" + Twine(*id)).str());
    auto templ = source->getParentOfType<TemplateOp>();
    Operation *parent = templ ? templ->getParentOp() : source->getParentOp();
    tables->getSymbolTable(parent).insert(clone);
    DenseMap<Attribute, Attribute> bindings;
    if (type.getParams()) {
      for (auto [name, value] : llvm::zip(source.getType().getParams(), type.getParams())) {
        bindings[name] = value;
      }
    }
    if (templ) {
      evaluateTemplateExprs(templ, bindings);
    }
    if (failed(substituteDefinition(
            clone, bindings, source.getType(), StructType::get(getFullyQualifiedName(clone))
        ))) {
      return failure();
    }
    publish(*id, clone);
    return StructType::get(getFullyQualifiedName(clone));
  }

  LogicalResult specializeCall(CallOp call) {
    auto target = call.getCalleeTarget(*tables);
    if (failed(target)) {
      return failure();
    }
    if (target->viaInclude()) {
      return call.emitError("inline includes before llzk-monomorphize");
    }
    if (target->get().isExternal()) {
      return call.emitError("external function specialization is unsupported");
    }
    bool structMethod = isa<StructDefOp>(target->get()->getParentOp());
    // Struct compute/constrain references are determined by their rewritten types.
    if (structMethod && call.getCalleeAttr().getLeafReference() == "compute") {
      if (auto type = getIfSingleton<StructType>(call.getResultTypes())) {
        call.setCalleeAttr(appendLeaf(type.getNameRef(), call.getCalleeAttr().getLeafReference()));
      }
      return success();
    }
    if (structMethod && call.getCalleeAttr().getLeafReference() == "constrain") {
      if (auto type = getAtIndex<StructType>(call.getArgOperands().getTypes(), 0)) {
        call.setCalleeAttr(appendLeaf(type.getNameRef(), call.getCalleeAttr().getLeafReference()));
      }
      return success();
    }
    FuncDefOp source = target->get();
    if (source->hasAttr("poly.specialization_id")) {
      return success();
    }
    if (isa<StructDefOp>(source->getParentOp())) {
      return call.emitError("unsupported non-compute/constrain struct method specialization");
    }
    DenseMap<Attribute, Attribute> bindings;
    SmallVector<Attribute> args;
    auto templ = dyn_cast<TemplateOp>(source->getParentOp());
    if (templ) {
      auto unified = call.unifyTypeSignature(source.getFunctionType());
      if (failed(unified)) {
        return call.emitError("cannot infer concrete function specialization");
      }
      unsigned n = 0;
      for (auto param : templ.getConstOps<TemplateParamOp>()) {
        auto name = FlatSymbolRefAttr::get(param.getSymNameAttr());
        Attribute value;
        if (auto explicitArgs = call.getTemplateParamsAttr();
            explicitArgs && !explicitArgs.empty()) {
          value = explicitArgs[n];
        } else {
          value = unified->lookup({name, Side::RHS});
        }
        ++n;
        if (!value || !isConcreteStructParamAttr(value, false)) {
          return call.emitError("unsupported non-concrete function parameter ") << name;
        }
        if (failed(call.verifyTemplateParamValueCompatibility(value, param))) {
          return failure();
        }
        args.push_back(value);
        bindings[name] = value;
      }
    }
    DenseMap<Value, std::optional<Attribute>> constantsMemo;
    DenseSet<Value> visiting;
    SmallVector<Attribute> constants;
    bool hasConstant = false;
    for (Value operand : call.getArgOperands()) {
      auto value = getScalarConstant(operand, constantsMemo, visiting);
      if (value) {
        if (auto felt = dyn_cast<FeltConstAttr>(*value);
            felt && felt.getType() == operand.getType()) {
          constants.push_back(felt);
          hasConstant = true;
          continue;
        }
        if (auto integer = dyn_cast<IntegerAttr>(*value);
            integer && integer.getType() == operand.getType() &&
            (operand.getType().isIndex() || isa<IntegerType>(operand.getType()))) {
          constants.push_back(integer);
          hasConstant = true;
          continue;
        }
      }
      constants.push_back(UnitAttr::get(root.getContext()));
    }
    ArrayAttr constantArgs =
        hasConstant ? ArrayAttr::get(root.getContext(), constants) : emptyArgs();
    auto id = reserve(source, ArrayAttr::get(root.getContext(), args), constantArgs);
    if (failed(id)) {
      return failure();
    }
    if (!entries[*id].definition) {
      OwningOpRef<FuncDefOp> ownedClone(source.clone());
      auto clone = ownedClone.get();
      clone.setSymName(("__llzk_spec_" + Twine(*id)).str());
      Operation *parent = templ ? templ->getParentOp() : source->getParentOp();
      // Body conversion resolves nested calls through the enclosing module.
      tables->getSymbolTable(parent).insert(clone);
      if (templ) {
        evaluateTemplateExprs(templ, bindings);
        if (failed(substituteDefinition(clone, bindings))) {
          tables->invalidateSymbolTable(parent);
          return failure();
        }
      }
      if (failed(substituteConstantArguments(clone, constantArgs))) {
        tables->invalidateSymbolTable(parent);
        return failure();
      }
      ownedClone.release();
      publish(*id, clone);
    }
    call.setCalleeAttr(
        getFullyQualifiedName(llvm::cast<SymbolOpInterface>(entries[*id].definition))
    );
    call.removeTemplateParamsAttr();
    return success();
  }

public:
  DefinitionRegistry(ModuleOp module, unsigned maximum) : root(module), limit(maximum) {}

  LogicalResult run() {
    auto main = getMainInstanceType(root);
    if (failed(main) || !*main) {
      return root.emitError("llzk-monomorphize requires a concrete llzk.main");
    }
    auto specialized = specialize(*main);
    if (failed(specialized)) {
      return failure();
    }
    root->setAttr(MAIN_ATTR_NAME, TypeAttr::get(*specialized));
    for (unsigned i = 0; i < entries.size(); ++i) {
      current = i;
      Operation *definition = entries[i].definition;
      AttrTypeReplacer types;
      types.addReplacement([&](StructType type) -> std::optional<Type> {
        auto result = specialize(type);
        if (failed(result)) {
          invalid = true;
          return type;
        }
        return *result;
      });
      // Provenance is immutable input metadata, not a live type reference.
      auto origin = definition->removeAttr("poly.origin");
      auto args = definition->removeAttr("poly.arguments");
      auto constantArgs = definition->removeAttr("poly.constant_arguments");
      types.recursivelyReplaceElementsIn(definition, true, false, true);
      definition->setAttr("poly.origin", origin);
      definition->setAttr("poly.arguments", args);
      if (constantArgs) {
        definition->setAttr("poly.constant_arguments", constantArgs);
      }
      if (invalid) {
        return failure();
      }
      auto schemas = definition->walk([&](MemberDefOp member) {
        return WalkResult(discoverMemberFamily(member));
      });
      if (schemas.wasInterrupted()) {
        return failure();
      }
      DenseMap<Value, Attribute> values;
      if (failed(discoverFamilies(definition, values))) {
        return failure();
      }
      auto result = definition->walk([&](CallOp call) {
        if (failed(specializeCall(call))) {
          return WalkResult::interrupt();
        }
        auto target = call.getCalleeTarget(*tables);
        if (failed(target)) {
          return WalkResult::interrupt();
        }
        auto owner = call->getParentOfType<FuncDefOp>();
        callEdges[owner].push_back(target->get());
        if (auto family = call->getAttrOfType<ArrayAttr>("poly.family_specializations")) {
          for (Attribute attr : family) {
            auto id = dyn_cast<IntegerAttr>(attr);
            if (!id || id.getInt() < 0 || static_cast<uint64_t>(id.getInt()) >= entries.size()) {
              call.emitError("invalid family specialization identity");
              return WalkResult::interrupt();
            }
            entries[id.getInt()].definition->walk([&](FuncDefOp function) {
              if (function.getSymName() == call.getCalleeAttr().getLeafReference()) {
                callEdges[owner].push_back(function);
              }
            });
          }
        }
        return WalkResult::advance();
      });
      if (result.wasInterrupted()) {
        return failure();
      }
    }
    return checkCallCycles();
  }

  unsigned size() const { return entries.size(); }
};

class MonomorphizationPassImpl
    : public llzk::polymorphic::impl::DefinitionMonomorphizationPassBase<MonomorphizationPassImpl> {
  using Base = DefinitionMonomorphizationPassBase<MonomorphizationPassImpl>;
  using Base::Base;

  void runOnOperation() override {
    auto root = getOperation();
    const auto start = std::chrono::steady_clock::now();
    uint64_t inputOperations = 0;
    root.walk([&](Operation *) { ++inputOperations; });
    DefinitionRegistry registry(root, maxSpecializations);
    if (failed(registry.run())) {
      signalPassFailure();
      return;
    }
    if (report) {
      uint64_t operations = 0;
      root.walk([&](Operation *) { ++operations; });
      uint64_t specializedOperations = 0;
      root.walk([&](Operation *op) {
        if (op->hasAttr("poly.specialization_id")) {
          op->walk([&](Operation *) { ++specializedOperations; });
        }
      });
      double milliseconds =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
              .count();
      llvm::errs() << "monomorphization: specializations=" << registry.size()
                   << " input_operations=" << inputOperations
                   << " specialized_operations=" << specializedOperations
                   << " pass_ms=" << milliseconds << " operations=" << operations << '\n';
    }
  }
};

} // namespace
