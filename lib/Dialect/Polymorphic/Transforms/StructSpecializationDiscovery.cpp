//===-- StructSpecializationDiscovery.cpp -----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "StructSpecializationDiscovery.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/AttrTypeSubElements.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/DenseSet.h>

#include <limits>
#include <optional>
#include <tuple>

using namespace mlir;
using namespace llzk;
using namespace llzk::component;
using namespace llzk::polymorphic;
using namespace llzk::polymorphic::detail;

namespace {

using Environment = DenseMap<Value, Attribute>;

/// Resolve types using template bindings and concrete affine operands from the
/// evaluation environment. Substitutes bound parameters in nested types and
/// evaluates affine expressions used as struct arguments.
class SpecializationTypeResolver {
  const StructSpecializationDiscovery::Bindings &bindings;

public:
  /// Borrow bindings so template-expression results added by the evaluator are
  /// available to later type uses. This helper never changes those bindings.
  explicit SpecializationTypeResolver(const StructSpecializationDiscovery::Bindings &parameters)
      : bindings(parameters) {}

  /// Resolve parameters only in value positions, never in struct symbol names.
  Attribute resolveAttribute(Attribute attr) const {
    if (auto bound = bindings.lookup(attr)) {
      attr = bound;
    }
    if (auto type = dyn_cast<TypeAttr>(attr)) {
      return TypeAttr::get(resolveType(type.getValue()));
    }
    return attr;
  }

  /// Resolve nested types without substituting symbol identities or mutating IR.
  Type resolveType(Type type) const {
    AttrTypeReplacer replacer;
    replacer.addReplacement([this](TypeVarType variable) -> std::optional<Type> {
      if (auto bound = dyn_cast_or_null<TypeAttr>(bindings.lookup(variable.getNameRef()))) {
        return bound.getValue();
      }
      return std::nullopt;
    });
    replacer.addReplacement(
        [this](StructType structure) -> std::optional<std::pair<Type, WalkResult>> {
      SmallVector<Attribute> args;
      if (!structure.getParams()) {
        return std::make_pair(Type(structure), WalkResult::skip());
      }
      for (Attribute arg : structure.getParams()) {
        args.push_back(resolveAttribute(arg));
      }
      return std::make_pair(
          Type(
              StructType::get(structure.getNameRef(), ArrayAttr::get(structure.getContext(), args))
          ),
          WalkResult::skip()
      );
    }
    );
    replacer.addReplacement(
        [this](array::ArrayType array) -> std::optional<std::pair<Type, WalkResult>> {
      SmallVector<Attribute> dimensions;
      for (Attribute dimension : array.getDimensionSizes()) {
        dimensions.push_back(resolveAttribute(dimension));
      }
      return std::make_pair(
          Type(array::ArrayType::get(resolveType(array.getElementType()), dimensions)),
          WalkResult::skip()
      );
    }
    );
    return replacer.replace(type);
  }

  /// Evaluate one affine argument, rejecting unknown inputs and poison results.
  Attribute resolveAffineArgument(AffineMapAttr attr, ArrayRef<Attribute> operands) const {
    if (attr.getValue().getNumInputs() != operands.size() ||
        llvm::any_of(operands, [](Attribute operand) {
      return !llvm::isa_and_present<IntegerAttr>(operand);
    })) {
      return {};
    }
    SmallVector<Attribute> results;
    bool poison = false;
    if (failed(attr.getValue().constantFold(operands, results, &poison)) || poison ||
        results.size() != 1) {
      return {};
    }
    return results.front();
  }

  /// Build a concrete struct type after resolving each direct affine argument.
  /// The caller supplies values from its array indices or call operand groups.
  template <typename ResolveArgument>
  FailureOr<StructType>
  resolveAffineStructArguments(StructType structure, ResolveArgument &&resolveArgument) const {
    SmallVector<Attribute> arguments;
    for (Attribute argument : structure.getParams()) {
      if (auto map = dyn_cast<AffineMapAttr>(argument)) {
        argument = resolveArgument(map);
        if (!argument) {
          return failure();
        }
      }
      arguments.push_back(argument);
    }
    return StructType::get(
        structure.getNameRef(), ArrayAttr::get(structure.getContext(), arguments)
    );
  }
};

/// Compute which concrete struct types one instantiation needs. For example, a
/// Parent<3> method that constructs Child<i> for i = 0, 1, 2 produces requests for
/// Child<0>, Child<1>, and Child<2>, each pointing to the original construction
/// site. The caller can then create or reuse those definitions.
///
/// Template bindings such as N = 3 apply to every method of this instantiation.
/// SSA values describe a particular execution of a method or region: the same
/// loop-body result can have a different value on each iteration. Keeping those
/// values in separate environments lets each iteration use its current inputs
/// without changing N or leaking local results into another method or branch.
/// This object owns the requests and evaluation budget for a single discover()
/// call so a later call with different template arguments starts from fresh state.
class DiscoveryRun {
  StructDefOp source;
  StructSpecializationDiscovery::Bindings bindings;
  SpecializationTypeResolver typeResolver;
  SymbolTableCollection tables;
  StructSpecializationDiscovery::Requests requests;
  DenseSet<std::tuple<Operation *, Type, ArrayAttr, Type, unsigned>> seen;
  DenseSet<std::pair<Operation *, Type>> enumeratedArrays;
  using FamilyKey = std::pair<Operation *, ArrayAttr>;
  DenseMap<FamilyKey, SmallVector<StructType>> familyCandidates;
  DenseSet<FamilyKey> knownFamilies;
  SmallVector<std::pair<function::CallOp, StructType>> rolledCalls;
  DenseSet<std::pair<Operation *, Type>> seenRolledCalls;
  uint64_t remaining;
  SmallVectorImpl<Operation *> *visitedOperations;
  StructSpecializationDiscovery::CallTargets *callTargets;

  /// Charge work before evaluating it, including empty loop bodies' terminators.
  LogicalResult consumeStep(Operation *site) {
    if (!remaining) {
      return site->emitError("struct specialization discovery step limit exceeded");
    }
    --remaining;
    return success();
  }

  /// Record nested type arguments as dependencies too, retaining input identity.
  LogicalResult recordRequest(
      Operation *site, StructType type, ArrayRef<int64_t> indices = {}, StructType familyType = {},
      StructSpecializationRequest::Kind kind = StructSpecializationRequest::Kind::Plain
  ) {
    if (type.getParams()) {
      for (Attribute argument : type.getParams()) {
        if (auto nested = dyn_cast<TypeAttr>(argument)) {
          if (hasAffineMapAttr(nested.getValue())) {
            return site->emitError("affine maps nested in struct type arguments are not supported");
          }
        }
        if (!isConcreteStructParamAttr(argument)) {
          return site->emitError("cannot resolve struct specialization argument ") << argument;
        }
        if (auto nested = dyn_cast<TypeAttr>(argument)) {
          if (failed(collectTypes(site, nested.getValue()))) {
            return failure();
          }
        }
      }
    }
    if (type == llvm::cast<StructType>(typeResolver.resolveType(source.getType()))) {
      return success();
    }
    SmallVector<Attribute> indexAttrs;
    for (int64_t index : indices) {
      indexAttrs.push_back(IntegerAttr::get(IndexType::get(source.getContext()), index));
    }
    auto requestKey = std::make_tuple(
        site, Type(type), ArrayAttr::get(source.getContext(), indexAttrs), Type(familyType),
        static_cast<unsigned>(kind)
    );
    if (!seen.insert(requestKey).second) {
      return success();
    }
    requests.push_back({site, type, llvm::to_vector(indices), familyType, kind});
    return success();
  }

  static bool hasDirectAffineArgument(StructType type) {
    return type.getParams() && llvm::any_of(type.getParams(), [](Attribute argument) {
      return isa<AffineMapAttr>(argument);
    });
  }

  FailureOr<FamilyKey> getFamilyKey(Operation *site, StructType rolled) {
    auto found = rolled.getDefinition(tables, site);
    if (failed(found)) {
      return failure();
    }
    return FamilyKey(found->get().getOperation(), rolled.getParams());
  }

  LogicalResult rememberFamily(Operation *site, StructType rolled) {
    auto key = getFamilyKey(site, rolled);
    if (failed(key)) {
      return failure();
    }
    knownFamilies.insert(*key);
    return success();
  }

  LogicalResult recordFamilyCandidate(Operation *site, StructType rolled, StructType concrete) {
    auto key = getFamilyKey(site, rolled);
    if (failed(key)) {
      return failure();
    }
    knownFamilies.insert(*key);
    auto &candidates = familyCandidates[*key];
    if (!llvm::is_contained(candidates, concrete)) {
      candidates.push_back(concrete);
    }
    return success();
  }

  /// Find struct dependencies inside an aggregate. Array indices also apply to
  /// struct fields nested in POD records; a nested array supplies new indices.
  LogicalResult
  collectTypes(Operation *site, Type type, ArrayRef<int64_t> indices = {}, bool inArray = false) {
    type = typeResolver.resolveType(type);
    if (auto structure = dyn_cast<StructType>(type)) {
      if (!hasDirectAffineArgument(structure)) {
        return recordRequest(site, structure);
      }
      if (!inArray) {
        if (isa<MemberDefOp>(site)) {
          return site->emitError("affine struct type requires an enclosing array");
        }
        return success();
      }
      SmallVector<Attribute> operands;
      for (int64_t index : indices) {
        operands.push_back(IntegerAttr::get(IndexType::get(source.getContext()), index));
      }
      auto concrete = typeResolver.resolveAffineStructArguments(
          structure, [this, site, &operands](AffineMapAttr map) -> Attribute {
        if (map.getValue().getNumInputs() != operands.size()) {
          site->emitError("affine struct argument requires one input per array dimension");
          return {};
        }
        Attribute argument = typeResolver.resolveAffineArgument(map, operands);
        if (!argument) {
          site->emitError("cannot resolve affine array element specialization argument");
        }
        return argument;
      }
      );
      if (failed(concrete)) {
        return failure();
      }
      if (failed(recordFamilyCandidate(site, structure, *concrete))) {
        return failure();
      }
      return recordRequest(
          site, *concrete, indices, structure, StructSpecializationRequest::Kind::ArrayElement
      );
    }
    if (auto array = dyn_cast<array::ArrayType>(type)) {
      if (!hasAffineMapAttr(array.getElementType())) {
        return collectTypes(site, array.getElementType());
      }
      if (!enumeratedArrays.insert({site, type}).second) {
        return success();
      }
      SmallVector<int64_t> shape;
      for (Attribute dimension : array.getDimensionSizes()) {
        auto integer = dyn_cast<IntegerAttr>(dimension);
        if (!integer || integer.getInt() < 0) {
          return site->emitError("affine family discovery requires concrete array dimensions");
        }
        shape.push_back(integer.getInt());
      }
      if (llvm::is_contained(shape, 0)) {
        bool invalid = false;
        array.getElementType().walk([this, site, &invalid](Type nested) {
          if (auto structure = dyn_cast<StructType>(nested);
              structure && hasDirectAffineArgument(structure)) {
            invalid |= failed(rememberFamily(site, structure));
          }
        });
        return failure(invalid);
      }
      SmallVector<int64_t> elementIndices;
      auto enumerate = [this, site, array, &shape,
                        &elementIndices](auto &&self, unsigned dimension) -> LogicalResult {
        if (dimension < shape.size()) {
          for (int64_t i = 0; i < shape[dimension]; ++i) {
            elementIndices.push_back(i);
            if (failed(self(self, dimension + 1))) {
              return failure();
            }
            elementIndices.pop_back();
          }
          return success();
        }
        if (failed(consumeStep(site))) {
          return failure();
        }
        return collectTypes(site, array.getElementType(), elementIndices, true);
      };
      return enumerate(enumerate, 0);
    }
    if (auto record = dyn_cast<pod::PodType>(type)) {
      for (auto field : record.getRecords()) {
        if (failed(collectTypes(site, field.getType(), indices, inArray))) {
          return failure();
        }
      }
      return success();
    }
    bool invalid = false;
    type.walk([this, site, indices, inArray, &invalid](Type nested) {
      if (auto structure = dyn_cast<StructType>(nested)) {
        invalid |= failed(collectTypes(site, structure, indices, inArray));
      }
    });
    return failure(invalid);
  }

  /// Calls supply separate SSA operand groups for affine type arguments.
  LogicalResult discoverCallTypes(function::CallOp call, const Environment &env) {
    unsigned group = 0;
    for (Type result : call.getResultTypes()) {
      auto structure = dyn_cast<StructType>(typeResolver.resolveType(result));
      if (!structure || !structure.getParams()) {
        if (failed(collectTypes(call, result))) {
          return failure();
        }
        continue;
      }
      auto concrete = typeResolver.resolveAffineStructArguments(
          structure, [this, &call, &env, &group](AffineMapAttr map) -> Attribute {
        if (group >= call.getMapOperands().size()) {
          call.emitError("missing affine specialization operands");
          return {};
        }
        SmallVector<Attribute> operands;
        for (Value operand : call.getMapOperands()[group++]) {
          operands.push_back(env.lookup(operand));
        }
        Attribute argument = typeResolver.resolveAffineArgument(map, operands);
        if (!argument) {
          call.emitError("cannot resolve affine struct specialization argument");
        }
        return argument;
      }
      );
      if (failed(concrete)) {
        return failure();
      }
      bool rolledResult = hasDirectAffineArgument(structure);
      if (rolledResult) {
        if (failed(recordFamilyCandidate(call, structure, *concrete))) {
          return failure();
        }
      }
      auto kind = rolledResult ? StructSpecializationRequest::Kind::RolledCallResult
                               : StructSpecializationRequest::Kind::Plain;
      if (failed(recordRequest(call, *concrete, {}, {}, kind))) {
        return failure();
      }
    }
    for (Type argument : call.getArgOperands().getTypes()) {
      if (auto structure = dyn_cast<StructType>(typeResolver.resolveType(argument));
          structure && hasDirectAffineArgument(structure)) {
        if (seenRolledCalls.insert({call, structure}).second) {
          rolledCalls.emplace_back(call, structure);
        }
      }
    }
    return success();
  }

  /// Decode a known signed control-flow value without narrowing it.
  static std::optional<int64_t> getKnownInteger(Attribute attr) {
    auto value = dyn_cast_or_null<IntegerAttr>(attr);
    if (!value || value.getValue().getSignificantBits() > 64) {
      return std::nullopt;
    }
    return value.getInt();
  }

  /// Bind every result, including unknown values, to avoid stale loop values.
  static void bindResults(ValueRange results, ArrayRef<Attribute> values, Environment &env) {
    for (auto [result, value] : llvm::zip_equal(results, values)) {
      env[result] = value;
    }
  }

  /// Execute only a known branch. With an unknown condition, collect requests
  /// from both paths and retain a result constant only if the paths agree.
  LogicalResult
  evaluateIf(scf::IfOp branch, const Environment &env, SmallVector<Attribute> &results) {
    auto condition = getKnownInteger(env.lookup(branch.getCondition()));
    SmallVector<Attribute> left, right;
    if (!condition || *condition) {
      if (failed(evaluateBlock(branch.getThenRegion().front(), env, left))) {
        return failure();
      }
    }
    if ((!condition || !*condition) && !branch.getElseRegion().empty()) {
      if (failed(evaluateBlock(branch.getElseRegion().front(), env, right))) {
        return failure();
      }
    }
    if (condition) {
      results = *condition ? left : right;
    } else {
      for (unsigned i = 0; i < results.size(); ++i) {
        results[i] = left[i] == right[i] ? left[i] : Attribute();
      }
    }
    return success();
  }

  /// Carry each iteration's yielded constants into the next one. The body gets
  /// its own SSA environment so an unknown result cannot reuse a prior value.
  LogicalResult
  evaluateFor(scf::ForOp loop, const Environment &env, SmallVector<Attribute> &results) {
    auto lower = getKnownInteger(env.lookup(loop.getLowerBound()));
    auto upper = getKnownInteger(env.lookup(loop.getUpperBound()));
    auto stride = getKnownInteger(env.lookup(loop.getStep()));
    if (!lower || !upper || !stride || *stride <= 0 || loop.getUnsignedCmp()) {
      return loop.emitError(
          "struct discovery requires known signed for bounds and a positive step"
      );
    }
    results.clear();
    for (Value initial : loop.getInitArgs()) {
      results.push_back(env.lookup(initial));
    }
    for (int64_t i = *lower; i < *upper;) {
      auto local = env;
      local[loop.getInductionVar()] = IntegerAttr::get(loop.getInductionVar().getType(), i);
      bindResults(loop.getRegionIterArgs(), results, local);
      results.clear();
      if (failed(evaluateBlock(*loop.getBody(), std::move(local), results))) {
        return failure();
      }
      if (i > std::numeric_limits<int64_t>::max() - *stride) {
        return loop.emitError("struct discovery loop induction overflow");
      }
      i += *stride;
    }
    return success();
  }

  /// Recheck the condition after each body yield and carry the condition's
  /// forwarded values either into the body or out as the loop results.
  LogicalResult
  evaluateWhile(scf::WhileOp loop, const Environment &env, SmallVector<Attribute> &results) {
    SmallVector<Attribute> carried;
    for (Value initial : loop.getInits()) {
      carried.push_back(env.lookup(initial));
    }
    while (true) {
      auto before = env;
      bindResults(loop.getBefore().front().getArguments(), carried, before);
      SmallVector<Attribute> forwarded;
      Attribute condition;
      if (failed(
              evaluateBlock(loop.getBefore().front(), std::move(before), forwarded, &condition)
          )) {
        return failure();
      }
      auto known = dyn_cast_or_null<IntegerAttr>(condition);
      if (!known || !known.getType().isInteger(1)) {
        return loop.emitError("struct discovery requires a known while condition");
      }
      if (known.getValue().isZero()) {
        results = std::move(forwarded);
        return success();
      }
      auto after = env;
      bindResults(loop.getAfter().front().getArguments(), forwarded, after);
      carried.clear();
      if (failed(evaluateBlock(loop.getAfter().front(), std::move(after), carried))) {
        return failure();
      }
    }
  }

  /// Materialize a template binding in the scalar type requested by read_const.
  /// An absent binding stays unknown rather than supplying a default value.
  Attribute evaluateConstRead(ConstReadOp read) {
    Attribute value = bindings.lookup(read.getConstNameAttr());
    if (auto number = dyn_cast_or_null<IntegerAttr>(value)) {
      if (auto type = dyn_cast<felt::FeltType>(read.getType())) {
        value = felt::FeltConstAttr::get(read.getContext(), number.getValue(), type);
      } else if (read.getType().isIndex() || isa<IntegerType>(read.getType())) {
        unsigned width = read.getType().isIndex() ? 64 : read.getType().getIntOrFloatBitWidth();
        auto bits = width == 1 ? APInt(1, !number.getValue().isZero())
                               : number.getValue().sextOrTrunc(width);
        value = IntegerAttr::get(read.getType(), bits);
      }
    }
    return value;
  }

  /// Use existing arithmetic semantics when every operand is known. Unfoldable
  /// results stay unknown; a fold returning an SSA operand reuses its known value.
  void evaluateFoldableOp(Operation &op, const Environment &env, SmallVector<Attribute> &results) {
    SmallVector<Attribute> operands;
    for (Value operand : op.getOperands()) {
      operands.push_back(env.lookup(operand));
    }
    if (llvm::all_of(operands, [](Attribute attr) { return bool(attr); })) {
      SmallVector<OpFoldResult> folded;
      if (succeeded(op.fold(operands, folded)) && folded.size() == results.size()) {
        for (auto [index, value] : llvm::enumerate(folded)) {
          results[index] = dyn_cast<Attribute>(value);
          if (!results[index]) {
            results[index] = env.lookup(llvm::cast<Value>(value));
          }
        }
      }
    }
  }

  /// Dispatch evaluation and collect type uses at the operation being visited.
  /// Region handlers collect uses only along the paths they actually explore.
  LogicalResult
  evaluateOperation(Operation &op, const Environment &env, SmallVector<Attribute> &results) {
    if (visitedOperations) {
      visitedOperations->push_back(&op);
    }
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      return evaluateIf(branch, env, results);
    }
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      return evaluateFor(loop, env, results);
    }
    if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      return evaluateWhile(loop, env, results);
    }
    if (op.getNumRegions()) {
      return op.emitError("unsupported control flow in struct specialization discovery");
    }
    // The environment tracks scalar constants produced by template reads and
    // folding. Resolving global reads requires looking up global initializers;
    // resolving POD, array and struct-member reads requires tracking stored values
    // and writes. Those reads currently produce unknown values, including constant
    // global reads, which have no fold hook. Type uses are still collected below.
    // Unknown values are diagnosed when needed for a loop bound or specialization
    // argument; unknown if conditions cause both branches to be explored.
    if (auto read = dyn_cast<ConstReadOp>(op)) {
      results[0] = evaluateConstRead(read);
    } else if (isMemoryEffectFree(&op)) {
      evaluateFoldableOp(op, env, results);
    }
    if (auto call = dyn_cast<function::CallOp>(op)) {
      return discoverCallTypes(call, env);
    }
    for (Type type : op.getResultTypes()) {
      if (failed(collectTypes(&op, type))) {
        return failure();
      }
    }
    return success();
  }

  /// Sequence evaluation in a private SSA scope, charging the shared budget and
  /// binding each operation's results before evaluating its users. Terminators
  /// return values to the enclosing branch, loop, method or template expression.
  LogicalResult evaluateBlock(
      Block &body, Environment env, SmallVectorImpl<Attribute> &yielded,
      Attribute *condition = nullptr
  ) {
    for (Operation &op : body) {
      if (failed(consumeStep(&op))) {
        return failure();
      }
      if (auto terminator = dyn_cast<scf::ConditionOp>(op)) {
        if (!condition) {
          return op.emitError("scf.condition outside a while condition region");
        }
        *condition = env.lookup(terminator.getCondition());
        for (Value operand : terminator.getArgs()) {
          yielded.push_back(env.lookup(operand));
        }
        return success();
      }
      if (isa<YieldOp, scf::YieldOp, function::ReturnOp>(op)) {
        for (Value operand : op.getOperands()) {
          yielded.push_back(env.lookup(operand));
        }
        return success();
      }
      if (op.hasTrait<OpTrait::IsTerminator>()) {
        return op.emitError("unsupported control flow in struct specialization discovery");
      }
      SmallVector<Attribute> results(op.getNumResults());
      if (failed(evaluateOperation(op, env, results))) {
        return failure();
      }
      bindResults(op.getResults(), results, env);
    }
    return success();
  }

public:
  DiscoveryRun(
      StructDefOp structure, const StructSpecializationDiscovery::Bindings &parameters,
      uint64_t limit, SmallVectorImpl<Operation *> *visited,
      StructSpecializationDiscovery::CallTargets *targets = nullptr
  )
      : source(structure), bindings(parameters), typeResolver(bindings), remaining(limit),
        visitedOperations(visited), callTargets(targets) {}

  /// Extend the parameter environment with known template-expression results.
  FailureOr<StructSpecializationDiscovery::Bindings> evaluateBindings() {
    if (auto templ = source->getParentOfType<TemplateOp>()) {
      for (auto expr : templ.getConstOps<TemplateExprOp>()) {
        SmallVector<Attribute> values;
        if (failed(evaluateBlock(expr.getInitializerRegion().front(), Environment(), values))) {
          return failure();
        }
        if (values.size() == 1 && values[0]) {
          bindings[FlatSymbolRefAttr::get(expr.getSymNameAttr())] = values[0];
        }
      }
    }
    return bindings;
  }

  /// Evaluate template expressions, then discover members and each method.
  FailureOr<StructSpecializationDiscovery::Requests> run() {
    if (failed(evaluateBindings())) {
      return failure();
    }
    for (auto member : source.getOps<MemberDefOp>()) {
      if (failed(collectTypes(member, member.getType()))) {
        return failure();
      }
    }
    for (auto function : source.getOps<function::FuncDefOp>()) {
      if (!function.getBody().hasOneBlock()) {
        function.emitError(
            "struct discovery requires a defined single-block method; unstructured control flow "
            "is unsupported"
        );
        return failure();
      }
      for (Type type : function.getFunctionType().getInputs()) {
        if (failed(collectTypes(function, type))) {
          return failure();
        }
      }
      for (Type type : function.getFunctionType().getResults()) {
        if (failed(collectTypes(function, type))) {
          return failure();
        }
      }
      SmallVector<Attribute> values;
      if (failed(evaluateBlock(function.getBody().front(), Environment(), values))) {
        return failure();
      }
    }
    for (auto [call, family] : rolledCalls) {
      auto key = getFamilyKey(call, family);
      if (failed(key)) {
        return failure();
      }
      if (!knownFamilies.contains(*key)) {
        call.emitError("no known source for affine struct call argument ") << family;
        return failure();
      }
      auto candidates = familyCandidates.lookup(*key);
      for (StructType candidate : candidates) {
        if (failed(recordRequest(call, candidate))) {
          return failure();
        }
      }
      if (callTargets) {
        callTargets->push_back({call, candidates});
      }
    }
    return std::move(requests);
  }
};

} // namespace

FailureOr<StructSpecializationDiscovery::Bindings> StructSpecializationDiscovery::evaluateBindings(
    StructDefOp source, const Bindings &bindings
) const {
  return DiscoveryRun(source, bindings, limit, nullptr).evaluateBindings();
}

FailureOr<StructSpecializationDiscovery::Requests> StructSpecializationDiscovery::discover(
    StructDefOp source, const Bindings &bindings, SmallVectorImpl<Operation *> *visitedOperations,
    CallTargets *callTargets
) const {
  return DiscoveryRun(source, bindings, limit, visitedOperations, callTargets).run();
}
