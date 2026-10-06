//===-- TemplateMonomorphization.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "SharedImpl.h"
#include "StructSpecializationDiscovery.h"
#include "TemplateInstantiation.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/IR/AttrTypeSubElements.h>
#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

#include <llvm/ADT/MapVector.h>

namespace llzk::polymorphic {
#define GEN_PASS_DEF_TEMPLATEMONOMORPHIZATIONPASS
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h.inc"
} // namespace llzk::polymorphic

using namespace mlir;
using namespace llzk;
using namespace llzk::component;
using namespace llzk::felt;
using namespace llzk::polymorphic;
using namespace llzk::polymorphic::detail;

namespace {

/// Canonicalize equivalent typed arguments while preserving integers whose
/// original value may be forwarded to another template parameter.
/// Untyped parameters retain their supplied attributes.
FailureOr<ArrayAttr>
normalizeTypedArguments(StructDefOp source, ArrayAttr arguments, Operation *site) {
  auto templ = source->getParentOfType<TemplateOp>();
  if (!templ) {
    return arguments;
  }
  SmallVector<Attribute> normalized;
  for (auto [parameter, argument] :
       llvm::zip_equal(templ.getConstOps<TemplateParamOp>(), arguments)) {
    Attribute value = argument;
    if (std::optional<Type> expected = parameter.getTypeOpt()) {
      if (auto indexType = dyn_cast<IndexType>(*expected)) {
        if (auto felt = dyn_cast<FeltConstAttr>(argument)) {
          const APInt &number = felt.getValue();
          if (number.getActiveBits() <= 63) {
            value = IntegerAttr::get(indexType, number.getZExtValue());
          }
        } else if (
            auto integer = dyn_cast<IntegerAttr>(argument);
            integer && integer.getType().isSignlessInteger(1)
        ) {
          value = IntegerAttr::get(indexType, integer.getValue().getZExtValue());
        }
      } else if (
          auto integerType = dyn_cast<IntegerType>(*expected);
          integerType && integerType.isSignlessInteger(1)
      ) {
        if (auto integer = dyn_cast<IntegerAttr>(argument);
            integer && isa<IndexType>(integer.getType()) &&
            (integer.getValue().isZero() || integer.getValue().isOne())) {
          const APInt &number = integer.getValue();
          value = IntegerAttr::get(integerType, number.isOne() ? 1 : 0);
        }
      } else if (auto feltType = dyn_cast<FeltType>(*expected)) {
        if (auto integer = dyn_cast<IntegerAttr>(argument);
            integer && isa<IndexType>(integer.getType()) && !integer.getValue().isNegative() &&
            (!feltType.hasField() ||
             toDynamicAPInt(integer.getValue()) < feltType.getField().prime())) {
          value = FeltConstAttr::get(source.getContext(), integer.getValue(), feltType);
        }
      }
      bool matches = false;
      if (isa<TypeVarType>(*expected)) {
        matches = isa<TypeAttr>(value);
      } else if (auto typed = dyn_cast<TypedAttr>(value)) {
        matches = typed.getType() == *expected;
      }
      if (auto integer = dyn_cast<IntegerAttr>(value); !matches && integer &&
                                                       isa<IndexType>(integer.getType()) &&
                                                       expected->isSignlessInteger(1)) {
        matches = true;
      }
      if (!matches) {
        site->emitError("struct argument ") << argument << " does not match parameter @"
                                            << parameter.getSymName() << " of type " << *expected;
        return failure();
      }
    }
    normalized.push_back(value);
  }
  return ArrayAttr::get(source.getContext(), normalized);
}

/// Instantiate the dependency closure of llzk.main. Reserving each source/argument
/// pair before exploring its body makes repeated and cyclic requests share a clone.
class StructInstantiationWorklist {
  /// Keep source identity for call retargeting and the explored paths of its clone.
  struct Entry {
    StructDefOp source;
    StructDefOp clone;
    SmallVector<Operation *> visited;
    uint64_t remainingSteps;
  };

  using SpecializationKey = std::pair<Operation *, ArrayAttr>;
  enum class KeyMode { Required, Optional };

  ModuleOp root;
  ModuleOp topRoot;
  unsigned limit;
  uint64_t evaluationLimit;
  SymbolTableCollection tables;
  SmallVector<Entry> entries;
  DenseMap<SpecializationKey, unsigned> cache;
  DenseMap<Operation *, unsigned> cloneToId;

  /// Restore source identity when a rewritten body carries a previously
  /// specialized struct as a nested type argument.
  FailureOr<ArrayAttr> canonicalizeCloneArguments(ArrayAttr arguments, Operation *site) {
    bool invalid = false;
    AttrTypeReplacer replacer;
    replacer.addReplacement(
        [this, site, &invalid](StructType type) -> std::optional<std::pair<Type, WalkResult>> {
      auto found = type.getDefinition(tables, topRoot);
      if (failed(found)) {
        site->emitError("cannot resolve struct type argument ") << type;
        invalid = true;
        return std::make_pair(Type(type), WalkResult::skip());
      }
      if (!cloneToId.contains(found->get().getOperation())) {
        return std::nullopt;
      }
      auto origin = found->get()->getAttrOfType<SymbolRefAttr>(SPECIALIZATION_ORIGIN_ATTR);
      auto parameters = found->get()->getAttrOfType<ArrayAttr>(SPECIALIZATION_ARGUMENTS_ATTR);
      if (!origin || !parameters) {
        site->emitError("invalid struct specialization metadata for type argument ") << type;
        invalid = true;
        return std::make_pair(Type(type), WalkResult::skip());
      }
      return std::make_pair(Type(getStructTypeWithParams(origin, parameters)), WalkResult::skip());
    }
    );
    auto result = cast<ArrayAttr>(replacer.replace(arguments));
    if (invalid) {
      return failure();
    }
    return result;
  }

  /// Resolve a struct use and canonicalize its arguments for specialization
  /// identity. Optional lookups leave unresolved uses for a later pass.
  FailureOr<std::optional<SpecializationKey>>
  getSpecializationKey(StructType type, Operation *lookupFrom, Operation *site, KeyMode mode) {
    auto found = type.getDefinition(tables, lookupFrom);
    if (failed(found)) {
      return failure();
    }
    if (found->viaInclude()) {
      site->emitError("inline includes before template monomorphization");
      return failure();
    }
    ArrayAttr arguments = type.getParams();
    if (!arguments) {
      arguments = ArrayAttr::get(root.getContext(), {});
    }
    if (!llvm::all_of(arguments, [](Attribute argument) {
      return isConcreteStructParamAttr(argument);
    })) {
      if (mode == KeyMode::Optional) {
        return std::optional<SpecializationKey>();
      }
      site->emitError("template monomorphization requires concrete struct arguments") << type;
      return failure();
    }
    auto canonical = rebaseTemplateParams(tables, arguments, lookupFrom, topRoot, site);
    if (failed(canonical)) {
      return failure();
    }
    auto sourceArguments = canonicalizeCloneArguments(*canonical, site);
    if (failed(sourceArguments)) {
      return failure();
    }
    auto names = found->get().getType().getParams();
    if ((names ? names.size() : 0) != sourceArguments->size()) {
      if (mode == KeyMode::Optional) {
        return std::optional<SpecializationKey>();
      }
      site->emitError("struct specialization argument count mismatch");
      return failure();
    }
    auto normalized = normalizeTypedArguments(found->get(), *sourceArguments, site);
    if (failed(normalized)) {
      return failure();
    }
    return std::make_optional<SpecializationKey>(found->get().getOperation(), *normalized);
  }

  /// Reuse previously published identities when the pass is run again. Metadata
  /// IDs are dense worklist indices, including those referenced by rolled uses.
  LogicalResult loadExistingSpecializations() {
    SmallVector<StructDefOp> existing;
    root.walk([&existing](StructDefOp structure) {
      if (structure->hasAttr(SPECIALIZATION_ID_ATTR)) {
        existing.push_back(structure);
      }
    });
    for (auto structure : existing) {
      if (!structure->getAttrOfType<IntegerAttr>(SPECIALIZATION_ID_ATTR)) {
        return structure.emitError("expected integer specialization identity");
      }
    }
    llvm::sort(existing, [](StructDefOp left, StructDefOp right) {
      return left->getAttrOfType<IntegerAttr>(SPECIALIZATION_ID_ATTR).getInt() <
             right->getAttrOfType<IntegerAttr>(SPECIALIZATION_ID_ATTR).getInt();
    });
    for (auto clone : existing) {
      auto origin = clone->getAttrOfType<SymbolRefAttr>(SPECIALIZATION_ORIGIN_ATTR);
      auto arguments = clone->getAttrOfType<ArrayAttr>(SPECIALIZATION_ARGUMENTS_ATTR);
      if (!origin || !arguments ||
          clone->getAttrOfType<IntegerAttr>(SPECIALIZATION_ID_ATTR).getInt() !=
              static_cast<int64_t>(entries.size())) {
        return clone.emitError("invalid struct specialization metadata");
      }
      auto key = getSpecializationKey(
          StructType::get(origin, arguments), topRoot, clone, KeyMode::Required
      );
      if (failed(key) || !*key) {
        return failure();
      }
      clone->setAttr(SPECIALIZATION_ARGUMENTS_ATTR, (*key)->second);
      if (!cache.try_emplace(**key, entries.size()).second) {
        return clone.emitError("duplicate struct specialization identity");
      }
      cloneToId[clone.getOperation()] = entries.size();
      entries.push_back({cast<StructDefOp>((*key)->first), clone, {}, evaluationLimit});
    }
    return success();
  }

  /// Resolve source identity before forming the cache key, so symbol aliases do
  /// not create duplicate definitions. The limit bounds growing recursive families.
  FailureOr<unsigned> instantiate(StructType type, Operation *site) {
    auto resolved = getSpecializationKey(type, site, site, KeyMode::Required);
    if (failed(resolved) || !*resolved) {
      return failure();
    }
    auto key = **resolved;
    StructDefOp source = cast<StructDefOp>(key.first);
    if (auto existing = cloneToId.find(source); existing != cloneToId.end()) {
      return existing->second;
    }
    auto destination = getRootModule(source);
    if (failed(destination)) {
      return failure();
    }
    auto names = source.getType().getParams();
    auto localArguments = rebaseTemplateParams(tables, key.second, topRoot, *destination, site);
    if (failed(localArguments)) {
      return failure();
    }
    if (auto existing = cache.find(key); existing != cache.end()) {
      return existing->second;
    }
    if (entries.size() >= limit) {
      site->emitError("template monomorphization specialization limit exceeded");
      return failure();
    }
    Operation *branch = nullptr;
    source.walk([&branch](Operation *op) {
      if (isa<BranchOpInterface>(op)) {
        branch = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (branch) {
      branch->emitError("unstructured control flow is unsupported in template monomorphization");
      return failure();
    }
    StructSpecializationDiscovery::Bindings bindings;
    if (names) {
      for (auto [name, value] : llvm::zip_equal(names, *localArguments)) {
        bindings[name] = value;
      }
    }
    auto templ = source->getParentOfType<TemplateOp>();
    uint64_t remainingSteps = evaluationLimit;
    if (templ) {
      auto evaluated = StructSpecializationDiscovery(evaluationLimit)
                           .evaluateBindings(source, bindings, &remainingSteps);
      if (failed(evaluated)) {
        return failure();
      }
      bindings = std::move(*evaluated);
    }
    auto clone = source.clone();
    clone.setSymName((source.getSymName() + "__spec_" + Twine(entries.size())).str());
    Operation *parent = templ ? templ->getParentOp() : source->getParentOp();
    tables.getSymbolTable(parent).insert(clone);
    unsigned id = entries.size();
    cache[key] = id;
    cloneToId[clone.getOperation()] = id;
    entries.push_back({source, clone, {}, remainingSteps});
    convertCalleesInPlace(clone, bindings);
    TemplateTypeConverter parameterConverter(bindings);
    clone.walk([&parameterConverter](function::CallOp call) {
      if (auto parameters = call.getTemplateParamsAttr()) {
        SmallVector<Attribute> values;
        for (Attribute parameter : parameters) {
          values.push_back(parameterConverter.convertAttr(parameter));
        }
        call.setTemplateParamsAttr(ArrayAttr::get(call.getContext(), values));
      }
    });
    if (!names || names.empty()) {
      StructType canonicalSelf = source.getType();
      AttrTypeReplacer selfTypes;
      selfTypes.addReplacement(
          [canonicalSelf](StructType current) -> std::optional<std::pair<Type, WalkResult>> {
        if (current.getNameRef() != canonicalSelf.getNameRef() ||
            (current.getParams() && !current.getParams().empty())) {
          return std::nullopt;
        }
        return std::make_pair(Type(canonicalSelf), WalkResult::skip());
      }
      );
      selfTypes.recursivelyReplaceElementsIn(clone, true, false, true);
    }
    SmallVector<Diagnostic> diagnostics;
    if (failed(substituteStructBody(clone, source.getType(), bindings, diagnostics))) {
      return failure();
    }
    ConstReadOp unresolved;
    clone.walk([&unresolved](ConstReadOp read) {
      if (!unresolved) {
        unresolved = read;
      }
    });
    if (unresolved) {
      if (templ && templ.getConstNamed<TemplateExprOp>(unresolved.getConstNameAttr())) {
        site->emitError("template expression ")
            << unresolved.getConstNameAttr() << " could not be evaluated for " << type;
      } else {
        site->emitError("unresolved template binding ")
            << unresolved.getConstNameAttr() << " while specializing " << type;
      }
      return failure();
    }
    reportDelayedDiagnostics(site, std::move(diagnostics));
    Builder builder(root.getContext());
    clone->setAttr(SPECIALIZATION_ID_ATTR, builder.getI64IntegerAttr(id));
    auto origin = getPathRelativeToAncestor(source, topRoot, [source] {
      return source->emitError("specialization origin");
    });
    if (failed(origin)) {
      return failure();
    }
    clone->setAttr(SPECIALIZATION_ORIGIN_ATTR, *origin);
    clone->setAttr(SPECIALIZATION_ARGUMENTS_ATTR, key.second);
    return id;
  }

  /// Associate rolled uses with concrete definitions while keeping index tuples
  /// for array values and a deduplicated set of targets for affine calls.
  void recordFamily(const StructSpecializationRequest &request, unsigned id) {
    Builder builder(root.getContext());
    StringRef name;
    Attribute record;
    if (request.kind == StructSpecializationRequest::Kind::ArrayElement) {
      name = FAMILY_ATTR;
      SmallVector<Attribute> indices;
      for (int64_t index : request.arrayIndices) {
        indices.push_back(builder.getIndexAttr(index));
      }
      record = builder.getDictionaryAttr(
          {builder.getNamedAttr(FAMILY_INDICES_KEY, builder.getArrayAttr(indices)),
           builder.getNamedAttr(FAMILY_SPECIALIZATION_KEY, builder.getI64IntegerAttr(id)),
           builder.getNamedAttr(FAMILY_TYPE_KEY, TypeAttr::get(request.familyType))}
      );
    } else if (request.kind == StructSpecializationRequest::Kind::RolledCallResult) {
      name = FAMILY_SPECIALIZATIONS_ATTR;
      record = builder.getI64IntegerAttr(id);
    } else {
      return;
    }
    SmallVector<Attribute> records;
    if (auto existing = request.site->getAttrOfType<ArrayAttr>(name)) {
      llvm::append_range(records, existing);
    }
    if (!llvm::is_contained(records, record)) {
      records.push_back(record);
    }
    request.site->setAttr(name, builder.getArrayAttr(records));
  }

  /// Publish the union found from every array and affine call in this struct
  /// instance. An empty list records a known empty family.
  LogicalResult recordRolledCallTargets(const RolledCallTargets &targets) {
    auto call = cast<function::CallOp>(targets.site);
    SmallVector<Attribute> ids;
    if (auto existing = call->getAttrOfType<ArrayAttr>(FAMILY_SPECIALIZATIONS_ATTR)) {
      llvm::append_range(ids, existing);
    }
    for (StructType candidate : targets.candidates) {
      auto id = instantiate(candidate, call);
      if (failed(id)) {
        return failure();
      }
      auto attr = Builder(root.getContext()).getI64IntegerAttr(*id);
      if (!llvm::is_contained(ids, attr)) {
        ids.push_back(attr);
      }
    }
    call->setAttr(FAMILY_SPECIALIZATIONS_ATTR, ArrayAttr::get(root.getContext(), ids));
    return success();
  }

  /// Write a method reference using the caller's lookup root and the inserted
  /// clone's actual name, including any namespace prefix or collision suffix.
  LogicalResult setSpecializedCallee(function::CallOp call, StructDefOp clone, StringAttr method) {
    auto lookupRoot = getRootModule(call);
    if (failed(lookupRoot)) {
      return failure();
    }
    auto name = getPathRelativeToAncestor(clone, *lookupRoot, [call] {
      return call->emitError("specialized method owner");
    });
    if (failed(name)) {
      return failure();
    }
    call.setCalleeAttr(appendLeaf(*name, method));
    return success();
  }

  /// Choose a method's owner from the concrete struct carried by the call. Rolled
  /// affine calls keep their original callee and use family metadata for dispatch.
  LogicalResult retargetCall(function::CallOp call, const Entry &caller, bool visited) {
    auto target = call.getCalleeTarget(tables);
    if (failed(target)) {
      return failure();
    }
    if (target->viaInclude()) {
      return call.emitError("inline includes before template monomorphization");
    }
    auto owner = target->get()->getParentOfType<StructDefOp>();
    if (!owner) {
      if (visited) {
        return call.emitError(
            "free-function calls are not yet supported by template monomorphization"
        );
      }
      return success();
    }
    if (call->hasAttr(FAMILY_SPECIALIZATIONS_ATTR)) {
      return success();
    }
    SmallVector<Type> types(call.getResultTypes());
    llvm::append_range(types, call.getArgOperands().getTypes());
    for (Type type : types) {
      auto structure = dyn_cast<StructType>(type);
      if (!structure || hasAffineMapAttr(structure)) {
        continue;
      }
      auto found = structure.getDefinition(tables, call);
      if (failed(found)) {
        return failure();
      }
      auto cloneId = cloneToId.find(found->get().getOperation());
      if (cloneId != cloneToId.end() && entries[cloneId->second].source == owner) {
        return setSpecializedCallee(
            call, entries[cloneId->second].clone, target->get().getSymNameAttr()
        );
      }
      if (found->get() == owner) {
        return success();
      }
    }
    if (owner == caller.source) {
      return setSpecializedCallee(call, caller.clone, target->get().getSymNameAttr());
    }
    if (visited) {
      return call.emitError("cannot determine concrete struct owner for method call");
    }
    return success();
  }

public:
  StructInstantiationWorklist(ModuleOp module, unsigned maximum, uint64_t steps)
      : root(module), limit(maximum), evaluationLimit(steps) {}

  /// Discover each appended clone once, then rewrite uses after all identities
  /// have been reserved. This ordering also resolves mutually dependent types.
  LogicalResult run() {
    auto top = getTopRootModule(root);
    if (failed(top)) {
      return failure();
    }
    topRoot = *top;
    if (failed(loadExistingSpecializations())) {
      return failure();
    }
    auto main = getMainInstanceType(root);
    if (failed(main) || !*main) {
      return root.emitError("template monomorphization requires a concrete llzk.main");
    }
    auto mainId = instantiate(*main, root);
    if (failed(mainId)) {
      return failure();
    }
    for (unsigned i = 0; i < entries.size(); ++i) {
      auto clone = entries[i].clone;
      SmallVector<Operation *> visited;
      StructSpecializationDiscovery::CallTargets rolledCalls;
      auto requests =
          StructSpecializationDiscovery(entries[i].remainingSteps)
              .discover(clone, StructSpecializationDiscovery::Bindings(), &visited, &rolledCalls);
      if (failed(requests)) {
        return failure();
      }
      entries[i].visited = std::move(visited);
      for (const auto &request : *requests) {
        auto id = instantiate(request.type, request.site);
        if (failed(id)) {
          return failure();
        }
        recordFamily(request, *id);
      }
      for (const auto &targets : rolledCalls) {
        if (failed(recordRolledCallTargets(targets))) {
          return failure();
        }
      }
    }
    // Each replacer resolves input names in one root. The cache identifies the
    // definition independently of its spelling, including uses in preserved paths
    // that discovery did not execute.
    llvm::MapVector<Operation *, SmallVector<StructDefOp>> clonesByRoot;
    for (Entry &entry : entries) {
      auto lookupRoot = getRootModule(entry.clone);
      if (failed(lookupRoot)) {
        return failure();
      }
      clonesByRoot[lookupRoot->getOperation()].push_back(entry.clone);
    }
    for (auto &[rootOp, clones] : clonesByRoot) {
      auto lookupRoot = cast<ModuleOp>(rootOp);
      bool invalid = false;
      // These checks validate names after substitution and clone insertion. The
      // callback receives only a type, so failures are reported at its lookup root.
      // For verified input, these failures indicate a bug in the pass.
      AttrTypeReplacer replacer;
      replacer.addReplacement(
          [this, lookupRoot,
           &invalid](StructType type) -> std::optional<std::pair<Type, WalkResult>> {
        auto key = getSpecializationKey(type, lookupRoot, lookupRoot, KeyMode::Optional);
        if (failed(key)) {
          invalid = true;
          return std::make_pair(Type(type), WalkResult::skip());
        }
        if (!*key) {
          return std::nullopt;
        }
        auto existing = cache.find(**key);
        if (existing == cache.end()) {
          return std::nullopt;
        }
        auto name =
            getPathRelativeToAncestor(entries[existing->second].clone, lookupRoot, [lookupRoot] {
          return lookupRoot->emitError("specialized struct");
        });
        if (failed(name)) {
          invalid = true;
          return std::make_pair(Type(type), WalkResult::skip());
        }
        return std::make_pair(Type(StructType::get(*name)), WalkResult::skip());
      }
      );
      for (auto clone : clones) {
        // Provenance uses top-root names, not the root of the rewritten body.
        auto origin = clone->removeAttr(SPECIALIZATION_ORIGIN_ATTR);
        auto args = clone->removeAttr(SPECIALIZATION_ARGUMENTS_ATTR);
        replacer.recursivelyReplaceElementsIn(clone, true, false, true);
        clone->setAttr(SPECIALIZATION_ORIGIN_ATTR, origin);
        clone->setAttr(SPECIALIZATION_ARGUMENTS_ATTR, args);
      }
      if (invalid) {
        return failure();
      }
    }
    for (Entry &entry : entries) {
      DenseSet<Operation *> visited(entry.visited.begin(), entry.visited.end());
      auto result = entry.clone.walk([this, &entry, &visited](function::CallOp call) {
        return WalkResult(retargetCall(call, entry, visited.contains(call)));
      });
      if (result.wasInterrupted()) {
        return failure();
      }
    }
    auto mainRoot = getRootModule(root);
    if (failed(mainRoot)) {
      return failure();
    }
    auto mainName = getPathRelativeToAncestor(entries[*mainId].clone, *mainRoot, [this] {
      return root.emitError("specialized main");
    });
    if (failed(mainName)) {
      return failure();
    }
    root->setAttr(MAIN_ATTR_NAME, TypeAttr::get(StructType::get(*mainName)));
    return success();
  }
};

/// Materialize the concrete struct definitions required by the module entry point.
class TemplateMonomorphizationPass
    : public llzk::polymorphic::impl::TemplateMonomorphizationPassBase<
          TemplateMonomorphizationPass> {
public:
  using Base = TemplateMonomorphizationPassBase<TemplateMonomorphizationPass>;
  using Base::Base;

private:
  void runOnOperation() override {
    if (failed(
            StructInstantiationWorklist(getOperation(), specializationLimit, evaluationLimit).run()
        )) {
      signalPassFailure();
    }
  }
};

} // namespace
