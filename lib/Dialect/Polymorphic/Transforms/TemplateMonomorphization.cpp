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
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/Walk.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/AttrTypeSubElements.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLFunctionalExtras.h>

#include <cstdint>

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

/// Normalize typed arguments for specialization identity and substitution.
/// Fail if an argument cannot be converted to the declared parameter type.
/// An i1 binding accepts only boolean values or index zero/one; subsequent reads
/// and forwarding use the normalized boolean value.
/// Untyped parameters retain their supplied attributes.
FailureOr<ArrayAttr> normalizeTypedArguments(
    Operation *source, ArrayAttr arguments, Operation *site, StringRef kind = "struct"
) {
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
          APInt number = felt.getValue();
          if (felt.getType().hasField()) {
            const Field &field = felt.getType().getField();
            number = toAPInt(field.reduce(number), field.bitWidth());
          }
          if (number.getActiveBits() > 63) {
            return site->emitError("field element ")
                   << argument << " cannot be converted to index parameter @"
                   << parameter.getSymName()
                   << ": value exceeds the nonnegative signed 64-bit index range";
          }
          value = IntegerAttr::get(indexType, number.getZExtValue());
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
            integer && isa<IndexType>(integer.getType())) {
          const APInt &number = integer.getValue();
          if (!number.isZero() && !number.isOne()) {
            return site->emitError("index argument ")
                   << argument << " cannot be converted to i1 parameter @" << parameter.getSymName()
                   << ": expected zero or one";
          }
          value = IntegerAttr::get(integerType, number.isOne() ? 1 : 0);
        }
      } else if (auto feltType = dyn_cast<FeltType>(*expected)) {
        if (auto integer = dyn_cast<IntegerAttr>(argument);
            integer && isa<IndexType>(integer.getType())) {
          if (feltType.hasField()) {
            const Field &field = feltType.getField();
            auto reduced = field.reduce(llvm::DynamicAPInt(integer.getValue()));
            value = FeltConstAttr::get(
                source->getContext(), toAPInt(reduced, field.bitWidth()), feltType
            );
          } else {
            if (integer.getValue().isNegative()) {
              return site->emitError("index argument ")
                     << argument << " cannot be converted to felt parameter @"
                     << parameter.getSymName() << ": negative values require a known field modulus";
            }
            value = FeltConstAttr::get(source->getContext(), integer.getValue(), feltType);
          }
        } else if (
            auto felt = dyn_cast<FeltConstAttr>(argument);
            felt && (!felt.getType().getFieldName() || felt.getType() == feltType)
        ) {
          APInt number = felt.getValue();
          if (feltType.hasField()) {
            const Field &field = feltType.getField();
            number = toAPInt(field.reduce(number), field.bitWidth());
          }
          value = FeltConstAttr::get(source->getContext(), number, feltType);
        }
      }
      bool matches = false;
      if (isa<TypeVarType>(*expected)) {
        matches = isa<TypeAttr>(value);
      } else if (auto typed = dyn_cast<TypedAttr>(value)) {
        matches = typed.getType() == *expected;
      }
      if (!matches) {
        return site->emitError() << kind << " argument " << argument
                                 << " does not match parameter @" << parameter.getSymName()
                                 << " of type " << *expected;
      }
    }
    normalized.push_back(value);
  }
  return ArrayAttr::get(source->getContext(), normalized);
}

/// Instantiate the dependency closure of llzk.main. Reserving each source/argument
/// pair before exploring its body makes repeated and cyclic requests share a clone.
class TemplateInstantiationWorklist {
  /// Keep source identity for call retargeting and the explored paths of its clone.
  struct Entry {
    StructDefOp source;
    StructDefOp clone;
    SmallVector<Operation *> visited;
    uint64_t remainingSteps;
  };

  struct FunctionEntry {
    function::FuncDefOp source;
    function::FuncDefOp clone;
    SmallVector<Operation *> visited;
    uint64_t remainingSteps;
    bool active;
    bool discovered;
    /// The call that requested this clone, if created during this pass run.
    std::optional<Location> request;
  };

  using SpecializationKey = std::pair<Operation *, ArrayAttr>;
  enum class KeyMode : std::uint8_t { Required, Optional };

  ModuleOp root;
  ModuleOp topRoot;
  unsigned limit;
  uint64_t evaluationLimit;
  SymbolTableCollection tables;
  SmallVector<Entry> entries;
  SmallVector<FunctionEntry> functions;
  DenseMap<SpecializationKey, unsigned> cache;
  DenseMap<SpecializationKey, unsigned> functionCache;
  DenseMap<Operation *, unsigned> cloneToId;
  DenseMap<Operation *, unsigned> functionCloneToId;

  /// Restore source identity when a rewritten body carries a previously
  /// specialized struct as a nested type argument.
  FailureOr<ArrayAttr> canonicalizeCloneArguments(ArrayAttr arguments, Operation *site) {
    AttrTypeReplacer replacer;
    replacer.addReplacement(
        [this, site](StructType type) -> std::optional<std::pair<Type, WalkResult>> {
      auto found = type.getDefinition(tables, topRoot);
      if (failed(found)) {
        site->emitError("cannot resolve struct type argument ") << type;
        return std::make_pair(Type(type), WalkResult::interrupt());
      }
      if (!cloneToId.contains(found->get().getOperation())) {
        return std::nullopt;
      }
      auto origin = found->get()->getAttrOfType<SymbolRefAttr>(SPECIALIZATION_ORIGIN_ATTR);
      auto parameters = found->get()->getAttrOfType<ArrayAttr>(SPECIALIZATION_ARGUMENTS_ATTR);
      if (!origin || !parameters) {
        site->emitError("invalid struct specialization metadata for type argument ") << type;
        return std::make_pair(Type(type), WalkResult::interrupt());
      }
      return std::make_pair(Type(getStructTypeWithParams(origin, parameters)), WalkResult::skip());
    }
    );
    auto result = dyn_cast_if_present<ArrayAttr>(replacer.replace(arguments));
    if (!result) {
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
      return site->emitError("inline includes before template monomorphization");
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
      return site->emitError("template monomorphization requires concrete struct arguments")
             << type;
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
      return site->emitError("struct specialization argument count mismatch");
    }
    auto normalized = normalizeTypedArguments(found->get().getOperation(), *sourceArguments, site);
    if (failed(normalized)) {
      return failure();
    }
    return std::make_optional<SpecializationKey>(found->get().getOperation(), *normalized);
  }

  /// Reuse previously published identities when the pass is run again. Metadata
  /// IDs are dense worklist indices, including those referenced by rolled uses.
  LogicalResult loadExistingSpecializations() {
    auto existing = walkCollect<StructDefOp>(root, [](StructDefOp structure) {
      return structure->hasAttr(SPECIALIZATION_ID_ATTR);
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
              checkedCast<int64_t>(entries.size())) {
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

  /// Recover function clones so a second pass run reuses their definitions.
  LogicalResult loadExistingFunctions() {
    auto existing = walkCollect<function::FuncDefOp>(root, [](function::FuncDefOp function) {
      return function->hasAttr(SPECIALIZATION_ORIGIN_ATTR);
    });
    for (auto clone : existing) {
      auto origin = clone->getAttrOfType<SymbolRefAttr>(SPECIALIZATION_ORIGIN_ATTR);
      auto arguments = clone->getAttrOfType<ArrayAttr>(SPECIALIZATION_ARGUMENTS_ATTR);
      if (!origin || !arguments) {
        return clone.emitError("invalid function specialization metadata");
      }
      auto source = tables.lookupSymbolIn<function::FuncDefOp>(topRoot, origin);
      if (!source) {
        return clone.emitError("cannot resolve function specialization origin");
      }
      auto templ = source->getParentOfType<TemplateOp>();
      if (arguments.size() !=
              (templ ? llvm::range_size(templ.getConstOps<TemplateParamOp>()) : 0) ||
          !llvm::all_of(arguments, [](Attribute argument) {
        return isConcreteStructParamAttr(argument);
      })) {
        return clone.emitError("invalid function specialization arguments");
      }
      auto key = SpecializationKey(source.getOperation(), arguments);
      if (!functionCache.try_emplace(key, functions.size()).second) {
        return clone.emitError("duplicate function specialization identity");
      }
      functionCloneToId[clone.getOperation()] = functions.size();
      functions.push_back({source, clone, {}, evaluationLimit, false, false, std::nullopt});
    }
    return success();
  }

  /// Resolve explicit or signature-inferred template arguments at a free call.
  FailureOr<SpecializationKey> getFunctionKey(
      function::CallOp call, function::FuncDefOp source, ArrayRef<StringRef> targetNamespace
  ) {
    auto templ = source->getParentOfType<TemplateOp>();
    SmallVector<Attribute> arguments;
    if (templ) {
      auto parameters = templ.getConstOps<TemplateParamOp>();
      auto explicitArguments = call.getTemplateParamsAttr();
      if (explicitArguments && explicitArguments.size() != llvm::range_size(parameters)) {
        return call.emitError("function specialization argument count mismatch");
      }
      std::optional<UnificationMap> unified;
      bool needsInference =
          !explicitArguments || llvm::any_of(explicitArguments, [](Attribute arg) {
        return classifyAttrConcreteness(arg) == AttrConcreteness::Wildcard;
      });
      if (needsInference) {
        auto result =
            call.unifyTypeSignatureWithNamespace(source.getFunctionType(), targetNamespace);
        if (failed(result)) {
          return call.emitError("cannot infer concrete function specialization");
        }
        unified = *result;
      }
      for (auto [index, parameter] : llvm::enumerate(parameters)) {
        Attribute value = explicitArguments ? explicitArguments[index] : Attribute();
        if (!value || classifyAttrConcreteness(value) == AttrConcreteness::Wildcard) {
          value = inferUnifiedParam(*unified, FlatSymbolRefAttr::get(parameter.getSymNameAttr()))
                      .value_or(Attribute());
        }
        if (!value || !isConcreteStructParamAttr(value)) {
          return call.emitError("cannot resolve free-function template parameter @")
                 << parameter.getSymName();
        }
        if (failed(call.verifyTemplateParamValueCompatibility(value, parameter))) {
          return failure();
        }
        arguments.push_back(value);
      }
    }
    auto canonical = rebaseTemplateParams(
        tables, ArrayAttr::get(root.getContext(), arguments), call, topRoot, call
    );
    if (failed(canonical)) {
      return failure();
    }
    auto sourceArguments = canonicalizeCloneArguments(*canonical, call);
    if (failed(sourceArguments)) {
      return failure();
    }
    auto normalized = normalizeTypedArguments(source, *sourceArguments, call, "function");
    if (failed(normalized)) {
      return failure();
    }
    return SpecializationKey(source.getOperation(), *normalized);
  }

  /// Substitute explicit arguments on nested calls before discovering their targets.
  static void
  convertCallArguments(Operation *clone, const StructSpecializationDiscovery::Bindings &bindings) {
    TemplateTypeConverter converter(bindings);
    clone->walk([&converter](function::CallOp call) {
      if (auto parameters = call.getTemplateParamsAttr()) {
        SmallVector<Attribute> values;
        for (Attribute parameter : parameters) {
          values.push_back(converter.convertAttr(parameter));
        }
        call.setTemplateParamsAttr(ArrayAttr::get(call.getContext(), values));
      }
    });
  }

  /// Bind concrete template parameters and evaluate the source's template
  /// expressions before substituting the clone body.
  FailureOr<StructSpecializationDiscovery::Bindings> bindArguments(
      Operation *source, ArrayRef<Attribute> names, ArrayAttr arguments, uint64_t &remainingSteps
  ) {
    StructSpecializationDiscovery::Bindings bindings;
    for (auto [name, argument] : llvm::zip_equal(names, arguments)) {
      bindings[name] = argument;
    }
    remainingSteps = evaluationLimit;
    if (!source->getParentOfType<TemplateOp>()) {
      return bindings;
    }
    return StructSpecializationDiscovery(evaluationLimit)
        .evaluateBindings(source, bindings, &remainingSteps);
  }

  /// Report a template read left after substitution at the specialization
  /// request site, distinguishing an unknown expression from a missing binding.
  static LogicalResult diagnoseUnresolvedBinding(
      Operation *clone, TemplateOp templ, Operation *site,
      llvm::function_ref<void(InFlightDiagnostic &)> describe
  ) {
    ConstReadOp unresolved;
    clone->walk([&unresolved](ConstReadOp read) {
      if (!unresolved) {
        unresolved = read;
      }
    });
    if (!unresolved) {
      return success();
    }
    if (templ && templ.getConstNamed<TemplateExprOp>(unresolved.getConstNameAttr())) {
      auto diagnostic = site->emitError("template expression ");
      diagnostic << unresolved.getConstNameAttr() << " could not be evaluated for ";
      describe(diagnostic);
      return diagnostic;
    }
    auto diagnostic = site->emitError("unresolved template binding ");
    diagnostic << unresolved.getConstNameAttr() << " while specializing ";
    describe(diagnostic);
    return diagnostic;
  }

  /// Reject unstructured branches before cloning a definition, including
  /// branches in regions that discovery would not explore.
  static LogicalResult verifyStructuredControlFlow(Operation *source) {
    Operation *branch = nullptr;
    source->walk([&branch](Operation *op) {
      if (isa<BranchOpInterface>(op)) {
        branch = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (branch) {
      return branch->emitError(
          "unstructured control flow is unsupported in template monomorphization"
      );
    }
    return success();
  }

  /// A retained call needs a concrete callee when type rewriting will change
  /// one of its operand or result types to a struct specialization.
  FailureOr<bool> hasSpecializedSignature(function::CallOp call) {
    bool required = false;
    bool invalid = false;
    auto inspect = [this, call, &required, &invalid](Type type) {
      type.walk([this, call, &required, &invalid](StructType structure) {
        auto key = getSpecializationKey(structure, call, call, KeyMode::Optional);
        if (failed(key)) {
          invalid = true;
          return WalkResult::interrupt();
        }
        if (*key && cache.contains(**key)) {
          required = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    };
    for (Type type : call.getResultTypes()) {
      inspect(type);
    }
    for (Type type : call.getArgOperands().getTypes()) {
      inspect(type);
    }
    if (invalid) {
      return failure();
    }
    return required;
  }

  /// Clone a reachable free function and retarget its call to that clone.
  LogicalResult instantiateFunction(function::CallOp call, bool active = true) {
    auto target = call.getCalleeTarget(tables);
    if (failed(target)) {
      return failure();
    }
    if (target->viaInclude()) {
      return call.emitError("inline includes before template monomorphization");
    }
    auto source = target->get();
    if (auto clone = functionCloneToId.find(source.getOperation());
        clone != functionCloneToId.end()) {
      functions[clone->second].active |= active;
      return success();
    }
    if (source.isExternal()) {
      return success();
    }
    auto key = getFunctionKey(call, source, target->getNamespace());
    if (failed(key)) {
      return failure();
    }
    unsigned id;
    if (auto found = functionCache.find(*key); found != functionCache.end()) {
      id = found->second;
      functions[id].active |= active;
    } else {
      if (entries.size() + functions.size() >= limit) {
        return call.emitError("template monomorphization specialization limit exceeded");
      }
      if (failed(verifyStructuredControlFlow(source))) {
        return failure();
      }
      auto destination = getRootModule(source);
      if (failed(destination)) {
        return failure();
      }
      auto localArguments = rebaseTemplateParams(tables, key->second, topRoot, *destination, call);
      if (failed(localArguments)) {
        return failure();
      }
      auto origin = getPathRelativeToAncestor(source, topRoot, [call] {
        return call->emitError("free-function specialization origin");
      });
      if (failed(origin)) {
        return failure();
      }
      auto templ = source->getParentOfType<TemplateOp>();
      SmallVector<Attribute> names;
      if (templ) {
        for (auto parameter : templ.getConstOps<TemplateParamOp>()) {
          names.push_back(FlatSymbolRefAttr::get(parameter.getSymNameAttr()));
        }
      }
      uint64_t remainingSteps;
      auto evaluated = bindArguments(source, names, *localArguments, remainingSteps);
      if (failed(evaluated)) {
        return failure();
      }
      auto bindings = std::move(*evaluated);
      auto clone = source.clone();
      clone.setSymName((source.getSymName() + "__spec_" + Twine(functions.size())).str());
      Operation *parent = templ ? templ->getParentOp() : source->getParentOp();
      tables.getSymbolTable(parent).insert(clone);
      id = functions.size();
      functionCache[*key] = id;
      functionCloneToId[clone.getOperation()] = id;
      functions.push_back({source, clone, {}, remainingSteps, active, false, call.getLoc()});
      convertCalleesInPlace(clone, bindings);
      convertCallArguments(clone, bindings);
      SmallVector<Diagnostic> diagnostics;
      if (failed(substituteFunctionBody(clone, bindings, diagnostics))) {
        return failure();
      }
      if (failed(diagnoseUnresolvedBinding(
              clone, templ, call,
              [name = *origin, arguments = key->second](InFlightDiagnostic &diagnostic) {
        diagnostic << "free function " << name << " with arguments " << arguments;
      }
          ))) {
        return failure();
      }
      reportDelayedDiagnostics(call, std::move(diagnostics));
      clone->setAttr(SPECIALIZATION_ORIGIN_ATTR, *origin);
      clone->setAttr(SPECIALIZATION_ARGUMENTS_ATTR, key->second);
    }
    auto lookupRoot = getRootModule(call);
    if (failed(lookupRoot)) {
      return failure();
    }
    auto name = getPathRelativeToAncestor(functions[id].clone, *lookupRoot, [call] {
      return call->emitError("specialized free-function callee");
    });
    if (failed(name)) {
      return failure();
    }
    call.setCalleeAttr(*name);
    call.removeTemplateParamsAttr();
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
    if (entries.size() + functions.size() >= limit) {
      return site->emitError("template monomorphization specialization limit exceeded");
    }
    if (failed(verifyStructuredControlFlow(source))) {
      return failure();
    }
    auto templ = source->getParentOfType<TemplateOp>();
    uint64_t remainingSteps;
    auto evaluated = bindArguments(
        source, names ? names.getValue() : ArrayRef<Attribute>(), *localArguments, remainingSteps
    );
    if (failed(evaluated)) {
      return failure();
    }
    auto bindings = std::move(*evaluated);
    TemplateTypeConverter parameterConverter(bindings);
    auto reads = source.walk([&bindings, &parameterConverter](ConstReadOp read) -> WalkResult {
      return WalkResult(resolveConstReadBinding(
          read, bindings.lookup(read.getConstNameAttr()),
          parameterConverter.convertType(read.getType())
      ));
    });
    if (reads.wasInterrupted()) {
      return failure();
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
    // Materialize reads with the same conversions used by expression discovery.
    auto constants = clone.walk([&bindings, &parameterConverter](ConstReadOp read) -> WalkResult {
      auto value = resolveConstReadBinding(
          read, bindings.lookup(read.getConstNameAttr()),
          parameterConverter.convertType(read.getType())
      );
      if (failed(value)) {
        return WalkResult::interrupt();
      }
      if (!*value) {
        return WalkResult::advance();
      }
      OpBuilder builder(read);
      Value constant;
      if (auto felt = dyn_cast<FeltConstAttr>(*value)) {
        constant = FeltConstantOp::create(builder, read.getLoc(), felt);
      } else {
        constant = arith::ConstantOp::create(builder, read.getLoc(), cast<TypedAttr>(*value));
      }
      read.replaceAllUsesWith(constant);
      read.erase();
      return WalkResult::advance();
    });
    if (constants.wasInterrupted()) {
      return failure();
    }
    SmallVector<Diagnostic> diagnostics;
    if (failed(substituteStructBody(clone, source.getType(), bindings, diagnostics))) {
      return failure();
    }
    if (failed(diagnoseUnresolvedBinding(
            clone, templ, site, [type](InFlightDiagnostic &diagnostic) { diagnostic << type; }
        ))) {
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
  LogicalResult retargetCall(function::CallOp call, const Entry *caller, bool visited) {
    auto target = call.getCalleeTarget(tables);
    if (failed(target)) {
      return failure();
    }
    if (target->viaInclude()) {
      return call.emitError("inline includes before template monomorphization");
    }
    auto owner = target->get()->getParentOfType<StructDefOp>();
    if (!owner) {
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
    if (caller && owner == caller->source) {
      return setSpecializedCallee(call, caller->clone, target->get().getSymNameAttr());
    }
    if (visited) {
      return call.emitError("cannot determine concrete struct owner for method call");
    }
    return success();
  }

  /// External free functions retain their original signatures. Diagnose calls
  /// whose struct types would change during the final type rewrite.
  LogicalResult verifyExternalCalls() {
    auto check = [this](Operation *clone) {
      return clone->walk([this](function::CallOp call) {
        auto target = call.getCalleeTarget(tables);
        if (failed(target)) {
          return WalkResult::interrupt();
        }
        if (target->get()->getParentOfType<StructDefOp>() || !target->get().isExternal()) {
          return WalkResult::advance();
        }
        auto required = hasSpecializedSignature(call);
        if (failed(required)) {
          return WalkResult::interrupt();
        }
        if (*required) {
          call.emitError("external function ")
              << call.getCalleeAttr()
              << " cannot be called with specialized struct types; external declarations are "
                 "not specialized";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    };
    for (Entry &entry : entries) {
      if (check(entry.clone).wasInterrupted()) {
        return failure();
      }
    }
    for (FunctionEntry &entry : functions) {
      if (check(entry.clone).wasInterrupted()) {
        return failure();
      }
    }
    return success();
  }

public:
  TemplateInstantiationWorklist(ModuleOp module, unsigned maximum, uint64_t steps)
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
    if (failed(loadExistingFunctions())) {
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
    // Discovery appends to these worklists. Indices remain valid as they grow.
    for (unsigned structIndex = 0;;) {
      bool isStruct = structIndex < entries.size();
      unsigned functionIndex = 0;
      if (!isStruct) {
        while (functionIndex < functions.size() &&
               (!functions[functionIndex].active || functions[functionIndex].discovered)) {
          ++functionIndex;
        }
        if (functionIndex == functions.size()) {
          break;
        }
      }
      Operation *clone = isStruct ? entries[structIndex].clone.getOperation()
                                  : functions[functionIndex].clone.getOperation();
      SmallVector<Operation *> visited;
      uint64_t steps =
          isStruct ? entries[structIndex].remainingSteps : functions[functionIndex].remainingSteps;
      std::optional<ScopedDiagnosticHandler> requestNote;
      if (!isStruct && functions[functionIndex].request) {
        Location request = *functions[functionIndex].request;
        Attribute origin = clone->getAttr(SPECIALIZATION_ORIGIN_ATTR);
        Attribute arguments = clone->getAttr(SPECIALIZATION_ARGUMENTS_ATTR);
        requestNote.emplace(clone->getContext(), [request, origin, arguments](Diagnostic &diag) {
          if (diag.getSeverity() == DiagnosticSeverity::Error) {
            diag.attachNote(request)
                << "while specializing free function " << origin << " with arguments " << arguments;
          }
          return failure();
        });
      }
      StructSpecializationDiscovery::CallTargets rolledCalls;
      StructSpecializationDiscovery discovery(steps);
      auto requests = discovery.discover(
          clone, StructSpecializationDiscovery::Bindings(), &visited, &rolledCalls
      );
      if (failed(requests)) {
        return failure();
      }
      DenseSet<Operation *> visitedSet(visited.begin(), visited.end());
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
      auto calls = clone->walk([this, &visitedSet](function::CallOp call) {
        if (!visitedSet.contains(call)) {
          return WalkResult::advance();
        }
        auto target = call.getCalleeTarget(tables);
        if (failed(target)) {
          return WalkResult::interrupt();
        }
        if (!target->get()->getParentOfType<StructDefOp>() && failed(instantiateFunction(call))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (calls.wasInterrupted()) {
        return failure();
      }
      if (isStruct) {
        entries[structIndex].visited = std::move(visited);
        ++structIndex;
      } else {
        functions[functionIndex].visited = std::move(visited);
        functions[functionIndex].discovered = true;
      }
    }
    // A call in an unselected branch is not a dependency to explore. Retain
    // its template callee unless a known struct specialization changes its
    // signature; in that case a concrete callee is needed for valid IR.
    for (unsigned structIndex = 0, functionIndex = 0;
         structIndex < entries.size() || functionIndex < functions.size();) {
      bool isStruct = structIndex < entries.size();
      Operation *clone = isStruct ? entries[structIndex].clone.getOperation()
                                  : functions[functionIndex].clone.getOperation();
      auto &visited = isStruct ? entries[structIndex].visited : functions[functionIndex].visited;
      DenseSet<Operation *> visitedSet(visited.begin(), visited.end());
      auto calls = clone->walk([this, &visitedSet](function::CallOp call) {
        if (visitedSet.contains(call)) {
          return WalkResult::advance();
        }
        auto target = call.getCalleeTarget(tables);
        if (failed(target)) {
          return WalkResult::interrupt();
        }
        if (target->get()->getParentOfType<StructDefOp>()) {
          return WalkResult::advance();
        }
        auto required = hasSpecializedSignature(call);
        if (failed(required) || (*required && failed(instantiateFunction(call, false)))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (calls.wasInterrupted()) {
        return failure();
      }
      if (isStruct) {
        ++structIndex;
      } else {
        ++functionIndex;
      }
    }
    if (failed(verifyExternalCalls())) {
      return failure();
    }
    // Each replacer resolves input names in one root. The cache identifies the
    // definition independently of its spelling, including uses in preserved paths
    // that discovery did not execute.
    llvm::MapVector<Operation *, SmallVector<Operation *>> clonesByRoot;
    for (Entry &entry : entries) {
      auto lookupRoot = getRootModule(entry.clone);
      if (failed(lookupRoot)) {
        return failure();
      }
      clonesByRoot[lookupRoot->getOperation()].push_back(entry.clone);
    }
    for (FunctionEntry &entry : functions) {
      auto lookupRoot = getRootModule(entry.clone);
      if (failed(lookupRoot)) {
        return failure();
      }
      clonesByRoot[lookupRoot->getOperation()].push_back(entry.clone);
    }
    for (auto &[rootOp, clones] : clonesByRoot) {
      auto lookupRoot = cast<ModuleOp>(rootOp);
      // recursivelyReplaceElementsIn returns void and suppresses null replacements,
      // so the callback must retain a failure flag for this operation-level rewrite.
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
        return WalkResult(retargetCall(call, &entry, visited.contains(call)));
      });
      if (result.wasInterrupted()) {
        return failure();
      }
    }
    for (FunctionEntry &entry : functions) {
      DenseSet<Operation *> visited(entry.visited.begin(), entry.visited.end());
      auto result = entry.clone.walk([this, &visited](function::CallOp call) {
        return WalkResult(retargetCall(call, nullptr, visited.contains(call)));
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

/// Materialize the concrete definitions required by the module entry point.
class TemplateMonomorphizationPass
    : public llzk::polymorphic::impl::TemplateMonomorphizationPassBase<
          TemplateMonomorphizationPass> {
public:
  using Base = TemplateMonomorphizationPassBase<TemplateMonomorphizationPass>;
  using Base::Base;

private:
  void runOnOperation() override {
    if (failed(TemplateInstantiationWorklist(getOperation(), specializationLimit, evaluationLimit)
                   .run())) {
      signalPassFailure();
    }
  }
};

} // namespace
