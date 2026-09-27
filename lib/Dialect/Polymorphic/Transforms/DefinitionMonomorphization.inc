//===-- DefinitionMonomorphization.inc ---------------------------*- C++ -*-===//
// Part of the LLZK Project, under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0
// Included by FlatteningPass.cpp to share private definition-cloning helpers.
// The worklist below never invokes the legacy pass driver.

namespace {

/// Definition-only worklist. This deliberately does not invoke the flattening driver,
/// greedy folding, inlining, or loop transformations. Shared cloning helpers above
/// substitute template constants without changing control-flow structure.
class DefinitionRegistry {
  struct Entry {
    Operation *source;
    Operation *definition;
    ArrayAttr arguments;
    std::optional<unsigned> parent;
  };
  ModuleOp root;
  unsigned limit;
  ConversionTracker tracker;
  SymbolTableCollection tables;
  DenseMap<std::pair<Operation *, ArrayAttr>, unsigned> cache;
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
      auto definition = specialized->getDefinition(tables, root);
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
        auto target = concrete->getDefinition(tables, root);
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

  /// Substitute explicit arguments on nested calls before their symbol verification.
  void substituteCallArguments(Operation *op, const DenseMap<Attribute, Attribute> &bindings) {
    AttrTypeReplacer replacer;
    replacer.addReplacement([&](TypeVarType type) -> std::optional<Type> {
      if (auto value = dyn_cast_or_null<TypeAttr>(bindings.lookup(type.getNameRef()))) {
        return value.getValue();
      }
      return std::nullopt;
    });
    replacer.addReplacement([&](SymbolRefAttr name) -> std::optional<Attribute> {
      if (auto value = bindings.lookup(name)) {
        return value;
      }
      return std::nullopt;
    });
    op->walk([&](CallOp call) {
      if (auto args = call.getTemplateParamsAttr()) {
        call.setTemplateParamsAttr(llvm::cast<ArrayAttr>(replacer.replace(args)));
      }
    });
  }

  ArrayAttr emptyArgs() { return ArrayAttr::get(root.getContext(), {}); }

  /// Reserve an identity before cloning, so repeated requests share a work item.
  FailureOr<unsigned> reserve(Operation *source, ArrayAttr args) {
    if (!args) {
      args = emptyArgs();
    }
    auto key = std::make_pair(source, args);
    if (auto it = cache.find(key); it != cache.end()) {
      return it->second;
    }
    for (auto ancestor = current; ancestor; ancestor = entries[*ancestor].parent) {
      if (entries[*ancestor].source == source) {
        auto diag = source->emitError("recursive specialization is unsupported");
        diag << ": requested " << args;
        for (auto p = current; p; p = entries[*p].parent) {
          diag.attachNote(entries[*p].source->getLoc())
              << "specialization chain: "
              << getFullyQualifiedName(llvm::cast<SymbolOpInterface>(entries[*p].source))
              << entries[*p].arguments;
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
    entries.push_back({source, nullptr, args, current});
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
  }

  FailureOr<StructType> specialize(StructType type) {
    if (isFamily(type)) {
      return type;
    }
    auto found = type.getDefinition(tables, root);
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
    if (type.getParams() &&
        !llvm::all_of(type.getParams(), [](Attribute a) { return isConcreteAttr<false>(a); })) {
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
    StructDefOp clone;
    if (!isNullOrEmpty(type.getParams())) {
      Step1_InstantiateStructs::StructCloner cloner(tracker, root);
      auto result = cloner.createInstantiatedClone(type);
      if (failed(result)) {
        return failure();
      }
      tables = SymbolTableCollection();
      auto def = result->getDefinition(tables, root);
      if (failed(def)) {
        return failure();
      }
      clone = def->get();
      DenseMap<Attribute, Attribute> bindings;
      for (auto [name, value] : llvm::zip(source.getType().getParams(), type.getParams())) {
        bindings[name] = value;
      }
      evaluateTemplateExprs(source->getParentOfType<TemplateOp>(), bindings);
      substituteCallArguments(clone, bindings);
    } else {
      clone = source.clone();
      clone.setSymName(("__llzk_spec_" + Twine(*id)).str());
      tables.getSymbolTable(source->getParentOp()).insert(clone);
      AttrTypeReplacer self;
      self.addReplacement([&](StructType t) -> std::optional<Type> {
        if (t == source.getType() || t == type) {
          return StructType::get(getFullyQualifiedName(clone));
        }
        return std::nullopt;
      });
      self.recursivelyReplaceElementsIn(clone, true, false, true);
    }
    publish(*id, clone);
    return StructType::get(getFullyQualifiedName(clone));
  }

  LogicalResult specializeCall(CallOp call) {
    auto target = call.getCalleeTarget(tables);
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
          value =
              Step2_InstantiateFunctions::inferUnifiedParam(*unified, name).value_or(Attribute());
        }
        ++n;
        if (!value || !isConcreteAttr<false>(value)) {
          return call.emitError("unsupported non-concrete function parameter ") << name;
        }
        if (failed(call.verifyTemplateParamValueCompatibility(value, param))) {
          return failure();
        }
        args.push_back(value);
        bindings[name] = value;
      }
    }
    auto id = reserve(source, ArrayAttr::get(root.getContext(), args));
    if (failed(id)) {
      return failure();
    }
    if (!entries[*id].definition) {
      auto clone = source.clone();
      clone.setSymName(("__llzk_spec_" + Twine(*id)).str());
      Operation *parent = templ ? templ->getParentOp() : source->getParentOp();
      tables.getSymbolTable(parent).insert(clone);
      if (templ) {
        evaluateTemplateExprs(templ, bindings);
        convertCalleesInPlace(clone, bindings);
        substituteCallArguments(clone, bindings);
        if (failed(Step2_InstantiateFunctions::applyBodyConversions(call, clone, bindings))) {
          return failure();
        }
      }
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
      types.recursivelyReplaceElementsIn(definition, true, false, true);
      definition->setAttr("poly.origin", origin);
      definition->setAttr("poly.arguments", args);
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
        auto target = call.getCalleeTarget(tables);
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
