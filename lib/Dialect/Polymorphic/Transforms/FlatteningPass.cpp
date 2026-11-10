//===-- LLZKFlatteningPass.cpp - Implements -llzk-flatten pass --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-flatten` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/SymbolDefTree.h"
#include "llzk/Analysis/SymbolUseGraph.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/Concepts.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/SymbolTableLLZK.h"
#include "llzk/Util/TypeHelper.h"
#include "llzk/Util/Walk.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdint>

// Include the generated base pass class definitions.
namespace llzk::polymorphic {
#define GEN_PASS_DEF_FLATTENINGPASS
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h.inc"
} // namespace llzk::polymorphic

#include "FlatteningContext.h"
#include "FlatteningScalarization.h"
#include "FlatteningTypeRefinement.h"
#include "SharedImpl.h"
#include "TemplateInstantiation.h"

#define DEBUG_TYPE "llzk-flatten"

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::constrain;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::polymorphic;
using namespace llzk::polymorphic::detail;
using namespace llzk::polymorphic::detail::flattening;

namespace {

/// Patterns can use this listener and call notifyMatchFailure(..) for failures where the entire
/// pass must fail, i.e., where instantiation would introduce an illegal type conversion.
struct MatchFailureListener : public RewriterBase::Listener {
  bool hadFailure = false;

  /// Destroy the listener through the MLIR listener base class.
  ~MatchFailureListener() override = default;

  /// Convert match failures into reported diagnostics and remember that the pass must fail.
  void notifyMatchFailure(Location loc, function_ref<void(Diagnostic &)> reasonCallback) override {
    hadFailure = true;

    InFlightDiagnostic diag = emitError(loc);
    reasonCallback(*diag.getUnderlyingDiagnostic());
    diag.report();
  }
};

/// Reach a local fixpoint while protecting skipped loops from folding and DCE.
/// Filtered sweeps reconsider new operations and symbol dependencies within this step, with
/// loop preservation recomputed before admitting new operations to the greedy worklist.
static LogicalResult
applyAndFoldGreedily(ModuleOp modOp, ConversionTracker &tracker, RewritePatternSet &&patterns) {
  bool currStepModified = false;
  MatchFailureListener failureListener;
  auto greedyRewriteConfig =
      GreedyRewriteConfig().enableFolding().setMaxIterations(20).setListener(&failureListener);
  DenseSet<Operation *> preserved = tracker.getPreservedLoopOps(modOp);
  LogicalResult result = success();
  if (preserved.empty()) {
    result = applyPatternsGreedily(
        modOp->getRegion(0), std::move(patterns), greedyRewriteConfig, &currStepModified
    );
  } else {
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    // Strictness prevents operand/user updates from re-enqueuing a preserved loop or
    // admitting a new loop before its bounds have been checked in the next local sweep.
    greedyRewriteConfig.setStrictness(GreedyRewriteStrictness::ExistingOps);
    for (int64_t iteration = 0;; ++iteration) {
      // The operation-list driver does not enforce the region driver's iteration limit.
      if (iteration == greedyRewriteConfig.getMaxIterations()) {
        result = modOp.emitError("greedy rewriting with preserved loops failed to converge after ")
                 << iteration << " iterations";
        break;
      }
      if (iteration != 0) {
        preserved = tracker.getPreservedLoopOps(modOp);
      }
      // The greedy worklist is LIFO: visit loop bodies before helpers that replace their loops.
      auto candidates = walkCollect<Operation *, WalkOrder::PreOrder>(
          modOp.getBodyRegion(), [&preserved](Operation *op) { return !preserved.contains(op); }
      );
      bool sweepModified = false;
      result =
          applyOpPatternsGreedily(candidates, frozenPatterns, greedyRewriteConfig, &sweepModified);
      currStepModified |= sweepModified;
      if (failed(result) || failureListener.hadFailure || !sweepModified) {
        break;
      }
    }
  }
  tracker.updateModifiedFlag(currStepModified);
  return failure(result.failed() || failureListener.hadFailure);
}

/// Clone a function and substitute concrete template callees throughout its copied body.
static FuncDefOp cloneFunctionWithConvertedCallees(
    FuncDefOp function, const DenseMap<Attribute, Attribute> &paramNameToConcrete
) {
  FuncDefOp clone = function.clone();
  convertCalleesInPlace(clone, paramNameToConcrete);
  return clone;
}

/// Create the initially empty shell of a partially-instantiated template.
///
/// The caller owns insertion and populating the preserved template symbols, because those steps
/// vary between struct and function instantiation.
static TemplateOp
createPartialInstantiationTemplate(TemplateOp sourceTemplate, const InstantiationLayout &layout) {
  TemplateOp newTemplate = sourceTemplate.cloneWithoutRegions();
  newTemplate.setSymName(layout.templateNameWithAttrs);
  setInstantiationNamePattern(newTemplate, layout.namePattern);
  assert(newTemplate->getNumRegions() > 0 && "template must have a body region");
  newTemplate.getBodyRegion().emplaceBlock();
  return newTemplate;
}

/// Return true iff `op` calls a single-nested symbol rooted at a parameter of its parent template.
static bool calleeReferencesTemplateParam(CallOp op) {
  SymbolRefAttr callee = op.getCalleeAttr();
  if (!callee || callee.getNestedReferences().size() != 1) {
    return false;
  }
  TemplateOp parentTemplate = getParentOfType<TemplateOp>(op);
  if (!parentTemplate) {
    return false;
  }
  return parentTemplate.hasConstNamed<TemplateParamOp>(callee.getRootReference());
}

/// Return whether a member type references a type variable, optionally limited to candidate
/// bindings. Inspect struct parameters and their preimages without mistaking unrelated symbol names
/// for type variables, so unrelated generic writes cannot delay this member's refinement.
static bool referencesMemberTypeVar(
    Type type, const DenseMap<TypeVarType, Type> *bindings, const ConversionTracker &tracker
) {
  return type
      .walk([&](Type nestedType) {
    if (auto variable = dyn_cast<TypeVarType>(nestedType);
        variable && (!bindings || bindings->contains(variable))) {
      return WalkResult::interrupt();
    }
    if (auto structType = dyn_cast<StructType>(nestedType)) {
      if (ArrayAttr params = structType.getParams()) {
        for (Attribute param : params) {
          if (auto name = dyn_cast<FlatSymbolRefAttr>(param);
              name && (!bindings || bindings->contains(TypeVarType::get(name)))) {
            return WalkResult::interrupt();
          }
        }
      }
      if (auto preimage = tracker.getPreimage(structType);
          preimage && referencesMemberTypeVar(*preimage, bindings, tracker)) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  }).wasInterrupted();
}

/// Record a projected type-variable binding only when its symbolic constraints can be retained.
/// Candidate identity bindings remain in place; relevant sibling bindings must resolve before a
/// different member can discard the shared variable from its declaration.
static bool appendMemberTypeVarBinding(
    TypeVarType variable, Type binding, const ConversionTracker &tracker,
    SmallVectorImpl<std::pair<TypeVarType, Type>> &bindings,
    const DenseMap<TypeVarType, Type> *relevantBindings
) {
  if (relevantBindings && !relevantBindings->contains(variable)) {
    return true;
  }
  bool mustResolve = relevantBindings || variable != binding;
  if (mustResolve && referencesMemberTypeVar(binding, nullptr, tracker)) {
    return false;
  }
  bindings.emplace_back(variable, binding);
  return true;
}

/// Collect type-variable bindings at matching positions within member and written types.
/// Aggregate shapes belong to their members, while each nested type variable denotes a shared
/// template binding. Keep repeated occurrences so compatible partial bindings can be combined.
/// When checking a sibling, skip subtrees unrelated to the candidate bindings.
static bool collectMemberTypeVarBindings(
    Type declaredType, Type writtenType, const ConversionTracker &tracker,
    SmallVectorImpl<std::pair<TypeVarType, Type>> &bindings,
    const DenseMap<TypeVarType, Type> *relevantBindings = nullptr
) {
  if (!referencesMemberTypeVar(declaredType, relevantBindings, tracker)) {
    return true;
  }
  if (auto variable = dyn_cast<TypeVarType>(declaredType)) {
    return appendMemberTypeVarBinding(variable, writtenType, tracker, bindings, relevantBindings);
  }
  if (auto declaredArray = dyn_cast<ArrayType>(declaredType)) {
    auto writtenArray = dyn_cast<ArrayType>(writtenType);
    if (!writtenArray ||
        writtenArray.getDimensionSizes().size() < declaredArray.getDimensionSizes().size()) {
      return false;
    }
    // Array-valued bindings flatten into the enclosing array's trailing dimensions.
    return collectMemberTypeVarBindings(
        declaredArray.getElementType(),
        writtenArray.getSelectionType(declaredArray.getDimensionSizes().size()), tracker, bindings,
        relevantBindings
    );
  }
  if (auto declaredPod = dyn_cast<pod::PodType>(declaredType)) {
    auto writtenPod = dyn_cast<pod::PodType>(writtenType);
    if (!writtenPod || declaredPod.getRecords().size() != writtenPod.getRecords().size()) {
      return false;
    }
    for (auto [declaredRecord, writtenRecord] :
         llvm::zip_equal(declaredPod.getRecords(), writtenPod.getRecords())) {
      if (declaredRecord.getName() != writtenRecord.getName() ||
          !collectMemberTypeVarBindings(
              declaredRecord.getType(), writtenRecord.getType(), tracker, bindings, relevantBindings
          )) {
        return false;
      }
    }
    return true;
  }
  if (auto declaredStruct = dyn_cast<StructType>(declaredType)) {
    auto writtenStruct = dyn_cast<StructType>(writtenType);
    if (!writtenStruct) {
      return false;
    }
    // Concrete and partially instantiated structs hide their original parameter lists.
    while (auto preimage = tracker.getPreimage(declaredStruct)) {
      declaredStruct = *preimage;
    }
    while (auto preimage = tracker.getPreimage(writtenStruct)) {
      writtenStruct = *preimage;
    }
    if (declaredStruct.getNameRef() != writtenStruct.getNameRef()) {
      return false;
    }
    ArrayAttr declaredParams = declaredStruct.getParams();
    ArrayAttr writtenParams = writtenStruct.getParams();
    if (!declaredParams || !writtenParams) {
      return !declaredParams && !writtenParams;
    }
    if (declaredParams.size() != writtenParams.size()) {
      return false;
    }
    for (auto [declaredParam, writtenParam] : llvm::zip_equal(declaredParams, writtenParams)) {
      auto writtenTypeParam = dyn_cast<TypeAttr>(writtenParam);
      if (auto name = dyn_cast<FlatSymbolRefAttr>(writtenParam)) {
        // Bare symbols and type-valued symbols retain the same binding constraints.
        writtenTypeParam = TypeAttr::get(TypeVarType::get(name));
      }
      if (!writtenTypeParam) {
        continue;
      }
      if (auto declaredTypeParam = dyn_cast<TypeAttr>(declaredParam)) {
        if (!collectMemberTypeVarBindings(
                declaredTypeParam.getValue(), writtenTypeParam.getValue(), tracker, bindings,
                relevantBindings
            )) {
          return false;
        }
      } else if (auto name = dyn_cast<FlatSymbolRefAttr>(declaredParam)) {
        // Project symbolic parameters through the same binding checks as type-valued ones.
        if (!appendMemberTypeVarBinding(
                TypeVarType::get(name), writtenTypeParam.getValue(), tracker, bindings,
                relevantBindings
            )) {
          return false;
        }
      }
    }
  }
  return true;
}

/// Return whether a refinement would separate an unchanged variable from another occurrence.
/// An identity for U does not prevent refining T, but T -> T and T -> felt must stay correlated.
static bool refinesRepeatedMemberTypeVar(ArrayRef<std::pair<TypeVarType, Type>> bindings) {
  DenseSet<TypeVarType> retained;
  for (auto [variable, binding] : bindings) {
    if (variable == binding) {
      retained.insert(variable);
    }
  }
  return llvm::any_of(bindings, [&](auto binding) {
    return binding.first != binding.second && retained.contains(binding.first);
  });
}

/// Reject incompatible requirements connected by shared witnesses, members, and external calls.
/// This only checks unification: a compatible type need not have a representable common refinement.
/// No conversion or IR mutation occurs while collecting and checking the component.
class NonDetMemberTypeVerifier {
  ModuleOp module;
  const ConversionTracker &tracker;
  SymbolTableCollection tables;
  DenseSet<Operation *> visited;
  DenseSet<Operation *> indexedStructs;
  DenseMap<Operation *, SmallVector<MemberWriteOp>> writesByMember;
  DenseMap<Operation *, SymbolRefAttr> comparisonNames;
  DenseMap<std::pair<Type, Operation *>, Type> normalizedTypes;
  SmallVector<SymbolLookupResult<StructDefOp>> resolvedDefinitions;

  /// Normalize instantiated families and namespace spellings for comparison, preserving parameters.
  Type normalize(Type type, Operation *scope) {
    auto key = std::make_pair(type, scope);
    if (auto cached = normalizedTypes.lookup(key)) {
      return cached;
    }
    Type normalized = type;
    if (auto structure = dyn_cast<StructType>(type)) {
      if (auto preimage = tracker.getPreimage(structure)) {
        normalized = normalize(*preimage, module);
      } else {
        auto definition = structure.getDefinition(tables, scope);
        if (failed(definition)) {
          return {};
        }
        Operation *family = definition->get();
        auto [name, inserted] = comparisonNames.try_emplace(family);
        if (inserted) {
          // These private names are used only by typesUnify, never published in IR or diagnostics.
          name->second = FlatSymbolRefAttr::get(
              type.getContext(), "__nondet_family_" + std::to_string(comparisonNames.size())
          );
          resolvedDefinitions.push_back(std::move(*definition));
        }
        SymbolRefAttr comparisonName = name->second;
        SmallVector<Attribute> params;
        if (structure.getParams()) {
          for (Attribute param : structure.getParams()) {
            if (auto typed = dyn_cast<TypeAttr>(param)) {
              Type nested = normalize(typed.getValue(), scope);
              if (!nested) {
                return {};
              }
              param = TypeAttr::get(nested);
            }
            params.push_back(param);
          }
        }
        normalized =
            params.empty()
                ? StructType::get(comparisonName)
                : StructType::get(comparisonName, ArrayAttr::get(type.getContext(), params));
      }
    } else if (auto array = dyn_cast<ArrayType>(type)) {
      Type element = normalize(array.getElementType(), scope);
      if (!element) {
        return {};
      }
      normalized = array.cloneWith(element);
    } else if (auto podType = dyn_cast<pod::PodType>(type)) {
      SmallVector<pod::RecordAttr> records;
      for (auto record : podType.getRecords()) {
        Type nested = normalize(record.getType(), scope);
        if (!nested) {
          return {};
        }
        records.push_back(pod::RecordAttr::get(type.getContext(), record.getName(), nested));
      }
      normalized = pod::PodType::get(type.getContext(), records);
    }
    if (normalized) {
      normalizedTypes.try_emplace(key, normalized);
    }
    return normalized;
  }

  /// Index only reached struct bodies; unrelated modules need not be revisited for one witness.
  LogicalResult indexWriters(MemberDefOp member) {
    auto owner = member->getParentOfType<StructDefOp>();
    if (!indexedStructs.insert(owner).second) {
      return success();
    }
    WalkResult result = owner.walk([&](MemberWriteOp write) {
      auto definition = write.getMemberDefOp(tables);
      if (failed(definition)) {
        return WalkResult::interrupt();
      }
      writesByMember[definition->get()].push_back(write);
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

public:
  /// Construct a read-only checker for the current IR; discard it before changing that IR.
  NonDetMemberTypeVerifier(ModuleOp root, const ConversionTracker &conversions)
      : module(root), tracker(conversions) {}

  /// Check a witness's connected member-write requirements, including a cached replacement.
  /// Resolve an optional cached replacement in its original scope.
  LogicalResult verifyComponent(NonDetOp witness, Value extraValue = {}) {
    // Module checking shares visited nodes; mutation-time callers use a fresh checker.
    if (visited.contains(witness) && !extraValue) {
      return success();
    }
    struct Requirement {
      Type normalized;
      Type original;
      Operation *origin;
      bool external;
    };
    SmallVector<Requirement> requirements;
    DenseSet<Type> seenTypes;
    bool hasExternalRequirement = false;
    auto addRequirement = [&](Type type, Operation *origin,
                              Operation *declaration = nullptr) -> LogicalResult {
      bool external = declaration != nullptr;
      hasExternalRequirement |= external;
      Type comparison = normalize(type, external ? declaration : origin);
      if (!comparison) {
        return failure();
      }
      if (!seenTypes.insert(comparison).second) {
        return success();
      }
      requirements.push_back({comparison, type, origin, external});
      return success();
    };
    if (extraValue && failed(addRequirement(extraValue.getType(), extraValue.getDefiningOp()))) {
      return failure();
    }
    DenseSet<Operation *> componentVisited;
    DenseSet<Operation *> memberWrites;
    SmallVector<Operation *> pending {witness};
    while (!pending.empty()) {
      Operation *node = pending.pop_back_val();
      if (!componentVisited.insert(node).second) {
        continue;
      }
      visited.insert(node);
      if (auto nondet = dyn_cast<NonDetOp>(node)) {
        if (failed(addRequirement(nondet.getType(), nondet))) {
          return failure();
        }
        for (OpOperand &use : nondet.getResult().getUses()) {
          auto write = dyn_cast<MemberWriteOp>(use.getOwner());
          if (!write || write.getVal() != nondet.getResult()) {
            if (auto call = dyn_cast<CallOp>(use.getOwner())) {
              auto callee = call.getCalleeTarget(tables);
              if (failed(callee)) {
                return failure();
              }
              unsigned argIdx =
                  use.getOperandNumber() - call.getArgOperands().getBeginOperandIndex();
              if (argIdx >= call.getArgOperands().size() ||
                  failed(addRequirement(
                      callee->get().getFunctionType().getInput(argIdx), call, callee->get()
                  ))) {
                return failure();
              }
            }
            continue;
          }
          auto member = write.getMemberDefOp(tables);
          if (failed(member)) {
            return failure();
          }
          pending.push_back(member->get());
        }
        continue;
      }
      auto member = llvm::cast<MemberDefOp>(node);
      if (failed(addRequirement(member.getType(), member)) || failed(indexWriters(member))) {
        return failure();
      }
      for (MemberWriteOp write : writesByMember[member]) {
        memberWrites.insert(write);
        if (failed(addRequirement(write.getVal().getType(), write))) {
          return failure();
        }
        if (auto other = write.getVal().getDefiningOp<NonDetOp>()) {
          pending.push_back(other);
        }
      }
    }
    // A single write also needs coordination when an external call observes the witness.
    // Components without member writes are not refined here.
    if (!extraValue &&
        (memberWrites.empty() || (memberWrites.size() < 2 && !hasExternalRequirement))) {
      return success();
    }
    for (size_t i = 0; i < requirements.size(); ++i) {
      const Requirement &current = requirements[i];
      for (const Requirement &previous : ArrayRef(requirements).take_front(i)) {
        // Unification is not transitive: a generic type may unify with two conflicting types.
        if (!typesUnify(previous.normalized, current.normalized)) {
          auto diagnostic = witness.emitError(
              previous.external || current.external
                  ? "cannot refine nondeterministic witness because an external call requires an "
                    "incompatible type"
                  : "cannot refine shared nondeterministic witness because member writes require "
                    "incompatible types"
          );
          diagnostic.attachNote(previous.origin->getLoc())
              << "first required type: " << previous.original;
          diagnostic.attachNote(current.origin->getLoc())
              << "conflicting required type: " << current.original;
          return failure();
        }
      }
    }
    return success();
  }

  /// Check every connected witness component before conversion can mutate types or clone symbols.
  LogicalResult verifyModule() {
    WalkResult result = module.walk([&](NonDetOp witness) {
      return failed(verifyComponent(witness)) ? WalkResult::interrupt() : WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }
};

namespace Step1_InstantiateStructs {

/// Implements cloning a `StructDefOp` for a specific instantiation site, using the concrete
/// parameters from the instantiation to replace parameters from the original `StructDefOp`.
class StructCloner {
  ConversionTracker &tracker_;
  ModuleOp rootMod;
  SymbolTableCollection symTables;
  bool reportMissing = true;

  FailureOr<StructType> genClone(StructType typeAtCaller, ArrayRef<Attribute> typeAtCallerParams) {
    LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   attempting clone of " << typeAtCaller << '\n');
    // Find the StructDefOp for the original StructType
    FailureOr<SymbolLookupResult<StructDefOp>> r =
        typeAtCaller.getDefinition(symTables, rootMod, reportMissing);
    if (failed(r)) {
      LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   skip: cannot find StructDefOp \n");
      return failure(); // getDefinition() already emits a sufficient error message
    }
    LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   found definition\n";);

    StructDefOp origStruct = r->get();
    StructType typeAtDef = origStruct.getType();
    MLIRContext *ctx = origStruct.getContext();
    TemplateOp parentTemplate = getParentOfType<TemplateOp>(origStruct);
    assert(parentTemplate && "parameterized struct must be nested in a TemplateOp");
    ModuleOp parentModule = getParentOfType<ModuleOp>(parentTemplate);
    assert(parentModule && "TemplateOp must be nested in a ModuleOp");

    // Map of StructDefOp parameter name to concrete Attribute at the current instantiation site.
    DenseMap<Attribute, Attribute> paramNameToConcrete;
    // Reduced from `typeAtCallerParams` to contain only the non-concrete Attributes.
    ArrayAttr reducedCallerParams = nullptr;
    SmallVector<Attribute> nonConcreteParams;
    {
      ArrayAttr paramNames = typeAtDef.getParams();

      // pre-conditions
      assert(!isNullOrEmpty(paramNames));
      assert(paramNames.size() == typeAtCallerParams.size());

      for (size_t i = 0, e = paramNames.size(); i < e; ++i) {
        Attribute next = typeAtCallerParams[i];
        if (isConcreteStructParamAttr(next, /*allowStructParams=*/false)) {
          paramNameToConcrete[paramNames[i]] = next;
        } else {
          nonConcreteParams.push_back(next);
        }
      }
      // post-conditions
      assert(nonConcreteParams.size() + paramNameToConcrete.size() == paramNames.size());

      if (paramNameToConcrete.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   skip: no concrete params \n");
        return failure();
      }
      if (!nonConcreteParams.empty()) {
        reducedCallerParams = ArrayAttr::get(ctx, nonConcreteParams);
      }
    }

    FailureOr<InstantiationLayout> layoutResult =
        buildInstantiationLayout(parentTemplate, ArrayAttr(), paramNameToConcrete);
    if (failed(layoutResult)) {
      return failure();
    }
    InstantiationLayout layout = std::move(*layoutResult);
    assert(layout.remainingNames.size() == nonConcreteParams.size());

    // This list will be used to build the new remote/external type.
    SmallVector<FlatSymbolRefAttr> typeAtCallerSymPieces = getPieces(typeAtCaller.getNameRef());
    typeAtCallerSymPieces.pop_back(); // drop struct name

    // Evaluate any poly.expr symbols whose param dependencies are now concrete; add them to the
    // map so ClonedBodyConstReadOpPattern can replace uses of those symbols too.
    evaluateTemplateExprs(parentTemplate, paramNameToConcrete);

    // Clone the original struct.
    StructDefOp newStruct = origStruct.clone();
    convertCalleesInPlace(newStruct, paramNameToConcrete);
    Operation *insertedCloneRoot = nullptr;
    if (layout.remainingNames.empty()) { // FULL INSTANTIATION CASE
      // Set name of the new struct by prepending its name with instantiated template name.
      newStruct.setSymName(
          (layout.templateNameWithAttrs + mlir::Twine('_') + newStruct.getSymName()).str()
      );
      // Insert 'newStruct' into the parent ModuleOp of the original TemplateOp. Use the
      // `SymbolTable::insert()` function so that the name will be made unique if necessary.
      symTables.getSymbolTable(parentModule).insert(newStruct, Block::iterator(parentTemplate));
      insertedCloneRoot = newStruct.getOperation();
      // Drop the old template name from the list.
      typeAtCallerSymPieces.pop_back();
    } else { // PARTIAL INSTANTIATION CASE
      // Clone the template and set instantiated name.
      TemplateOp newTemplate = createPartialInstantiationTemplate(parentTemplate, layout);

      // Clone preserved const param/expr ops.
      for (Attribute name : layout.remainingNames) {
        FlatSymbolRefAttr nameSym = llvm::dyn_cast<FlatSymbolRefAttr>(name);
        assert(nameSym && "expected FlatSymbolRefAttr");

        Operation *symOp = symTables.getSymbolTable(parentTemplate).lookup(nameSym.getAttr());
        assert(symOp && "symbol must exist");
        newTemplate.insert(newTemplate.begin(), symOp->clone());
      }

      // Insert the struct into the template and the template into the module. Use the
      // `SymbolTable::insert()` function so that the name will be made unique if necessary.
      symTables.getSymbolTable(newTemplate).insert(newStruct);
      symTables.getSymbolTable(parentModule).insert(newTemplate, Block::iterator(parentTemplate));
      insertedCloneRoot = newTemplate.getOperation();

      // Replace the old template name in the list with the new one (get template name after
      // symbol table insertion since it may be modified to make it unique).
      typeAtCallerSymPieces.back() = FlatSymbolRefAttr::get(newTemplate.getSymNameAttr());
    }

    // Retrieve the new type AFTER inserting since the struct name may be appended to make
    // it unique and use the remaining non-concrete parameters from the original type.
    StructType newLocalType = newStruct.getType(reducedCallerParams);
    typeAtCallerSymPieces.push_back(
        FlatSymbolRefAttr::get(newLocalType.getNameRef().getLeafReference())
    );
    StructType newRemoteType =
        StructType::get(asSymbolRefAttr(typeAtCallerSymPieces), newLocalType.getParams());
    LLVM_DEBUG({
      llvm::dbgs() << "[StructCloner]   original def type: " << typeAtDef << '\n';
      llvm::dbgs() << "[StructCloner]   cloned def type: " << newStruct.getType() << '\n';
      llvm::dbgs() << "[StructCloner]   original remote type: " << typeAtCaller << '\n';
      llvm::dbgs() << "[StructCloner]   cloned local type: " << newLocalType << '\n';
      llvm::dbgs() << "[StructCloner]   cloned remote type: " << newRemoteType << '\n';
    });

    // Within the new struct, replace all references to the original StructType (i.e., the
    // locally-parameterized version) with the new locally-parameterized StructType,
    // and replace all uses of the removed struct parameters with the concrete values.
    if (failed(substituteStructBody(
            newStruct, typeAtDef, paramNameToConcrete, tracker_.delayedDiagnosticSet(newRemoteType)
        ))) {
      LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   instantiating body of struct failed \n");
      assert(insertedCloneRoot && "clone root must have been inserted before body conversion");
      insertedCloneRoot->erase();
      return failure();
    }
    return newRemoteType;
  }

public:
  /// Construct a cloner rooted at `root` and reporting modifications through `tracker`.
  StructCloner(ConversionTracker &tracker, ModuleOp root)
      : tracker_(tracker), rootMod(root), symTables() {}

  /// Create a full or partial instantiated clone for `orig`, if `orig` has concrete parameters.
  FailureOr<StructType> createInstantiatedClone(StructType orig) {
    LLVM_DEBUG(llvm::dbgs() << "[StructCloner] orig: " << orig << '\n');
    if (ArrayAttr params = orig.getParams()) {
      return genClone(orig, params.getValue());
    }
    LLVM_DEBUG(llvm::dbgs() << "[StructCloner]   skip: nullptr for params \n");
    return failure();
  }

  /// Re-enable diagnostics when a referenced struct definition cannot be found.
  void enableReportMissing() { reportMissing = true; }

  /// Temporarily suppress missing-symbol diagnostics during speculative legality checks.
  void disableReportMissing() { reportMissing = false; }
};

class DisableReportMissing;

class ParameterizedStructUseTypeConverter : public TypeConverter {
  ConversionTracker &tracker_;
  StructCloner cloner;

  friend DisableReportMissing;

public:
  /// Build a type converter that instantiates parameterized struct uses on demand.
  ParameterizedStructUseTypeConverter(ConversionTracker &tracker, ModuleOp root)
      : tracker_(tracker), cloner(tracker, root) {

    addConversion([](Type inputTy) { return inputTy; });

    addConversion([this](StructType inputTy) -> StructType {
      LLVM_DEBUG(
          llvm::dbgs() << "[ParameterizedStructUseTypeConverter] attempting conversion of "
                       << inputTy << '\n';
      );
      // First check for a cached entry
      if (auto opt = tracker_.getInstantiation(inputTy)) {
        return opt.value();
      }

      // Otherwise, try to create a clone of the struct with instantiated params. If that can't be
      // done, return the original type to indicate that it's still legal (for this step at least).
      FailureOr<StructType> cloneRes = cloner.createInstantiatedClone(inputTy);
      if (failed(cloneRes)) {
        return inputTy;
      }
      StructType newTy = cloneRes.value();
      LLVM_DEBUG(
          llvm::dbgs() << "[ParameterizedStructUseTypeConverter] instantiating " << inputTy
                       << " as " << newTy << '\n'
      );
      tracker_.recordInstantiation(inputTy, newTy);
      return newTy;
    });

    addConversion([this](ArrayType inputTy) {
      return inputTy.cloneWith(convertType(inputTy.getElementType()));
    });

    addConversion([this](pod::PodType inputTy) -> pod::PodType {
      SmallVector<pod::RecordAttr> convertedRecords;
      for (pod::RecordAttr record : inputTy.getRecords()) {
        Type convertedType = convertType(record.getType());
        if (!convertedType) {
          return {};
        }
        convertedRecords.push_back(
            pod::RecordAttr::get(inputTy.getContext(), record.getName(), convertedType)
        );
      }
      return pod::PodType::get(inputTy.getContext(), convertedRecords);
    });
  }
};

/// Rewrite calls to struct `compute`/`constrain` functions after their struct types have been
/// instantiated.
class CallStructFuncPattern : public OpConversionPattern<CallOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the call rewrite pattern using the active type converter and tracker.
  CallStructFuncPattern(TypeConverter &converter, MLIRContext *ctx, ConversionTracker &tracker)
      // benefit>0 so this applies instead of CallOpClassReplacePattern
      : OpConversionPattern<CallOp>(converter, ctx, /*benefit=*/1), tracker_(tracker) {}

  /// Replace a call with converted result types and, when needed, a callee rooted at the
  /// instantiated struct type.
  LogicalResult matchAndRewrite(
      CallOp op, OpAdaptor adapter, ConversionPatternRewriter &rewriter
  ) const override {
    LLVM_DEBUG(llvm::dbgs() << "[CallStructFuncPattern] CallOp: " << op << '\n');

    // Convert the result types of the CallOp
    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), newResultTypes))) {
      return op->emitError("Could not convert Op result types.");
    }
    LLVM_DEBUG({
      llvm::dbgs() << "[CallStructFuncPattern]   newResultTypes: "
                   << debug::toStringList(newResultTypes) << '\n';
    });

    // Update the callee to reflect the new struct target if necessary. These checks are based on
    // `CallOp::calleeIsStructC*()` but the types must not come from the CallOp in this case.
    // Instead they must come from the converted versions.
    SymbolRefAttr calleeAttr = op.getCalleeAttr();
    if (op.calleeIsStructCompute()) {
      if (StructType newStTy = getIfSingleton<StructType>(newResultTypes)) {
        LLVM_DEBUG(llvm::dbgs() << "[CallStructFuncPattern]   newStTy: " << newStTy << '\n');
        calleeAttr = appendLeaf(newStTy.getNameRef(), calleeAttr.getLeafReference());
        tracker_.reportDelayedDiagnostics(newStTy, op);
      }
    } else if (op.calleeIsStructConstrain()) {
      if (StructType newStTy = getAtIndex<StructType>(adapter.getArgOperands().getTypes(), 0)) {
        LLVM_DEBUG(llvm::dbgs() << "[CallStructFuncPattern]   newStTy: " << newStTy << '\n');
        calleeAttr = appendLeaf(newStTy.getNameRef(), calleeAttr.getLeafReference());
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "[CallStructFuncPattern] replaced " << op);
    CallOp newOp = replaceOpWithNewOp<CallOp>(
        rewriter, op, newResultTypes, calleeAttr, adapter.getMapOperands(),
        op.getNumDimsPerMapAttr(), adapter.getArgOperands()
    );
    (void)newOp; // tell compiler it's intentionally unused in release builds
    LLVM_DEBUG(llvm::dbgs() << " with " << newOp << '\n');
    return success();
  }
};

/// Ensure `struct.member` types are converted even if no read/write pattern visits them.
class MemberDefOpPattern : public OpConversionPattern<MemberDefOp> {
public:
  /// Construct the member definition conversion pattern.
  MemberDefOpPattern(TypeConverter &converter, MLIRContext *ctx, ConversionTracker &)
      // benefit>0 so this applies instead of GeneralTypeReplacePattern<MemberDefOp>
      : OpConversionPattern<MemberDefOp>(converter, ctx, /*benefit=*/1) {}

  /// Update the member definition type when the active type converter changes it.
  LogicalResult matchAndRewrite(
      MemberDefOp op, OpAdaptor /*adapter*/, ConversionPatternRewriter &rewriter
  ) const override {
    LLVM_DEBUG(llvm::dbgs() << "[MemberDefOpPattern] MemberDefOp: " << op << '\n');

    Type oldMemberType = op.getType();
    Type newMemberType = getTypeConverter()->convertType(oldMemberType);
    if (oldMemberType == newMemberType) {
      return failure(); // nothing changed
    }
    rewriter.modifyOpInPlace(op, [&op, &newMemberType]() { op.setType(newMemberType); });
    return success();
  }
};

/// Return a fully concrete struct type when `lhs` and `rhs` supply complementary integer bindings.
///
/// This deliberately only handles the direct parameter refinement needed while instantiating
/// struct uses. More general refinement is performed by the propagation steps below.
static std::optional<StructType>
getComplementaryStructRefinement(StructType lhs, StructType rhs, const ConversionTracker &tracker) {
  lhs = tracker.getPreimage(lhs).value_or(lhs);
  rhs = tracker.getPreimage(rhs).value_or(rhs);
  if (lhs.getNameRef() != rhs.getNameRef()) {
    return std::nullopt;
  }

  ArrayAttr lhsParams = lhs.getParams();
  ArrayAttr rhsParams = rhs.getParams();
  if (!lhsParams || !rhsParams || lhsParams.size() != rhsParams.size()) {
    return std::nullopt;
  }

  SmallVector<Attribute> refinedParams;
  for (auto [lhsParam, rhsParam] : llvm::zip_equal(lhsParams, rhsParams)) {
    if (lhsParam == rhsParam) {
      refinedParams.push_back(lhsParam);
      continue;
    }
    auto lhsInt = llvm::dyn_cast<IntegerAttr>(lhsParam);
    auto rhsInt = llvm::dyn_cast<IntegerAttr>(rhsParam);
    if (lhsInt && !isDynamic(lhsInt) && llvm::isa<AffineMapAttr>(rhsParam)) {
      refinedParams.push_back(lhsParam);
      continue;
    }
    if (rhsInt && !isDynamic(rhsInt) && llvm::isa<AffineMapAttr>(lhsParam)) {
      refinedParams.push_back(rhsParam);
      continue;
    }
    return std::nullopt;
  }
  return StructType::get(lhs.getNameRef(), ArrayAttr::get(lhs.getContext(), refinedParams));
}

/// Convert nondeterministic result types when a shared witness is refined to an instantiated type.
class NonDetOpPattern : public OpConversionPattern<NonDetOp> {
  ConversionTracker *tracker_;
  ModuleOp rootModule_;
  bool &hadConflict_;

public:
  /// Construct the nondeterministic value conversion pattern.
  NonDetOpPattern(
      TypeConverter &converter, MLIRContext *ctx, ConversionTracker *tracker, ModuleOp rootModule,
      bool &hadConflict
  )
      : OpConversionPattern<NonDetOp>(converter, ctx, /*benefit=*/1), tracker_(tracker),
        rootModule_(rootModule), hadConflict_(hadConflict) {}

  /// Rebuild the op with a type refined jointly with directly written member types.
  LogicalResult
  matchAndRewrite(NonDetOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    if (hadConflict_) {
      return failure();
    }
    if (tracker_ && failed(NonDetMemberTypeVerifier(rootModule_, *tracker_).verifyComponent(op))) {
      hadConflict_ = true;
      return failure();
    }
    Type replacementInputType = op.getType();
    SmallVector<MemberDefOp> refinedMemberDefs;
    bool canRefineAllMembers = true;
    if (tracker_) {
      SymbolTableCollection tables;
      for (OpOperand &use : op.getResult().getUses()) {
        auto memberWrite = llvm::dyn_cast<MemberWriteOp>(use.getOwner());
        if (!memberWrite || memberWrite.getVal() != op.getResult()) {
          continue;
        }
        FailureOr<SymbolLookupResult<MemberDefOp>> memberDef = memberWrite.getMemberDefOp(tables);
        if (failed(memberDef)) {
          return failure();
        }
        auto witnessType = llvm::dyn_cast<StructType>(replacementInputType);
        auto memberType = llvm::dyn_cast<StructType>(memberDef->get().getType());
        if (!witnessType || !memberType) {
          continue;
        }
        std::optional<StructType> commonType =
            getComplementaryStructRefinement(witnessType, memberType, *tracker_);
        if (!commonType) {
          // A compatible but unsupported join must not commit a successful prefix of the uses.
          // Conflicting requirements are rejected by the shared-witness preflight.
          canRefineAllMembers = false;
          continue;
        }
        replacementInputType = *commonType;
        refinedMemberDefs.push_back(memberDef->get());
      }
    }
    if (!canRefineAllMembers) {
      replacementInputType = op.getType();
      refinedMemberDefs.clear();
    }

    // Ordinary type conversion updates all users consistently. The additional member-driven
    // refinement below only repairs member writes; it cannot leave other users observing an
    // obsolete conversion cast or silently split their witness from the member writes.
    if (replacementInputType != op.getType()) {
      for (OpOperand &use : op.getResult().getUses()) {
        auto write = dyn_cast<MemberWriteOp>(use.getOwner());
        if (!write || write.getVal() != op.getResult()) {
          auto diagnostic = op.emitError(
              "cannot refine nondeterministic witness from member writes because it has an "
              "unsupported external use"
          );
          diagnostic.attachNote(use.getOwner()->getLoc()) << "source witness is also used here";
          hadConflict_ = true;
          return failure();
        }
      }
    }

    Type newType = getTypeConverter()->convertType(replacementInputType);
    if (!newType) {
      return op->emitError("Could not convert Op result type.");
    }
    bool witnessChanged = newType != op.getType();
    bool membersChanged = llvm::any_of(refinedMemberDefs, [newType](MemberDefOp member) {
      return member.getType() != newType;
    });
    if (!witnessChanged && !membersChanged) {
      return failure();
    }
    for (MemberDefOp memberDef : refinedMemberDefs) {
      if (memberDef.getType() != newType) {
        rewriter.modifyOpInPlace(memberDef, [&memberDef, &newType]() {
          memberDef.setType(newType);
        });
      }
    }
    if (witnessChanged) {
      replaceOpWithNewOp<NonDetOp>(rewriter, op, newType);
    }
    return success();
  }
};

/// Bypass a stale conversion cast when a shared nondeterministic value and its member were both
/// refined to the same instantiated struct type.
class MemberWriteRefinedNonDetPattern : public OpConversionPattern<MemberWriteOp> {
public:
  /// Construct the member-write conversion pattern.
  MemberWriteRefinedNonDetPattern(TypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<MemberWriteOp>(converter, ctx, /*benefit=*/2) {}

  /// Rebuild the write using the jointly refined witness instead of its pre-refinement cast.
  LogicalResult matchAndRewrite(
      MemberWriteOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto cast = adaptor.getVal().getDefiningOp<UnrealizedConversionCastOp>();
    if (!cast || cast.getInputs().size() != 1) {
      return failure();
    }
    Value refinedWitness = cast.getInputs().front();
    if (!refinedWitness.getDefiningOp<NonDetOp>()) {
      return failure();
    }

    SymbolTableCollection tables;
    FailureOr<SymbolLookupResult<MemberDefOp>> memberDef = op.getMemberDefOp(tables);
    if (failed(memberDef) || memberDef->get().getType() != refinedWitness.getType()) {
      return failure();
    }
    replaceOpWithNewOp<MemberWriteOp>(
        rewriter, op, adaptor.getComponent(), op.getMemberNameAttr(), refinedWitness
    );
    return success();
  }
};

/// Disables reporting of missing struct symbols during legality checks to avoid showing error
/// diagnostics that are not actually errors.
class DisableReportMissing : public LegalityCheckCallback {
  ParameterizedStructUseTypeConverter &tyConv;

public:
  /// Tie the callback to the converter whose cloner should suppress lookup diagnostics.
  explicit DisableReportMissing(ParameterizedStructUseTypeConverter &tc) : tyConv(tc) {}

  /// Suppress missing-symbol diagnostics before a speculative legality check begins.
  void checkStarted() override { tyConv.cloner.disableReportMissing(); }

  /// Re-enable missing-symbol diagnostics after the speculative legality check finishes.
  void checkEnded(bool) override { tyConv.cloner.enableReportMissing(); }
};

/// Run struct instantiation and call/member rewrites for the current module.
LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  if (failed(NonDetMemberTypeVerifier(modOp, tracker).verifyModule())) {
    return failure();
  }
  bool hadConflict = false;
  ParameterizedStructUseTypeConverter tyConv(tracker, modOp);
  DisableReportMissing drm(tyConv);
  ConversionTarget target = newConverterDefinedTargetWithCallback<>(tyConv, ctx, drm);
  target.addDynamicallyLegalOp<NonDetOp>([&tyConv](Operation *op) {
    return defaultLegalityCheck(tyConv, op);
  });
  RewritePatternSet patterns = newGeneralRewritePatternSet(tyConv, ctx, target);
  patterns.add<CallStructFuncPattern, MemberDefOpPattern>(tyConv, ctx, tracker);
  patterns.add<NonDetOpPattern>(tyConv, ctx, &tracker, modOp, hadConflict);
  patterns.add<MemberWriteRefinedNonDetPattern>(tyConv, ctx);
  LogicalResult result = applyPartialConversion(modOp, target, std::move(patterns));
  return failure(failed(result) || hadConflict);
}

/// Create an instantiated clone of the struct type specified by the `MAIN_ATTR_NAME` attribute
/// on `modOp`, if it exists, and update the attribute to refer to the new instantiated type.
LogicalResult instantiateMainStruct(ModuleOp modOp, ConversionTracker &tracker) {
  FailureOr<StructType> mainTypeOpt = getMainInstanceType(modOp);
  if (failed(mainTypeOpt)) {
    return failure();
  }

  StructType mainType = mainTypeOpt.value();
  if (!mainType || isNullOrEmpty(mainType.getParams()) || tracker.getInstantiation(mainType)) {
    return success();
  }

  StructCloner cloner(tracker, modOp);
  FailureOr<StructType> cloneRes = cloner.createInstantiatedClone(mainType);
  if (failed(cloneRes)) {
    return failure();
  }

  StructType instantiatedMainType = cloneRes.value();
  tracker.recordInstantiation(mainType, instantiatedMainType);
  modOp->setAttr(MAIN_ATTR_NAME, TypeAttr::get(instantiatedMainType));
  return success();
}

} // namespace Step1_InstantiateStructs

namespace Step2_InstantiateFunctions {

/// Emit the match failure used when an inferred instantiation violates a template parameter's
/// declared type restriction.
static inline LogicalResult failIncompatibleInferredParam(
    CallOp op, PatternRewriter &rewriter, FlatSymbolRefAttr paramName, TemplateParamOp paramOp
) {
  LLVM_DEBUG(
      llvm::dbgs() << "[InstantiateFuncAtCallOp]  unification for param '" << paramName
                   << "': incompatible with specified param type. MUST FAIL!\n"
  );
  return rewriter.notifyMatchFailure(op, [&paramName, &paramOp](Diagnostic &diag) {
    diag.append("inferred value for parameter '")
        .append(paramName)
        .append("' is incompatible with specified param type")
        .attachNote(paramOp.getLoc())
        .append("template parameter declared here");
  });
}

/// Searches a parameterized callee body for concrete type evidence that resolves a wildcard
/// template parameter, following both local unifiable casts and nested template calls.
class WildcardTypeBodyInferer final {
  SymbolTableCollection &symTables_;
  const DenseMap<Attribute, Attribute> &paramNameToConcrete_;
  SmallVector<std::pair<Operation *, FlatSymbolRefAttr>> activeInferences_;

public:
  /// Construct a body inferer over the current symbol tables and known concrete bindings.
  WildcardTypeBodyInferer(
      SymbolTableCollection &symTables, const DenseMap<Attribute, Attribute> &paramNameToConcrete
  )
      : symTables_(symTables), paramNameToConcrete_(paramNameToConcrete) {}

  /// Search `func` for a concrete value that can resolve `paramName`.
  std::optional<Attribute> infer(FuncDefOp func, FlatSymbolRefAttr paramName) {
    return infer(func, paramName, paramNameToConcrete_);
  }

private:
  /// Search `func` using concrete bindings expressed in that function's template scope.
  std::optional<Attribute> infer(
      FuncDefOp func, FlatSymbolRefAttr paramName,
      const DenseMap<Attribute, Attribute> &paramNameToConcrete
  ) {
    if (llvm::any_of(activeInferences_, [&](const auto &e) {
      return e.first == func.getOperation() && e.second == paramName;
    })) {
      return std::nullopt;
    }
    activeInferences_.emplace_back(func.getOperation(), paramName);

    TemplateTypeConverter tyConv(paramNameToConcrete);
    std::optional<Attribute> inferred;
    bool ambiguous = false;

    // Record a concrete candidate unless it conflicts with an earlier one, in which
    // case the wildcard is treated as ambiguous and left unresolved.
    auto noteCandidate = [&inferred, &ambiguous](Attribute candidate) {
      if (!candidate || !isConcreteStructParamAttr(candidate)) {
        return WalkResult::advance();
      }
      if (!inferred.has_value()) {
        inferred = candidate;
        return WalkResult::advance();
      }
      if (*inferred != candidate) {
        ambiguous = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    };

    WalkResult walkResult = func.walk([&](Operation *bodyOp) {
      if (auto castOp = llvm::dyn_cast<UnifiableCastOp>(bodyOp)) {
        Type inputTy = tyConv.convertType(castOp.getInput().getType());
        Type resultTy = tyConv.convertType(castOp.getResult().getType());
        if (auto inputTvar = llvm::dyn_cast<TypeVarType>(inputTy);
            inputTvar && inputTvar.getNameRef() == paramName && isConcreteType(resultTy)) {
          return noteCandidate(TypeAttr::get(resultTy));
        }
        if (auto resultTvar = llvm::dyn_cast<TypeVarType>(resultTy);
            resultTvar && resultTvar.getNameRef() == paramName && isConcreteType(inputTy)) {
          return noteCandidate(TypeAttr::get(inputTy));
        }

        // A requested wildcard can occur beneath an aggregate type on either side of a local
        // cast. Recursively unify both converted types to retain the concrete binding recorded
        // at that nested position.
        UnificationMap unifications;
        if (!typesUnify(inputTy, resultTy, {}, &unifications)) {
          return WalkResult::advance();
        }
        for (Side side : {Side::LHS, Side::RHS}) {
          auto it = unifications.find({paramName, side});
          if (it != unifications.end()) {
            WalkResult candidateResult = noteCandidate(it->second);
            if (candidateResult.wasInterrupted()) {
              return candidateResult;
            }
          }
        }
        return WalkResult::advance();
      }

      auto nestedCall = llvm::dyn_cast<CallOp>(bodyOp);
      if (!nestedCall) {
        return WalkResult::advance();
      }

      FailureOr<SymbolLookupResult<FuncDefOp>> nestedTgtOpt =
          nestedCall.getCalleeTarget(symTables_);
      if (failed(nestedTgtOpt)) {
        return WalkResult::advance();
      }
      FuncDefOp nestedTgt = nestedTgtOpt->get();
      auto nestedTemplate = llvm::dyn_cast<TemplateOp>(nestedTgt->getParentOp());
      if (!nestedTemplate) {
        return WalkResult::advance();
      }

      // Infer through either side of a nested call: an enclosing type variable can appear
      // in a call result or in a call operand. In both cases, the corresponding nested
      // callee type variable may be concretely bound by the nested call.
      auto inferFromNestedBinding = [&](Type enclosingTy, Type nestedTy) {
        DenseMap<Attribute, Attribute> nestedParamNameToConcrete =
            getNestedConcreteBindings(nestedCall, nestedTgt, nestedTemplate, tyConv);

        // In addition to bindings from the call, type variables in the nested signature can
        // obtain their concrete values from the nested callee body. Add that evidence to the
        // nested scope before converting its full signature type below.
        nestedTy.walk([&](TypeVarType nestedTvar) {
          if (nestedParamNameToConcrete.contains(nestedTvar.getNameRef())) {
            return;
          }
          std::optional<Attribute> nestedCandidate =
              infer(nestedTgt, nestedTvar.getNameRef(), nestedParamNameToConcrete);
          if (nestedCandidate) {
            nestedParamNameToConcrete[nestedTvar.getNameRef()] = *nestedCandidate;
          }
        });
        TemplateTypeConverter nestedTyConv(nestedParamNameToConcrete);
        Type convertedNestedTy = nestedTyConv.convertType(nestedTy);
        if (!isConcreteType(convertedNestedTy)) {
          return WalkResult::advance();
        }

        Type convertedEnclosingTy = tyConv.convertType(enclosingTy);
        // Apply the nested call's bindings recursively before unifying. In particular, this
        // preserves evidence when the nested callee's type variable occurs within an aggregate
        // signature rather than as the complete operand or result type.
        UnificationMap unifications;
        if (!typesUnify(convertedEnclosingTy, convertedNestedTy, {}, &unifications)) {
          return WalkResult::advance();
        }
        auto it = unifications.find({paramName, Side::LHS});
        if (it != unifications.end()) {
          return noteCandidate(it->second);
        }
        return WalkResult::advance();
      };

      for (auto [result, nestedResultTy] :
           llvm::zip_equal(nestedCall.getResults(), nestedTgt.getFunctionType().getResults())) {
        WalkResult candidateResult = inferFromNestedBinding(result.getType(), nestedResultTy);
        if (candidateResult.wasInterrupted()) {
          return candidateResult;
        }
      }
      for (auto [operand, nestedInputTy] :
           llvm::zip_equal(nestedCall.getArgOperands(), nestedTgt.getFunctionType().getInputs())) {
        WalkResult candidateResult = inferFromNestedBinding(operand.getType(), nestedInputTy);
        if (candidateResult.wasInterrupted()) {
          return candidateResult;
        }
      }
      return WalkResult::advance();
    });

    activeInferences_.pop_back();
    if (ambiguous || (walkResult.wasInterrupted() && !inferred.has_value())) {
      return std::nullopt;
    }
    return inferred;
  }

  /// Map concrete bindings for a nested callee into the nested template's parameter scope.
  ///
  /// The current converter's keys name parameters of the enclosing template. Reusing it while
  /// walking a nested callee would therefore incorrectly bind same-named nested parameters.
  static DenseMap<Attribute, Attribute> getNestedConcreteBindings(
      CallOp nestedCall, FuncDefOp nestedTgt, TemplateOp nestedTemplate,
      const TemplateTypeConverter &enclosingTyConv
  ) {
    DenseMap<Attribute, Attribute> bindings;
    auto nestedParams = nestedTemplate.getConstOps<TemplateParamOp>();
    ArrayAttr callParams = nestedCall.getTemplateParamsAttr();

    if (!isNullOrEmpty(callParams)) {
      for (auto [paramOp, arg] : llvm::zip_equal(nestedParams, callParams.getValue())) {
        Attribute value = enclosingTyConv.convertAttr(arg);
        if (isConcreteStructParamAttr(value)) {
          bindings[FlatSymbolRefAttr::get(paramOp.getSymNameAttr())] = value;
        }
      }
    }

    // Explicit wildcards and unresolved enclosing parameters can still acquire concrete
    // bindings from the nested call's signature. Preserve concrete explicit arguments as
    // the source of truth and fill only the remaining parameters from that evidence.
    FailureOr<UnificationMap> unifyResult =
        nestedCall.unifyTypeSignature(nestedTgt.getFunctionType());
    if (failed(unifyResult)) {
      return bindings;
    }
    for (TemplateParamOp paramOp : nestedParams) {
      auto paramName = FlatSymbolRefAttr::get(paramOp.getSymNameAttr());
      if (bindings.contains(paramName)) {
        continue;
      }
      if (std::optional<Attribute> value = inferUnifiedParam(*unifyResult, paramName)) {
        Attribute convertedValue = enclosingTyConv.convertAttr(*value);
        if (isConcreteStructParamAttr(convertedValue)) {
          bindings[paramName] = convertedValue;
        }
      }
    }
    return bindings;
  }
};

/// Use `TemplateTypeConverter` to apply the given substitutions from instantiation and verify
/// that `CallOp` in the converted function are valid for their respective targets (we can emit a
/// more helpful error at this point rather than discovering it later when verifying the module).
static LogicalResult applyBodyConversions(
    CallOp op, FuncDefOp newFunc, const DenseMap<Attribute, Attribute> &paramNameToConcrete
) {
  SmallVector<Diagnostic> delayedDiagnostics;
  if (failed(substituteFunctionBody(newFunc, paramNameToConcrete, delayedDiagnostics))) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "[InstantiateFuncAtCallOp]   instantiated clone: " << newFunc << '\n');
  llzk::polymorphic::detail::reportDelayedDiagnostics(op, std::move(delayedDiagnostics));

  SymbolTableCollection tables;
  WalkResult res = newFunc.walk([&tables](CallOp nestedCall) {
    return WalkResult(nestedCall.verifySymbolUses(tables));
  });
  return failure(res.wasInterrupted());
}

/// Copy unresolved template expressions referenced by `newFuncs` into their partially-instantiated
/// parent template. Reads of concrete parameters within the copied expressions are materialized so
/// the new template contains no references to parameters that it does not preserve. Append
/// materialization warnings for the caller to report at the instantiation site.
static LogicalResult copyReferencedTemplateExprs(
    TemplateOp parentTemplate, Block &newTemplateBody, ArrayRef<FuncDefOp> newFuncs,
    const DenseMap<Attribute, Attribute> &paramNameToConcrete,
    SmallVector<Diagnostic> &delayedDiagnostics
) {
  DenseSet<StringAttr> referencedExprNames;
  auto collectExprRef = [&referencedExprNames, &paramNameToConcrete](FlatSymbolRefAttr name) {
    if (!paramNameToConcrete.contains(name)) {
      referencedExprNames.insert(name.getAttr());
    }
  };

  // Expression references can occur in a function signature, in type-bearing operands/results,
  // or in arbitrary attributes as well as in poly.read_const operations. In particular, a
  // signature-only reference must keep its defining expression in a partial template even if the
  // function body does not read that expression.
  for (FuncDefOp newFunc : newFuncs) {
    newFunc.walk([&](Operation *nestedOp) {
      auto collectTypeRefs = [&collectExprRef](Type type) { type.walk(collectExprRef); };
      for (Type type : nestedOp->getOperandTypes()) {
        collectTypeRefs(type);
      }
      for (Type type : nestedOp->getResultTypes()) {
        collectTypeRefs(type);
      }
      nestedOp->getAttrDictionary().walk(collectExprRef);
    });
    newFunc.walk([&collectExprRef](ConstReadOp readOp) {
      collectExprRef(readOp.getConstNameAttr());
    });
  }

  for (TemplateExprOp expr : parentTemplate.getConstOps<TemplateExprOp>()) {
    if (!referencedExprNames.contains(expr.getSymNameAttr())) {
      continue;
    }

    auto clonedExpr = llvm::cast<TemplateExprOp>(expr->clone());
    if (failed(substituteTemplateExprBody(clonedExpr, paramNameToConcrete, delayedDiagnostics))) {
      clonedExpr->erase();
      return failure();
    }
    newTemplateBody.push_back(clonedExpr);
  }
  return success();
}

/// Clone every free function in `parentTemplate` reached through a call in the cloned functions.
/// Rewriting these calls to the new template keeps partial instantiations self-contained.
/// Record redirected calls so symbol collisions can rename only references to copied siblings.
static SmallVector<FuncDefOp> copyReferencedTemplateSiblingFuncs(
    TemplateOp parentTemplate, TemplateOp newTemplate, FuncDefOp originalFunc, FuncDefOp newFunc,
    SymbolTableCollection &symTables, const DenseMap<Attribute, Attribute> &paramNameToConcrete,
    SmallVectorImpl<CallOp> &copiedSiblingCalls
) {
  DenseMap<Operation *, FuncDefOp> cloned;
  DenseMap<Operation *, FuncDefOp> cloneSources;
  SmallVector<FuncDefOp> copiedFuncs {newFunc};
  cloned[originalFunc] = newFunc;
  cloneSources[newFunc] = originalFunc;

  for (size_t i = 0; i < copiedFuncs.size(); ++i) {
    FuncDefOp current = copiedFuncs[i];
    SmallVector<CallOp> sourceCalls = walkCollect<CallOp>(*cloneSources.lookup(current));
    size_t callIndex = 0;
    current.walk([&](CallOp nestedCall) {
      assert(callIndex < sourceCalls.size() && "function clone must preserve calls");
      CallOp sourceCall = sourceCalls[callIndex++];
      FailureOr<SymbolLookupResult<FuncDefOp>> siblingTarget = lookupTopLevelSymbol<FuncDefOp>(
          symTables, sourceCall.getCalleeAttr(), sourceCall.getOperation(),
          /*reportMissing=*/false
      );
      if (failed(siblingTarget)) {
        return;
      }
      FuncDefOp sibling = siblingTarget->get();
      if (!sibling || sibling->getParentOp() != parentTemplate) {
        return;
      }

      FuncDefOp clonedSibling;
      if (auto found = cloned.find(sibling); found != cloned.end()) {
        clonedSibling = found->second;
      } else {
        clonedSibling = cloneFunctionWithConvertedCallees(sibling, paramNameToConcrete);
        cloned[sibling] = clonedSibling;
        cloneSources[clonedSibling] = sibling;
        copiedFuncs.push_back(clonedSibling);
      }

      // Use the resolved target's full path so calls through enclosing named modules do not
      // depend on a particular `@Template::@function` spelling.
      FailureOr<SymbolRefAttr> siblingPath = getPathFromTopRoot(sibling);
      if (failed(siblingPath)) {
        return;
      }
      SmallVector<FlatSymbolRefAttr> calleePieces = getPieces(*siblingPath);
      assert(
          calleePieces.size() >= 2 && "template function path must include template and function"
      );
      calleePieces.pop_back();
      calleePieces.pop_back();
      calleePieces.push_back(FlatSymbolRefAttr::get(newTemplate.getSymNameAttr()));
      calleePieces.push_back(FlatSymbolRefAttr::get(clonedSibling.getSymNameAttr()));
      nestedCall.setCalleeAttr(asSymbolRefAttr(calleePieces));
      copiedSiblingCalls.push_back(nestedCall);
    });
  }
  return copiedFuncs;
}

class InstantiateFuncAtCallOp final : public OpRewritePattern<CallOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the function-instantiation pattern.
  InstantiateFuncAtCallOp(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern<CallOp>(ctx), tracker_(tracker) {}

  /// Instantiate the target function or template at a call site and rewrite the callee reference.
  LogicalResult matchAndRewrite(CallOp op, PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "[InstantiateFuncAtCallOp] op: " << op << '\n');

    if (calleeReferencesTemplateParam(op)) {
      return failure();
    }

    // Lookup callee target function
    SymbolTableCollection symTables;
    FailureOr<SymbolLookupResult<FuncDefOp>> callTgtOpt = op.getCalleeTarget(symTables);
    if (failed(callTgtOpt)) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "could not find target function for call";
      });
    }
    FuncDefOp callTgt = callTgtOpt->get();

    // Check if callee is within a TemplateOp
    TemplateOp parentTemplate = llvm::dyn_cast<TemplateOp>(callTgt->getParentOp());
    if (!parentTemplate) {
      return failure(); // nothing to do if not parameterized
    }
    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateFuncAtCallOp]  target function in template "
                     << parentTemplate.getSymName() << '\n'
    );

    // Perform type unification with tracking to infer the instantiated type(s). Even though
    // `CallOp` verification already checked that caller and callee types unify, the progress of
    // instantiation so far may have brought together a chain of calls across templates where each
    // individual unification check passed due to permissive type variables and/or symbols in the
    // middle but the overall chain does not unify. Hence, this unification may fail and should
    // produce a meaningful error message if it does.
    // See: `test/Transforms/Flattening/instantiate_funcs_fail.llzk`
    FailureOr<UnificationMap> unifyResult = unifyTypeSignature(op, callTgt, rewriter);
    if (failed(unifyResult)) {
      return failure();
    }
    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateFuncAtCallOp]  unifications of types: "
                     << debug::toStringList(unifyResult.value()) << '\n'
    );

    // Maps template parameter symbols to the instantiation value at the call site.
    DenseMap<Attribute, Attribute> paramNameToConcrete;
    auto collectRes = collectConcreteTemplateParams(
        op, rewriter, symTables, callTgt, parentTemplate, unifyResult.value(), paramNameToConcrete
    );
    if (failed(collectRes)) {
      return failure();
    }

    if (paramNameToConcrete.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "[InstantiateFuncAtCallOp]  skip: no concrete params\n");
      return failure();
    }

    evaluateTemplateExprs(parentTemplate, paramNameToConcrete);

    FailureOr<InstantiationLayout> layoutResult =
        buildInstantiationLayout(parentTemplate, op.getTemplateParamsAttr(), paramNameToConcrete);
    if (failed(layoutResult)) {
      return failure();
    }
    InstantiationLayout layout = std::move(*layoutResult);
    ModuleOp parentModule = getParentOfType<ModuleOp>(parentTemplate);
    assert(parentModule && "TemplateOp must be nested in a ModuleOp");

    FailureOr<SymbolRefAttr> newCalleeAttr =
        layout.remainingNames.empty() ? instantiateFully(
                                            op, rewriter, symTables, callTgt, parentTemplate,
                                            parentModule, layout.templateNameWithAttrs,
                                            layout.concreteParamKey, paramNameToConcrete, tracker_
                                        )
                                      : instantiatePartially(
                                            op, rewriter, symTables, callTgt, parentTemplate,
                                            parentModule, layout, paramNameToConcrete, tracker_
                                        );
    if (failed(newCalleeAttr)) {
      return failure();
    }

    FailureOr<SymbolRefAttr> originalCalleePath = getPathFromTopRoot(callTgt);
    if (failed(originalCalleePath)) {
      return failure();
    }
    tracker_.recordInstantiation(*originalCalleePath);

    // Update the CallOp to point to the instantiated function and mark the module as modified.
    rewriter.modifyOpInPlace(op, [&op, &newCalleeAttr, &layout]() {
      LLVM_DEBUG({
        llvm::dbgs() << "[InstantiateFuncAtCallOp]  updating callee from " << op.getCalleeAttr()
                     << " to " << *newCalleeAttr << '\n';
      });
      op.setCalleeAttr(*newCalleeAttr);
      op.setTemplateParamsAttr(layout.rewrittenCallParams);
    });
    tracker_.updateModifiedFlag(true);
    return success();
  }

private:
  /// Re-run call/callee type unification so flattening can surface a useful error if a chain of
  /// partially-instantiated calls stops unifying once earlier substitutions have been applied.
  static FailureOr<UnificationMap>
  unifyTypeSignature(CallOp op, FuncDefOp callTgt, PatternRewriter &rewriter) {
    FailureOr<UnificationMap> unifyResult = op.unifyTypeSignature(callTgt.getFunctionType());
    if (succeeded(unifyResult)) {
      return unifyResult;
    }
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag.append("target function type does not unify with call type ")
          .append(op.getTypeSignature())
          .attachNote(callTgt.getLoc())
          .append("target function declared here");
    });
  }

  /// Populate the concrete subset of template parameters chosen for this instantiation, using
  /// explicit call-site arguments when present and otherwise relying on unification.
  static LogicalResult collectConcreteTemplateParams(
      CallOp op, PatternRewriter &rewriter, SymbolTableCollection &symTables, FuncDefOp callTgt,
      TemplateOp parentTemplate, const UnificationMap &unifyResult,
      DenseMap<Attribute, Attribute> &paramNameToConcrete
  ) {
    auto realParams = parentTemplate.getConstOps<TemplateParamOp>();
    ArrayAttr callParams = op.getTemplateParamsAttr();
    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateFuncAtCallOp]  TemplateParamsAttr: " << callParams << '\n'
    );

    auto recordConcreteParam = [&](FlatSymbolRefAttr paramName, TemplateParamOp paramOp,
                                   Attribute concreteValue) {
      if (failed(op.verifyTemplateParamValueCompatibility(concreteValue, paramOp))) {
        return failIncompatibleInferredParam(op, rewriter, paramName, paramOp);
      }
      paramNameToConcrete[paramName] = concreteValue;
      return success();
    };

    // If there's no template instantiation list, must infer all template parameters.
    if (isNullOrEmpty(callParams)) {
      for (auto paramOp : realParams) {
        auto paramName = FlatSymbolRefAttr::get(paramOp.getSymNameAttr());
        auto inferredValOpt = inferUnifiedParam(unifyResult, paramName);
        if (!inferredValOpt.has_value()) {
          LLVM_DEBUG(
              llvm::dbgs() << "[InstantiateFuncAtCallOp]  unification for param '" << paramName
                           << "': not found\n"
          );
          continue;
        }
        Attribute inferredVal = *inferredValOpt;
        LLVM_DEBUG(
            llvm::dbgs() << "[InstantiateFuncAtCallOp]  inferredVal: " << inferredVal << '\n'
        );
        if (!isConcreteStructParamAttr(inferredVal)) {
          LLVM_DEBUG(
              llvm::dbgs() << "[InstantiateFuncAtCallOp]  unification for param '" << paramName
                           << "': not concrete, " << inferredVal << '\n'
          );
          continue;
        }
        if (failed(recordConcreteParam(paramName, paramOp, inferredVal))) {
          return failure();
        }
      }
      return success();
    }

    // As stated earlier, need to run the verification checks again to ensure the
    // instantiation is valid, except for the size check because that cannot change.
    assert((callParams.size() == llvm::range_size(realParams)) && "per CallOpVerifier");
    if (failed(op.verifyTemplateParamValuesCompatibility(realParams))) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag.append("incompatible with specified param type(s)");
      });
    }
    if (failed(op.verifyTemplateParamsMatchInferred(realParams, unifyResult))) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag.append("incompatible with inferred param value(s)");
      });
    }

    // When template parameters are specified on the CallOp, use them as the source of truth
    // for concrete arguments, then infer wildcard parameters against the full explicit map.
    SmallVector<std::pair<TemplateParamOp, FlatSymbolRefAttr>> wildcardParams;
    for (auto [paramOp, attr] : llvm::zip_equal(realParams, callParams.getValue())) {
      auto paramName = FlatSymbolRefAttr::get(paramOp.getSymNameAttr());
      AttrConcreteness classification = classifyAttrConcreteness(attr);
      if (classification == AttrConcreteness::Concrete) {
        paramNameToConcrete[paramName] = attr;
        continue;
      }

      if (classification == AttrConcreteness::NonConcrete) {
        LLVM_DEBUG(
            llvm::dbgs() << "[InstantiateFuncAtCallOp]  unification for param '" << paramName
                         << "': not concrete, " << attr << '\n'
        );
        continue;
      }
      wildcardParams.emplace_back(paramOp, paramName);
    }

    WildcardTypeBodyInferer bodyInferer(symTables, paramNameToConcrete);
    for (auto [paramOp, paramName] : wildcardParams) {
      auto inferredValOpt = inferUnifiedParam(unifyResult, paramName);
      if (inferredValOpt.has_value() && isConcreteStructParamAttr(*inferredValOpt)) {
        LLVM_DEBUG(
            llvm::dbgs() << "[InstantiateFuncAtCallOp]  inferredVal: " << *inferredValOpt << '\n'
        );
        if (failed(recordConcreteParam(paramName, paramOp, *inferredValOpt))) {
          return failure();
        }
        continue;
      }

      inferredValOpt = bodyInferer.infer(callTgt, paramName);
      if (inferredValOpt.has_value() && isConcreteStructParamAttr(*inferredValOpt)) {
        LLVM_DEBUG(
            llvm::dbgs() << "[InstantiateFuncAtCallOp]  body-inferred value for param '"
                         << paramName << "': " << *inferredValOpt << '\n'
        );
        if (failed(recordConcreteParam(paramName, paramOp, *inferredValOpt))) {
          return failure();
        }
      }
    }
    return success();
  }

  /// Return the full-instantiation callee spelling that is valid at this call site.
  static SymbolRefAttr
  buildFullInstantiationCalleeForCall(CallOp op, FlatSymbolRefAttr newFuncName) {
    // Callee: drop template & original function names, add the new module-level function name.
    // Original: @[prefix...]::@TemplateName::@funcName
    // New:      @[prefix...]::@newFuncName
    SmallVector<FlatSymbolRefAttr> symPieces = getPieces(op.getCalleeAttr());
    assert(symPieces.size() >= 2 && "callee must include at least template and function names");
    symPieces.pop_back(); // remove original function name
    symPieces.pop_back(); // remove template name
    symPieces.push_back(newFuncName);
    return asSymbolRefAttr(symPieces);
  }

  /// Create or reuse a fully-instantiated clone in the parent module and return the rewritten
  /// module-level callee reference.
  static FailureOr<SymbolRefAttr> instantiateFully(
      CallOp op, PatternRewriter &rewriter, SymbolTableCollection &symTables, FuncDefOp callTgt,
      TemplateOp parentTemplate, ModuleOp parentModule, StringRef templateNameWithAttrs,
      ArrayAttr concreteParamKey, const DenseMap<Attribute, Attribute> &paramNameToConcrete,
      ConversionTracker &tracker
  ) {
    if (auto cachedPath = tracker.getFullFuncInstantiation(callTgt, concreteParamKey)) {
      LLVM_DEBUG(
          llvm::dbgs() << "[InstantiateFuncAtCallOp]  reusing full instantiation function: "
                       << *cachedPath << '\n'
      );
      return buildFullInstantiationCalleeForCall(op, getPieces(*cachedPath).back());
    }

    std::string newFuncName;
    llvm::raw_string_ostream(newFuncName) << templateNameWithAttrs << '_' << callTgt.getSymName();
    FuncDefOp newFunc = cloneFunctionWithConvertedCallees(callTgt, paramNameToConcrete);
    newFunc.setSymName(newFuncName);
    // Insert before the TemplateOp; symbol table may adjust the name to avoid existing symbols.
    symTables.getSymbolTable(parentModule).insert(newFunc, Block::iterator(parentTemplate));
    StringAttr actualNewFuncName = newFunc.getSymNameAttr();
    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateFuncAtCallOp]  created full instantiation function: "
                     << actualNewFuncName << '\n'
    );
    if (failed(applyBodyConversions(op, newFunc, paramNameToConcrete))) {
      LLVM_DEBUG(
          llvm::dbgs() << "[InstantiateFuncAtCallOp]   body conversion failed for "
                       << actualNewFuncName << '\n'
      );
      newFunc->erase();
      return rewriter.notifyMatchFailure(op, [&actualNewFuncName](Diagnostic &diag) {
        diag.append("failure while creating instantiated function ", actualNewFuncName);
      });
    }

    FailureOr<SymbolRefAttr> newCalleeAttr = getPathFromTopRoot(newFunc);
    if (failed(newCalleeAttr)) {
      return failure();
    }
    tracker.recordFullFuncInstantiation(callTgt, concreteParamKey, *newCalleeAttr);
    return buildFullInstantiationCalleeForCall(
        op, FlatSymbolRefAttr::get(newFunc.getSymNameAttr())
    );
  }

  /// Create or reuse a partially-instantiated template that preserves the remaining non-concrete
  /// parameters and return the rewritten nested callee reference.
  /// Reuse is keyed by the source function and exact ordered concrete bindings. The rendered name
  /// is only a preferred symbol name and may be changed by SymbolTable insertion.
  static FailureOr<SymbolRefAttr> instantiatePartially(
      CallOp op, PatternRewriter &rewriter, SymbolTableCollection &symTables, FuncDefOp callTgt,
      TemplateOp parentTemplate, ModuleOp parentModule, const InstantiationLayout &layout,
      const DenseMap<Attribute, Attribute> &paramNameToConcrete, ConversionTracker &tracker
  ) {
    if (auto cached = tracker.lookupPartialFuncInstantiation(callTgt, layout.concreteParamKey)) {
      SmallVector<FlatSymbolRefAttr> symPieces = getPieces(op.getCalleeAttr());
      SmallVector<FlatSymbolRefAttr> cachedSuffix = getPieces(*cached);
      assert(symPieces.size() >= 2 && "callee must include at least template and function names");
      assert(cachedSuffix.size() == 2 && "cached callee suffix must contain template and function");
      symPieces.pop_back();
      symPieces.pop_back();
      symPieces.push_back(cachedSuffix[0]);
      symPieces.push_back(cachedSuffix[1]);
      SymbolRefAttr cachedCallee = asSymbolRefAttr(symPieces);
      LLVM_DEBUG(
          llvm::dbgs() << "[InstantiateFuncAtCallOp]  reusing partial instantiation: "
                       << cachedCallee << '\n'
      );
      return cachedCallee;
    }
    TemplateOp newTemplate = createPartialInstantiationTemplate(parentTemplate, layout);

    Block &newTemplateBody = newTemplate.getBodyRegion().front();
    for (Attribute name : layout.remainingNames) {
      FlatSymbolRefAttr nameSym = llvm::cast<FlatSymbolRefAttr>(name);
      Operation *paramOp = symTables.getSymbolTable(parentTemplate).lookup(nameSym.getAttr());
      assert(paramOp && "symbol must exist");
      newTemplateBody.push_back(paramOp->clone());
    }

    // Clone and partially convert the function (concretize only the concrete params).
    FuncDefOp newFunc = cloneFunctionWithConvertedCallees(callTgt, paramNameToConcrete);
    SmallVector<CallOp> copiedSiblingCalls;
    SmallVector<FuncDefOp> copiedFuncs = copyReferencedTemplateSiblingFuncs(
        parentTemplate, newTemplate, callTgt, newFunc, symTables, paramNameToConcrete,
        copiedSiblingCalls
    );
    SmallVector<Diagnostic> expressionDiagnostics;
    auto copyRes = copyReferencedTemplateExprs(
        parentTemplate, newTemplateBody, copiedFuncs, paramNameToConcrete, expressionDiagnostics
    );
    if (failed(copyRes)) {
      for (FuncDefOp copiedFunc : copiedFuncs) {
        copiedFunc->erase();
      }
      newTemplate->erase();
      return rewriter.notifyMatchFailure(op, "failure while copying template expressions");
    }

    // Insert before body conversion so nested concrete callees verify from the root module. Use
    // SymbolTable::insert() so both physical symbol names are unique if necessary.
    for (FuncDefOp copiedFunc : copiedFuncs) {
      symTables.getSymbolTable(newTemplate).insert(copiedFunc);
    }
    StringAttr provisionalTemplateName = newTemplate.getSymNameAttr();
    symTables.getSymbolTable(parentModule).insert(newTemplate, Block::iterator(parentTemplate));
    if (newTemplate.getSymNameAttr() != provisionalTemplateName) {
      // Only calls redirected to copied siblings belong to this partial template. External
      // callees can share its provisional name and must retain their original references.
      for (CallOp copiedSiblingCall : copiedSiblingCalls) {
        SmallVector<FlatSymbolRefAttr> calleePieces = getPieces(copiedSiblingCall.getCalleeAttr());
        assert(
            calleePieces.size() >= 2 && "copied sibling path must include template and function"
        );
        calleePieces[calleePieces.size() - 2] =
            FlatSymbolRefAttr::get(newTemplate.getSymNameAttr());
        copiedSiblingCall.setCalleeAttr(asSymbolRefAttr(calleePieces));
      }
    }
    for (FuncDefOp copiedFunc : copiedFuncs) {
      if (succeeded(applyBodyConversions(op, copiedFunc, paramNameToConcrete))) {
        continue;
      }
      StringAttr newFuncName = copiedFunc.getSymNameAttr();
      LLVM_DEBUG(
          llvm::dbgs() << "[InstantiateFuncAtCallOp]   body conversion failed for " << newFuncName
                       << '\n'
      );
      newTemplate->erase();
      return rewriter.notifyMatchFailure(op, [&newFuncName](Diagnostic &diag) {
        diag.append("failure while creating instantiated function ", newFuncName);
      });
    }

    // Use the post-insertion names. The preferred template name may have collided.
    SmallVector<FlatSymbolRefAttr> symPieces = getPieces(op.getCalleeAttr());
    assert(symPieces.size() >= 2 && "callee must include at least template and function names");
    symPieces.pop_back();
    symPieces.pop_back(); // remove original template name
    symPieces.push_back(FlatSymbolRefAttr::get(newTemplate.getSymNameAttr()));
    symPieces.push_back(FlatSymbolRefAttr::get(newFunc.getSymNameAttr()));
    SymbolRefAttr newCallee = asSymbolRefAttr(symPieces);

    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateFuncAtCallOp]  created partial instantiation: " << newCallee
                     << '\n'
    );
    llzk::polymorphic::detail::reportDelayedDiagnostics(op, std::move(expressionDiagnostics));
    // Publish only after insertion and body conversion have succeeded.
    tracker.recordPartialFuncInstantiation(callTgt, layout.concreteParamKey, newTemplate, newFunc);
    return newCallee;
  }
};

/// Run function instantiation patterns once over the module.
LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<InstantiateFuncAtCallOp>(ctx, tracker);
  MatchFailureListener failureListener;
  walkAndApplyPatterns(modOp, std::move(patterns), &failureListener);
  return failure(failureListener.hadFailure);
}

} // namespace Step2_InstantiateFunctions

namespace Step3_Unroll {

/// Fully expand annotated loops using the trusted trip count and, for for loops, induction values.
/// Run before folding so zero- and single-trip loops cannot discard conflicting annotations.
template <HasInterface<LoopLikeOpInterface> OpClass>
class AnnotatedLoopUnrollPattern : public OpRewritePattern<OpClass> {
public:
  AnnotatedLoopUnrollPattern(MLIRContext *ctx, ConversionTracker &t)
      : OpRewritePattern<OpClass>(ctx), tracker(t) {}

  /// Fully unroll loop-like ops whose trip count is statically known.
  LogicalResult matchAndRewrite(OpClass loopOp, PatternRewriter &rewriter) const override {
    auto bounds = loopOp->template getAttrOfType<LoopBoundsAttr>(LoopBoundsAttr::name);
    if (!bounds || tracker.mustPreserveLoop(loopOp)) {
      return failure();
    }
    const llvm::DynamicAPInt &lower = bounds.getLower();
    const llvm::DynamicAPInt &upper = bounds.getUpper();
    const llvm::DynamicAPInt &step = bounds.getStep();
    if (step == 0) {
      return rewriter.notifyMatchFailure(
          loopOp, "llzk.loopbounds requires a nonzero step for unrolling"
      );
    }
    llvm::DynamicAPInt count = llvm::ceilDiv(upper - lower, step);
    if (count < 0) {
      count = llvm::DynamicAPInt(0);
    }
    if (auto computed = getConstantTripCount(loopOp);
        computed && toDynamicAPInt(*computed) != count) {
      loopOp.emitWarning() << "llzk.loopbounds trip count " << llvm::toString(toAPSInt(count), 10)
                           << " does not match computed trip count "
                           << llvm::toString(*computed, 10, false) << "; using llzk.loopbounds";
    }
    // The preservation guard already checked representability. Consume Expected even when
    // assertions are disabled, as checked LLVM builds require successful results to be checked.
    uint64_t tripCount =
        llvm::cantFail(checkedToUInt64(count), "unrepresentable trip counts must be preserved");

    if (failed(unroll(loopOp, rewriter, bounds, tripCount))) {
      return failure();
    }
    tracker.updateModifiedFlag(true);
    return success();
  }

private:
  /// Expand a for loop with induction values supplied by the annotation.
  template <typename ForOp>
  static LogicalResult
  unroll(ForOp loopOp, PatternRewriter &rewriter, LoopBoundsAttr bounds, uint64_t tripCount) {
    const llvm::DynamicAPInt &lower = bounds.getLower();
    const llvm::DynamicAPInt &step = bounds.getStep();

    Value iv = loopOp.getInductionVar();
    Type ivType = iv.getType();
    unsigned width =
        ivType.isIndex() ? IndexType::kInternalStorageBitWidth : ivType.getIntOrFloatBitWidth();

    // The preservation guard already checked all induction values that will be materialized.
    SmallVector<Value> results(loopOp.getInits());
    llvm::DynamicAPInt inductionValue = lower;
    for (uint64_t i = 0; i < tripCount; ++i, inductionValue += step) {
      IRMapping mapping;
      mapping.map(loopOp.getRegionIterArgs(), results);
      if (!iv.use_empty()) {
        Value constant = arith::ConstantOp::create(
            rewriter, loopOp.getLoc(),
            rewriter.getIntegerAttr(ivType, toAPSInt(inductionValue).sextOrTrunc(width))
        );
        mapping.map(iv, constant);
      }
      for (Operation &op : loopOp.getBody()->without_terminator()) {
        rewriter.clone(op, mapping);
      }
      for (auto [result, yielded] : llvm::zip_equal(results, loopOp.getYieldedValues())) {
        result = mapping.lookupOrDefault(yielded);
      }
    }
    rewriter.replaceOp(loopOp, results);
    return success();
  }

  /// Expand both while regions, including the final condition-region execution. The annotation
  /// controls how often the after region executes; the regions themselves compute carried values.
  static LogicalResult
  unroll(scf::WhileOp loopOp, PatternRewriter &rewriter, LoopBoundsAttr, uint64_t tripCount) {
    auto cloneBlock = [&](Block *block, ValueRange arguments, ValueRange forwarded) {
      IRMapping mapping;
      mapping.map(block->getArguments(), arguments);
      for (Operation &op : block->without_terminator()) {
        rewriter.clone(op, mapping);
      }
      return llvm::map_to_vector(forwarded, [&](Value value) {
        return mapping.lookupOrDefault(value);
      });
    };

    SmallVector<Value> values(loopOp.getInits());
    for (uint64_t i = 0; i < tripCount; ++i) {
      values = cloneBlock(loopOp.getBeforeBody(), values, loopOp.getConditionOp().getArgs());
      values = cloneBlock(loopOp.getAfterBody(), values, loopOp.getYieldOp().getResults());
    }
    values = cloneBlock(loopOp.getBeforeBody(), values, loopOp.getConditionOp().getArgs());
    rewriter.replaceOp(loopOp, values);
    return success();
  }

  ConversionTracker &tracker;
};

/// Unroll loops with a constant computed trip count within the configured limit.
template <HasInterface<LoopLikeOpInterface> OpClass>
class ComputedLoopUnrollPattern : public OpRewritePattern<OpClass> {
public:
  ComputedLoopUnrollPattern(MLIRContext *ctx, ConversionTracker &t)
      : OpRewritePattern<OpClass>(ctx), tracker(t) {}

  LogicalResult matchAndRewrite(OpClass loopOp, PatternRewriter &rewriter) const override {
    if (loopOp->hasAttr(LoopBoundsAttr::name) || tracker.mustPreserveLoop(loopOp)) {
      return failure();
    }
    if (auto maybeConstant = getConstantTripCount(loopOp)) {
      assert(
          maybeConstant->getActiveBits() <= 64 && "unrepresentable trip counts must be preserved"
      );
      uint64_t tripCount = maybeConstant->getZExtValue();
      if (tripCount == 0) {
        rewriter.replaceOp(loopOp, loopOp.getInits());
        return success();
      } else if (tripCount == 1) {
        return loopOp.promoteIfSingleIteration(rewriter);
      }
      return loopUnrollByFactor(loopOp, tripCount);
    }
    return failure();
  }

private:
  ConversionTracker &tracker;
};

/// Run loop unrolling for supported loop dialects.
LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();

  // Must unroll via `LoopBoundsAttr` if present
  RewritePatternSet annotatedPatterns(ctx);
  annotatedPatterns.add<
      AnnotatedLoopUnrollPattern<scf::ForOp>, AnnotatedLoopUnrollPattern<scf::WhileOp>,
      AnnotatedLoopUnrollPattern<affine::AffineForOp>>(ctx, tracker);
  MatchFailureListener failureListener;
  walkAndApplyPatterns(modOp, std::move(annotatedPatterns), &failureListener);
  if (failureListener.hadFailure) {
    return failure();
  }

  // Otherwise, fall through to computed trip count unrolling
  RewritePatternSet patterns(ctx);
  patterns
      .add<ComputedLoopUnrollPattern<scf::ForOp>, ComputedLoopUnrollPattern<affine::AffineForOp>>(
          ctx, tracker
      );
  return applyAndFoldGreedily(modOp, tracker, std::move(patterns));
}

} // namespace Step3_Unroll

namespace Step4_InstantiateAffineMaps {

/// Return constant integer values for all fold results, if every fold result is constant.
///
/// Adapted from `mlir::getConstantIntValues()` but that one failed in CI for an unknown reason.
/// This version uses a basic loop instead of llvm::map_to_vector().
std::optional<SmallVector<int64_t>> getConstantIntValues(ArrayRef<OpFoldResult> ofrs) {
  SmallVector<int64_t> res;
  for (OpFoldResult ofr : ofrs) {
    std::optional<int64_t> cv = getConstantIntValue(ofr);
    if (!cv.has_value()) {
      return std::nullopt;
    }
    res.push_back(cv.value());
  }
  return res;
}

/// Folds affine-map parameters using the map operands supplied at an instantiation site.
struct AffineMapFolder {
  /// Inputs that describe affine-map operands and the parameter list being folded.
  struct Input {
    /// Operand groups corresponding to affine-map parameters.
    OperandRangeRange mapOpGroups;
    /// Number of dimensions in each operand group.
    DenseI32ArrayAttr dimsPerGroup;
    /// Parameter list containing affine maps and non-map attributes.
    ArrayRef<Attribute> paramsOfStructTy;
  };

  /// Outputs after replacing foldable affine-map parameters with concrete attributes.
  struct Output {
    /// Operand groups for affine maps that could not be folded.
    SmallVector<SmallVector<Value>> mapOpGroups;
    /// Dimension counts corresponding to remaining map operand groups.
    SmallVector<int32_t> dimsPerGroup;
    /// Parameter list with folded values substituted where possible.
    SmallVector<Attribute> paramsOfStructTy;
  };

  /// Convert owned output operand groups into `ValueRange` views for op builders.
  static inline SmallVector<ValueRange> getConvertedMapOpGroups(const Output &out) {
    return llvm::map_to_vector(out.mapOpGroups, [](const SmallVector<Value> &grp) {
      return ValueRange(grp);
    });
  }

  /// Fold any affine-map attributes in `in.paramsOfStructTy` whose operands are all constants.
  static LogicalResult
  fold(PatternRewriter &rewriter, const Input &in, Output &out, Operation *op, const char *aspect) {
    if (in.mapOpGroups.empty()) {
      // No affine map operands so nothing to do
      return failure();
    }

    assert(in.mapOpGroups.size() <= in.paramsOfStructTy.size());
    assert(std::cmp_equal(in.mapOpGroups.size(), in.dimsPerGroup.size()));

    size_t idx = 0; // index in `mapOpGroups`, i.e., the number of AffineMapAttr encountered
    for (Attribute sizeAttr : in.paramsOfStructTy) {
      if (AffineMapAttr m = dyn_cast<AffineMapAttr>(sizeAttr)) {
        ValueRange currMapOps = in.mapOpGroups[idx++];
        LLVM_DEBUG(
            llvm::dbgs() << "[AffineMapFolder] currMapOps: " << debug::toStringList(currMapOps)
                         << '\n'
        );
        SmallVector<OpFoldResult> currMapOpsCast = getAsOpFoldResult(currMapOps);
        LLVM_DEBUG(
            llvm::dbgs() << "[AffineMapFolder] currMapOps as fold results: "
                         << debug::toStringList(currMapOpsCast) << '\n'
        );
        if (auto constOps = Step4_InstantiateAffineMaps::getConstantIntValues(currMapOpsCast)) {
          SmallVector<Attribute> result;
          bool hasPoison = false; // indicates divide by 0 or mod by <1
          auto constAttrs = llvm::map_to_vector(*constOps, [&rewriter](int64_t v) -> Attribute {
            return rewriter.getIndexAttr(v);
          });
          LogicalResult foldResult = m.getAffineMap().constantFold(constAttrs, result, &hasPoison);
          if (hasPoison) {
            // Diagnostic remark: could be removed for release builds if too noisy
            op->emitRemark()
                .append(
                    "Cannot fold affine_map for ", aspect, ' ', out.paramsOfStructTy.size(),
                    " due to divide by 0 or modulus with negative divisor"
                )
                .report();
            return failure();
          }
          if (failed(foldResult)) {
            // Diagnostic remark: could be removed for release builds if too noisy
            op->emitRemark()
                .append(
                    "Folding affine_map for ", aspect, ' ', out.paramsOfStructTy.size(), " failed"
                )
                .report();
            return failure();
          }
          if (result.size() != 1) {
            // Diagnostic remark: could be removed for release builds if too noisy
            op->emitRemark()
                .append(
                    "Folding affine_map for ", aspect, ' ', out.paramsOfStructTy.size(),
                    " produced ", result.size(), " results but expected 1"
                )
                .report();
            return failure();
          }
          assert(!llvm::isa<AffineMapAttr>(result[0]) && "not converted");
          out.paramsOfStructTy.push_back(result[0]);
          continue;
        }
        // If affine but not foldable, preserve the map ops
        out.mapOpGroups.emplace_back(currMapOps);
        out.dimsPerGroup.push_back(in.dimsPerGroup[idx - 1]); // idx was already incremented
      }
      // If not affine, preserve the original.
      out.paramsOfStructTy.push_back(sizeAttr);
    }
    assert(idx == in.mapOpGroups.size() && "all affine_map not processed");
    assert(
        in.paramsOfStructTy.size() == out.paramsOfStructTy.size() &&
        "produced wrong number of dimensions"
    );

    return success();
  }
};

/// At CreateArrayOp, instantiate ArrayType parameterized with affine_map dimension size(s)
class InstantiateAtCreateArrayOp final : public OpRewritePattern<CreateArrayOp> {
  [[maybe_unused]]
  ConversionTracker &tracker_;

public:
  /// Construct the array-creation affine-map instantiation pattern.
  InstantiateAtCreateArrayOp(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx), tracker_(tracker) {}

  /// Rewrite `array.new` when affine-map dimensions can be folded to concrete sizes.
  LogicalResult matchAndRewrite(CreateArrayOp op, PatternRewriter &rewriter) const override {
    ArrayType oldResultType = op.getType();

    AffineMapFolder::Output out;
    AffineMapFolder::Input in = {
        op.getMapOperands(),
        op.getNumDimsPerMapAttr(),
        oldResultType.getDimensionSizes(),
    };
    if (failed(AffineMapFolder::fold(rewriter, in, out, op, "array dimension"))) {
      return failure();
    }

    ArrayType newResultType = ArrayType::get(oldResultType.getElementType(), out.paramsOfStructTy);
    if (newResultType == oldResultType) {
      return failure(); // nothing changed
    }
    // ASSERT: folding only preserves the original Attribute or converts affine to integer
    assert(tracker_.isLegalConversion(oldResultType, newResultType, "InstantiateAtCreateArrayOp"));
    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateAtCreateArrayOp] instantiating " << oldResultType << " as "
                     << newResultType << " in \"" << op << "\"\n"
    );
    replaceOpWithNewOp<CreateArrayOp>(
        rewriter, op, newResultType, AffineMapFolder::getConvertedMapOpGroups(out), out.dimsPerGroup
    );
    return success();
  }
};

/// Instantiate parameterized StructType resulting from CallOp targeting "compute()" functions.
class InstantiateAtCallOpCompute final : public OpRewritePattern<CallOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the struct-compute call result instantiation pattern.
  InstantiateAtCallOpCompute(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx), tracker_(tracker) {}

  /// Refine the result type of calls to struct `compute` functions when parameters become known.
  LogicalResult matchAndRewrite(CallOp op, PatternRewriter &rewriter) const override {
    if (!op.calleeIsStructCompute()) {
      // this pattern only applies when the callee is "compute()" within a struct
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "[InstantiateAtCallOpCompute] target: " << op.getCallee() << '\n');
    StructType oldRetTy = op.getSingleResultTypeOfCompute();
    LLVM_DEBUG(llvm::dbgs() << "[InstantiateAtCallOpCompute]   oldRetTy: " << oldRetTy << '\n');
    ArrayAttr params = oldRetTy.getParams();
    if (isNullOrEmpty(params)) {
      // nothing to do if the StructType is not parameterized
      return failure();
    }

    AffineMapFolder::Output out;
    AffineMapFolder::Input in = {
        op.getMapOperands(),
        op.getNumDimsPerMapAttr(),
        params.getValue(),
    };
    if (!in.mapOpGroups.empty()) {
      // If there are affine map operands, attempt to fold them to a constant.
      if (failed(AffineMapFolder::fold(rewriter, in, out, op, "struct parameter"))) {
        return failure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "[InstantiateAtCallOpCompute]   folded affine_map in result type params\n";
      });
    } else {
      // If there are no affine map operands, attempt to refine the result type of the CallOp using
      // the function argument types and the type of the target function.
      auto callArgTypes = op.getArgOperands().getTypes();
      if (callArgTypes.empty()) {
        // no refinement possible if no function arguments
        return failure();
      }
      if (calleeReferencesTemplateParam(op)) {
        return failure();
      }
      SymbolTableCollection tables;
      auto lookupRes = lookupTopLevelSymbol<FuncDefOp>(tables, op.getCalleeAttr(), op);
      if (failed(lookupRes)) {
        return failure();
      }
      if (failed(instantiateViaTargetType(
              in, out, callArgTypes, lookupRes->get(), lookupRes->getNamespace()
          ))) {
        return failure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "[InstantiateAtCallOpCompute]   propagated instantiations via symrefs in "
                        "result type params: "
                     << debug::toStringList(out.paramsOfStructTy) << '\n';
      });
    }

    StructType newRetTy = StructType::get(oldRetTy.getNameRef(), out.paramsOfStructTy);
    LLVM_DEBUG(llvm::dbgs() << "[InstantiateAtCallOpCompute]   newRetTy: " << newRetTy << '\n');
    if (newRetTy == oldRetTy) {
      return failure(); // nothing changed
    }
    // The `newRetTy` is computed via instantiateViaTargetType() which can only preserve the
    // original Attribute or convert to a concrete attribute via the unification process. Thus, if
    // the conversion here is illegal it means there is a type conflict within the LLZK code that
    // prevents instantiation of the struct with the requested type.
    if (!tracker_.isLegalConversion(oldRetTy, newRetTy, "InstantiateAtCallOpCompute")) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag.append(
            "result type mismatch: due to struct instantiation, expected type ", newRetTy,
            ", but found ", oldRetTy
        );
      });
    }
    LLVM_DEBUG(llvm::dbgs() << "[InstantiateAtCallOpCompute] replaced " << op);
    CallOp newOp = replaceOpWithNewOp<CallOp>(
        rewriter, op, TypeRange {newRetTy}, op.getCallee(),
        AffineMapFolder::getConvertedMapOpGroups(out), out.dimsPerGroup, op.getArgOperands()
    );
    (void)newOp; // tell compiler it's intentionally unused in release builds
    LLVM_DEBUG(llvm::dbgs() << " with " << newOp << '\n');
    return success();
  }

private:
  /// Use the type of the target function to propagate instantiation knowledge from the function
  /// argument types to the function return type in the CallOp.
  inline LogicalResult instantiateViaTargetType(
      const AffineMapFolder::Input &in, AffineMapFolder::Output &out,
      OperandRange::type_range callArgTypes, FuncDefOp targetFunc,
      ArrayRef<StringRef> targetNamespace
  ) const {
    assert(targetFunc.isStructCompute()); // since `op.calleeIsStructCompute()`
    ArrayAttr targetResTyParams = targetFunc.getSingleResultTypeOfCompute().getParams();
    assert(!isNullOrEmpty(targetResTyParams)); // same cardinality as `in.paramsOfStructTy`
    assert(in.paramsOfStructTy.size() == targetResTyParams.size()); // verifier ensures this

    if (llvm::all_of(in.paramsOfStructTy, [](Attribute attr) {
      return isConcreteStructParamAttr(attr);
    })) {
      // Nothing can change if everything is already concrete
      return failure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << '[' << __FUNCTION__ << ']'
                   << " call arg types: " << debug::toStringList(callArgTypes) << '\n';
      llvm::dbgs() << '[' << __FUNCTION__ << ']' << " target func arg types: "
                   << debug::toStringList(targetFunc.getArgumentTypes()) << '\n';
      llvm::dbgs() << '[' << __FUNCTION__ << ']'
                   << " struct params @ call: " << debug::toStringList(in.paramsOfStructTy) << '\n';
      llvm::dbgs() << '[' << __FUNCTION__ << ']'
                   << " target struct params: " << debug::toStringList(targetResTyParams) << '\n';
    });

    UnificationMap unifications;
    bool unifies =
        typeListsUnify(callArgTypes, targetFunc.getArgumentTypes(), targetNamespace, &unifications);
    (void)unifies; // tell compiler it's intentionally unused in builds without assertions
    assert(unifies && "should have been checked by verifiers");

    LLVM_DEBUG({
      llvm::dbgs() << '[' << __FUNCTION__ << ']'
                   << " unifications of arg types: " << debug::toStringList(unifications) << '\n';
    });

    // Check for RHS SymRef (i.e., from the target function) that have LHS concrete Attributes (i.e.
    // from the call argument types) without any struct parameters (because the type with concrete
    // struct parameters will be used to instantiate the target struct rather than the fully
    // flattened struct type resulting in type mismatch of the callee to target) and perform those
    // replacements in the `targetFunc` return type to produce the new result type for the CallOp.
    SmallVector<Attribute> newReturnStructParams = llvm::map_to_vector(
        llvm::zip_equal(targetResTyParams.getValue(), in.paramsOfStructTy),
        [&unifications](std::tuple<Attribute, Attribute> p) {
      Attribute fromCall = std::get<1>(p);
      // Preserve attributes that are already concrete at the call site. Otherwise attempt to lookup
      // non-parameterized concrete unification for the target struct parameter symbol.
      if (!isConcreteStructParamAttr(fromCall)) {
        Attribute fromTgt = std::get<0>(p);
        LLVM_DEBUG({
          llvm::dbgs() << "[instantiateViaTargetType]   fromCall = " << fromCall << '\n';
          llvm::dbgs() << "[instantiateViaTargetType]   fromTgt = " << fromTgt << '\n';
        });
        assert(llvm::isa<SymbolRefAttr>(fromTgt));
        auto it = unifications.find(std::make_pair(llvm::cast<SymbolRefAttr>(fromTgt), Side::RHS));
        if (it != unifications.end()) {
          Attribute unifiedAttr = it->second;
          LLVM_DEBUG({
            llvm::dbgs() << "[instantiateViaTargetType]   unifiedAttr = " << unifiedAttr << '\n';
          });
          if (unifiedAttr && isConcreteStructParamAttr(unifiedAttr, /*allowStructParams=*/false)) {
            return unifiedAttr;
          }
        }
      }
      return fromCall;
    }
    );

    out.paramsOfStructTy = std::move(newReturnStructParams);
    assert(out.paramsOfStructTy.size() == in.paramsOfStructTy.size() && "post-condition");
    assert(out.mapOpGroups.empty() && "post-condition");
    assert(out.dimsPerGroup.empty() && "post-condition");
    return success();
  }
};

/// Run affine-map and target-type instantiation over arrays and struct `compute` calls.
LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<
      InstantiateAtCreateArrayOp, // CreateArrayOp
      InstantiateAtCallOpCompute  // CallOp, targeting struct "compute()"
      >(ctx, tracker);

  return applyAndFoldGreedily(modOp, tracker, std::move(patterns));
}

} // namespace Step4_InstantiateAffineMaps

namespace Step6_PropagateTypes {

/// Update the array element type from compatible initializers and writes.
class UpdateNewArrayElemFromWrite final : public OpRewritePattern<CreateArrayOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the create-array element-type propagation pattern.
  UpdateNewArrayElemFromWrite(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Update an `array.new` result element type from compatible initializers and writes.
  LogicalResult matchAndRewrite(CreateArrayOp op, PatternRewriter &rewriter) const override {
    Value createResult = op.getResult();
    ArrayType createResultType = dyn_cast<ArrayType>(createResult.getType());
    assert(createResultType && "CreateArrayOp must produce ArrayType");
    Type oldResultElemType = createResultType.getElementType();

    Type newResultElemType = nullptr;
    // An initializer constrains array.new just as strongly as a later array.write: every element
    // must have the result array's element type. A member-read refinement can change an
    // initializer without creating a WriteArrayOp, so consider the explicit elements first.
    if (!op.getElements().empty()) {
      Type initializerType = op.getElements().front().getType();
      if (!llvm::all_of(op.getElements(), [initializerType](Value element) {
        return element.getType() == initializerType;
      })) {
        return failure();
      }
      if (initializerType != oldResultElemType) {
        newResultElemType = initializerType;
      }
    }

    // Look for WriteArrayOp where the array reference is the result of the CreateArrayOp and the
    // element type is different.
    for (Operation *user : createResult.getUsers()) {
      if (WriteArrayOp writeOp = dyn_cast<WriteArrayOp>(user)) {
        if (writeOp.getArrRef() != createResult) {
          continue;
        }
        Type writeRValueType = writeOp.getRvalue().getType();
        if (writeRValueType == oldResultElemType) {
          continue;
        }
        if (newResultElemType && newResultElemType != writeRValueType) {
          LLVM_DEBUG(
              llvm::dbgs()
              << "[UpdateNewArrayElemFromWrite] multiple possible element types for CreateArrayOp "
              << newResultElemType << " vs " << writeRValueType << '\n'
          );
          return failure();
        }
        newResultElemType = writeRValueType;
      }
    }
    if (!newResultElemType) {
      // no replacement type found
      return failure();
    }
    if (!tracker_.isLegalConversion(
            oldResultElemType, newResultElemType, "UpdateNewArrayElemFromWrite"
        )) {
      return failure();
    }
    ArrayType newType = createResultType.cloneWith(newResultElemType);
    SymbolTableCollection tables;
    if (!canRefineCreateArrayUsersToType(op, newType, tables)) {
      return failure();
    }
    rewriter.modifyOpInPlace(op, [&createResult, &newType]() { createResult.setType(newType); });
    LLVM_DEBUG(
        llvm::dbgs() << "[UpdateNewArrayElemFromWrite] updated result type of " << op << '\n'
    );
    return success();
  }
};

namespace {

/// Update the array reference type on an array access op to match a scalar element type observed
/// through that access.
LogicalResult updateArrayElemFromArrAccessOp(
    ArrayAccessOpInterface op, Type scalarElemTy, ConversionTracker &tracker,
    PatternRewriter &rewriter
) {
  ArrayType oldArrType = op.getArrRefType();
  if (oldArrType.getElementType() == scalarElemTy) {
    return failure(); // no change needed
  }
  ArrayType newArrType = oldArrType.cloneWith(scalarElemTy);
  if (oldArrType == newArrType ||
      !tracker.isLegalConversion(oldArrType, newArrType, "updateArrayElemFromArrAccessOp")) {
    return failure();
  }
  rewriter.modifyOpInPlace(op, [&op, &newArrType]() { op.getArrRef().setType(newArrType); });
  LLVM_DEBUG(
      llvm::dbgs() << "[updateArrayElemFromArrAccessOp] updated base array type in " << op << '\n'
  );
  return success();
}

} // namespace

class UpdateArrayElemFromArrWrite final : public OpRewritePattern<WriteArrayOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the array-write based element-type propagation pattern.
  UpdateArrayElemFromArrWrite(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Update the referenced array type from the write value type.
  LogicalResult matchAndRewrite(WriteArrayOp op, PatternRewriter &rewriter) const override {
    return updateArrayElemFromArrAccessOp(op, op.getRvalue().getType(), tracker_, rewriter);
  }
};

class UpdateArrayElemFromArrRead final : public OpRewritePattern<ReadArrayOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the array-read based element-type propagation pattern.
  UpdateArrayElemFromArrRead(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Update the referenced array type from the read result type.
  LogicalResult matchAndRewrite(ReadArrayOp op, PatternRewriter &rewriter) const override {
    return updateArrayElemFromArrAccessOp(op, op.getResult().getType(), tracker_, rewriter);
  }
};

/// Return whether a member type cannot be refined by writes. Concrete struct arguments still
/// need specialization, and wildcard array dimensions can become static even though
/// isConcreteType accepts their integer representation.
static bool isFullySpecializedMemberType(Type type) {
  if (!isConcreteType(type, /*allowStructParams=*/false)) {
    return false;
  }
  return !type.walk([](ArrayType array) {
    for (Attribute dim : array.getDimensionSizes()) {
      if (auto integer = dyn_cast<IntegerAttr>(dim); integer && isDynamic(integer)) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  }).wasInterrupted();
}

/// Update the type of MemberDefOp instances by checking the updated types from MemberWriteOp.
class UpdateMemberDefTypeFromWrite final : public OpRewritePattern<MemberDefOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the member-definition propagation pattern.
  UpdateMemberDefTypeFromWrite(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Update a member definition type from compatible writes to that member.
  LogicalResult matchAndRewrite(MemberDefOp op, PatternRewriter &rewriter) const override {
    // Avoid a whole-struct symbol-use search when writes cannot refine the declaration.
    if (isFullySpecializedMemberType(op.getType())) {
      return failure();
    }

    // Find all uses of the member symbol name within its parent struct.
    StructDefOp parentRes = getParentOfType<StructDefOp>(op);
    assert(parentRes && "MemberDefOp parent is always StructDefOp"); // per ODS def

    // If the symbol is used by a MemberWriteOp with a different result type then change
    // the type of the MemberDefOp to match the MemberWriteOp result type.
    Type newType = nullptr;
    if (auto memberUsers = llzk::getSymbolUses(op, parentRes)) {
      std::optional<Location> newTypeLoc = std::nullopt;
      for (SymbolTable::SymbolUse symUse : memberUsers.value()) {
        if (MemberWriteOp writeOp = llvm::dyn_cast<MemberWriteOp>(symUse.getUser())) {
          Type writeToType = writeOp.getVal().getType();
          LLVM_DEBUG(llvm::dbgs() << "[UpdateMemberDefTypeFromWrite] checking " << writeOp << '\n');
          if (!newType) {
            // If a new type has not yet been discovered, store the new type.
            newType = writeToType;
            newTypeLoc = writeOp.getLoc();
          } else if (writeToType != newType) {
            // Typically, there will only be one write for each member of a struct but do not rely
            // on that assumption. If multiple writes with a different types A and B are found where
            // A->B is a legal conversion (i.e., more concrete unification), then it is safe to use
            // type B with the assumption that the write with type A will be updated by another
            // pattern to also use type B.
            auto commonType = getCommonRefinedType(newType, writeToType, tracker_);
            if (succeeded(commonType)) {
              newType = *commonType;
              newTypeLoc = writeOp.getLoc();
            } else if (!tracker_.isLegalConversion(
                           writeToType, newType, "UpdateMemberDefTypeFromWrite"
                       )) {
              if (tracker_.isLegalConversion(
                      newType, writeToType, "UpdateMemberDefTypeFromWrite"
                  )) {
                // 'writeToType' is the more concrete type
                newType = writeToType;
                newTypeLoc = writeOp.getLoc();
              } else {
                // Give an error if the types are incompatible.
                return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
                  diag.append(
                      "Cannot update type of '", MemberDefOp::getOperationName(),
                      "' because there are multiple '", MemberWriteOp::getOperationName(),
                      "' with different value types"
                  );
                  if (newTypeLoc) {
                    diag.attachNote(newTypeLoc).append("type written here is ", newType);
                  }
                  diag.attachNote(writeOp.getLoc()).append("type written here is ", writeToType);
                });
              }
            }
          }
        }
      }
    }
    if (!newType || newType == op.getType()) {
      return failure(); // nothing changed
    }
    if (!tracker_.isLegalConversion(op.getType(), newType, "UpdateMemberDefTypeFromWrite")) {
      auto commonType = getCommonRefinedType(op.getType(), newType, tracker_);
      if (failed(commonType) || *commonType != newType) {
        return failure();
      }
    }

    // Infer shared template bindings from the positions of type variables, including those
    // inside arrays, POD records and struct parameters. Comparing the complete member types
    // would incorrectly reject members with different surrounding shapes.
    SmallVector<std::pair<TypeVarType, Type>> candidateBindings;
    if (!collectMemberTypeVarBindings(op.getType(), newType, tracker_, candidateBindings)) {
      return failure();
    }
    // Later iterations can instantiate callers already present in this module. Keep repeated
    // bindings correlated until those callers have been checked.
    if (refinesRepeatedMemberTypeVar(candidateBindings)) {
      return failure();
    }
    DenseMap<TypeVarType, Type> sharedBindings;
    for (auto [variable, binding] : candidateBindings) {
      // Unchanged variables are safe here because mixed repeated bindings were deferred above.
      if (variable == binding) {
        continue;
      }
      auto [it, inserted] = sharedBindings.try_emplace(variable, binding);
      if (inserted) {
        continue;
      }
      auto commonType = getCommonRefinedType(it->second, binding, tracker_);
      if (failed(commonType)) {
        return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
          diag.append("cannot refine ")
              .append(variable)
              .append(" because member '")
              .append(op.getName())
              .append("' is written with incompatible bindings ")
              .append(it->second)
              .append(" and ")
              .append(binding);
        });
      }
      it->second = *commonType;
    }
    if (!sharedBindings.empty()) {
      for (MemberDefOp sibling : parentRes.getOps<MemberDefOp>()) {
        if (sibling == op) {
          continue;
        }
        if (!referencesMemberTypeVar(sibling.getType(), &sharedBindings, tracker_)) {
          continue;
        }
        if (auto siblingUses = llzk::getSymbolUses(sibling, parentRes)) {
          for (SymbolTable::SymbolUse siblingUse : siblingUses.value()) {
            auto siblingWrite = llvm::dyn_cast<MemberWriteOp>(siblingUse.getUser());
            if (!siblingWrite) {
              continue;
            }
            SmallVector<std::pair<TypeVarType, Type>> siblingBindings;
            if (!collectMemberTypeVarBindings(
                    sibling.getType(), siblingWrite.getVal().getType(), tracker_, siblingBindings,
                    &sharedBindings
                )) {
              return failure();
            }
            for (auto [variable, binding] : siblingBindings) {
              auto it = sharedBindings.find(variable);
              if (it == sharedBindings.end()) {
                continue;
              }
              auto commonType = getCommonRefinedType(it->second, binding, tracker_);
              if (failed(commonType)) {
                return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
                  diag.append("cannot refine ")
                      .append(variable)
                      .append(" to ")
                      .append(it->second)
                      .append(" because sibling member '")
                      .append(sibling.getName())
                      .append("' is written with incompatible type ")
                      .append(binding);
                  diag.attachNote(siblingWrite.getLoc()).append("incompatible write is here");
                });
              }
              it->second = *commonType;
            }
          }
        }
      }
    }

    // Do not commit a write-derived refinement that an existing read or one of its typed consumers
    // cannot adopt. In particular, two complementary writes can have a common refinement even
    // when a generic read is independently consumed as an incompatible concrete type.
    if (auto memberUsers = llzk::getSymbolUses(op, parentRes)) {
      SymbolTableCollection tables;
      for (SymbolTable::SymbolUse symUse : memberUsers.value()) {
        if (MemberWriteOp writeOp = llvm::dyn_cast<MemberWriteOp>(symUse.getUser())) {
          // A read of another member carries that source's independent type evidence. Wait
          // for its own propagation to establish the candidate type before narrowing this
          // destination; otherwise a generic read can hide an incompatible source write.
          if (writeOp.getVal().getDefiningOp<MemberReadOp>() &&
              writeOp.getVal().getType() != newType) {
            return failure();
          }
        }
        if (MemberReadOp readOp = llvm::dyn_cast<MemberReadOp>(symUse.getUser())) {
          bool canUse = canUseScalarizedValueAsType(
              readOp.getVal().getType(), newType, tracker_, "UpdateMemberDefTypeFromWrite"
          );
          if (!canUse) {
            return failure();
          }
          bool canRefine = canRefineMemberReadUsersToType(readOp, newType, tables, tracker_);
          if (!canRefine) {
            return failure();
          }
        }
      }
    }
    rewriter.modifyOpInPlace(op, [&op, &newType]() { op.setType(newType); });
    LLVM_DEBUG(llvm::dbgs() << "[UpdateMemberDefTypeFromWrite] updated type of " << op << '\n');
    return success();
  }
};

namespace {

/// Move all regions out of `op` so it can be recreated with updated result types.
SmallVector<std::unique_ptr<Region>> moveRegions(Operation *op) {
  SmallVector<std::unique_ptr<Region>> newRegions;
  for (Region &region : op->getRegions()) {
    auto newRegion = std::make_unique<Region>();
    newRegion->takeBody(region);
    newRegions.push_back(std::move(newRegion));
  }
  return newRegions;
}

} // namespace

/// Updates the result type in Ops with the InferTypeOpAdaptor trait including ReadArrayOp,
/// ExtractArrayOp, etc.
class UpdateInferredResultTypes final : public OpTraitRewritePattern<OpTrait::InferTypeOpAdaptor> {
  ConversionTracker &tracker_;

public:
  /// Construct the inferred-result propagation pattern.
  UpdateInferredResultTypes(MLIRContext *ctx, ConversionTracker &tracker)
      : OpTraitRewritePattern(ctx, 6), tracker_(tracker) {}

  /// Re-infer result types and recreate the op when the inferred types are more concrete.
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    SmallVector<Type, 1> inferredResultTypes;
    InferTypeOpInterface retTypeFn = llvm::cast<InferTypeOpInterface>(op);
    LogicalResult result = retTypeFn.inferReturnTypes(
        op->getContext(), op->getLoc(), op->getOperands(), op->getRawDictionaryAttrs(),
        op->getPropertiesStorage(), op->getRegions(), inferredResultTypes
    );
    if (failed(result)) {
      return failure();
    }
    if (op->getResultTypes() == inferredResultTypes) {
      return failure(); // nothing changed
    }
    if (!tracker_.areLegalConversions(
            op->getResultTypes(), inferredResultTypes, "UpdateInferredResultTypes"
        )) {
      return failure();
    }

    // Move nested region bodies and replace the original op with the updated types list.
    LLVM_DEBUG(llvm::dbgs() << "[UpdateInferredResultTypes] replaced " << *op);
    SmallVector<std::unique_ptr<Region>> newRegions = moveRegions(op);
    Operation *newOp = rewriter.create(
        op->getLoc(), op->getName().getIdentifier(), op->getOperands(), inferredResultTypes,
        op->getAttrs(), op->getSuccessors(), newRegions
    );
    rewriter.replaceOp(op, newOp);
    LLVM_DEBUG(llvm::dbgs() << " with " << *newOp << '\n');
    return success();
  }
};

/// Update FuncDefOp return type by checking the updated types from ReturnOp.
class UpdateFuncTypeFromReturn final : public OpRewritePattern<FuncDefOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the function-return based type propagation pattern.
  UpdateFuncTypeFromReturn(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Update a function type from its terminator operand types.
  LogicalResult matchAndRewrite(FuncDefOp op, PatternRewriter &rewriter) const override {
    Region &body = op.getFunctionBody();
    if (body.empty()) {
      return failure();
    }
    ReturnOp retOp = llvm::dyn_cast<ReturnOp>(body.back().getTerminator());
    assert(retOp && "final op in body region must be return");
    OperandRange::type_range tyFromReturnOp = retOp.getOperands().getTypes();

    FunctionType oldFuncTy = op.getFunctionType();
    if (oldFuncTy.getResults() == tyFromReturnOp) {
      return failure(); // nothing changed
    }
    if (!tracker_.areLegalConversions(
            oldFuncTy.getResults(), tyFromReturnOp, "UpdateFuncTypeFromReturn"
        )) {
      return failure();
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op.setFunctionType(rewriter.getFunctionType(oldFuncTy.getInputs(), tyFromReturnOp));
    });
    LLVM_DEBUG(
        llvm::dbgs() << "[UpdateFuncTypeFromReturn] changed " << op.getSymName() << " from "
                     << oldFuncTy << " to " << op.getFunctionType() << '\n'
    );
    return success();
  }
};

/// Update CallOp result type based on the updated return type from the target FuncDefOp.
/// This only applies to free (i.e., non-struct) functions because the functions within structs
/// only return StructType or nothing and propagating those can result in bringing un-instantiated
/// types from a templated struct into the current call which will give errors.
class UpdateFreeFuncCallOpTypes final : public OpRewritePattern<CallOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the free-function call result propagation pattern.
  UpdateFreeFuncCallOpTypes(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Rewrite a call to a free function when the target function result types were refined.
  LogicalResult matchAndRewrite(CallOp op, PatternRewriter &rewriter) const override {
    if (calleeReferencesTemplateParam(op)) {
      return failure();
    }
    SymbolTableCollection tables;
    auto lookupRes = lookupTopLevelSymbol<FuncDefOp>(tables, op.getCalleeAttr(), op);
    if (failed(lookupRes)) {
      return failure();
    }
    FuncDefOp targetFunc = lookupRes->get();
    if (targetFunc.isInStruct()) {
      // this pattern only applies when the callee is NOT in a struct
      return failure();
    }
    if (op.getResultTypes() == targetFunc.getFunctionType().getResults()) {
      return failure(); // nothing changed
    }
    if (!tracker_.areLegalConversions(
            op.getResultTypes(), targetFunc.getFunctionType().getResults(),
            "UpdateFreeFuncCallOpTypes"
        )) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "[UpdateFreeFuncCallOpTypes] replaced " << op);
    CallOp newOp = replaceOpWithNewOp<CallOp>(rewriter, op, targetFunc, op.getArgOperands());
    (void)newOp; // tell compiler it's intentionally unused in release builds
    LLVM_DEBUG(llvm::dbgs() << " with " << newOp << '\n');
    return success();
  }
};

namespace {

/// Update a member read/write value type from the referenced member definition.
LogicalResult updateMemberRefValFromMemberDef(
    MemberRefOpInterface op, ConversionTracker &tracker, PatternRewriter &rewriter
) {
  SymbolTableCollection tables;
  auto def = op.getMemberDefOp(tables);
  if (failed(def)) {
    return failure();
  }
  Type oldResultType = op.getVal().getType();
  Type newResultType = def->get().getType();
  if (oldResultType == newResultType ||
      !canUseScalarizedValueAsType(
          oldResultType, newResultType, tracker, "updateMemberRefValFromMemberDef"
      )) {
    return failure();
  }
  if (MemberReadOp readOp = llvm::dyn_cast<MemberReadOp>(op.getOperation())) {
    if (!canRefineMemberReadUsersToType(readOp, newResultType, tables, tracker)) {
      return failure();
    }
  }
  rewriter.modifyOpInPlace(op, [&op, &newResultType]() { op.getVal().setType(newResultType); });
  LLVM_DEBUG(
      llvm::dbgs() << "[updateMemberRefValFromMemberDef] updated value type in " << op << '\n'
  );
  return success();
}

} // namespace

/// Update the type of MemberReadOp result based on updated types from MemberDefOp.
class UpdateMemberReadValFromDef final : public OpRewritePattern<MemberReadOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the member-read value propagation pattern.
  UpdateMemberReadValFromDef(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Update a member read result type from its referenced member definition.
  LogicalResult matchAndRewrite(MemberReadOp op, PatternRewriter &rewriter) const override {
    return updateMemberRefValFromMemberDef(op, tracker_, rewriter);
  }
};

namespace {

/// Return whether `value` is used by any operation other than `user`.
static bool hasUsesOutside(Value value, Operation *user) {
  return llvm::any_of(value.getUses(), [user](OpOperand &use) { return use.getOwner() != user; });
}

/// Return whether every use of `value` is a member-write value operand that accepts `type`.
///
/// A generic cast result can feed several member writes whose member definitions are refined
/// independently. Retagging that shared result is valid only when every target member accepts the
/// cast input type.
static bool allMemberWritesAcceptType(Value value, Type type) {
  if (value.use_empty()) {
    return false;
  }

  SymbolTableCollection tables;
  return llvm::all_of(value.getUses(), [&tables, type](OpOperand &use) {
    auto writeOp = llvm::dyn_cast<MemberWriteOp>(use.getOwner());
    if (!writeOp || writeOp.getVal() != use.get()) {
      return false;
    }
    auto memberDef = writeOp.getMemberDefOp(tables);
    return succeeded(memberDef) && typesUnify(type, memberDef->get().getType());
  });
}

/// Return a shared typed nondeterministic replacement for member writes that reuse `value`.
static FailureOr<Value> materializeValueForMemberWrite(
    Location loc, Value value, Type type, PatternRewriter &rewriter, ConversionTracker &tracker,
    DenseMap<Value, Value> &typedNondetReplacements
) {
  auto cachedReplacement = typedNondetReplacements.find(value);
  if (cachedReplacement != typedNondetReplacements.end()) {
    Value replacement = cachedReplacement->second;
    FailureOr<Type> commonType = getCommonRefinedType(replacement.getType(), type, tracker);
    if (failed(commonType)) {
      return failure();
    }
    if (*commonType != replacement.getType()) {
      NonDetOp replacementNondetOp = replacement.getDefiningOp<NonDetOp>();
      if (!replacementNondetOp) {
        return failure();
      }
      rewriter.modifyOpInPlace(replacementNondetOp, [&replacement, &commonType]() {
        replacement.setType(*commonType);
      });
    }
    return replacement;
  }
  if (value.getType() == type) {
    return value;
  }

  NonDetOp nondetOp = value.getDefiningOp<NonDetOp>();
  if (!nondetOp) {
    return failure();
  }

  // A direct member write can become concrete after another write refines the member definition.
  // Do not split a shared witness from a compatible external constraint while materializing the
  // concrete value for that write. This mirrors the retargeting performed for scalarized array
  // initializers: redirect the constraint to the instantiated struct definition and use the one
  // replacement witness for both observations.
  SymbolTableCollection tables;
  auto externalConstraints =
      collectSpecializedWitnessConstraints(value, type, tables, [](OpOperand &use) {
    auto memberWrite = dyn_cast<MemberWriteOp>(use.getOwner());
    return memberWrite && memberWrite.getVal() == use.get();
  });
  if (failed(externalConstraints)) {
    return failure();
  }

  OpBuilder::InsertionGuard guard(rewriter);
  Value replacement =
      materializeSpecializedNondet(rewriter, nondetOp, loc, type, *externalConstraints);
  typedNondetReplacements.try_emplace(value, replacement);
  return replacement;
}

/// Erase an original nondeterministic value after all member writes have moved to replacements.
static void eraseUnusedNondetDef(
    Value value, PatternRewriter &rewriter, DenseMap<Value, Value> &typedNondetReplacements
) {
  if (value.use_empty()) {
    if (NonDetOp nondetOp = value.getDefiningOp<NonDetOp>()) {
      typedNondetReplacements.erase(value);
      rewriter.eraseOp(nondetOp);
    }
  }
}

/// Replace a member-write value and remove an obsolete nondeterministic definition if applicable.
static inline void replaceMemberWriteValue(
    MemberWriteOp op, Value originalValue, Value replacement, PatternRewriter &rewriter,
    DenseMap<Value, Value> &typedNondetReplacements
) {
  rewriter.modifyOpInPlace(op, [&op, replacement]() { op.getValMutable().assign(replacement); });
  eraseUnusedNondetDef(originalValue, rewriter, typedNondetReplacements);
}

} // namespace

/// Update stale `poly.unifiable_cast` result types after their input has been instantiated.
class UpdateUnifiableCastResultFromInput final : public OpRewritePattern<UnifiableCastOp> {
  ConversionTracker &tracker_;

public:
  /// Construct the unifiable-cast result propagation pattern.
  UpdateUnifiableCastResultFromInput(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx, 3), tracker_(tracker) {}

  /// Retag a cast result when all remaining uses can observe the same instantiated type.
  LogicalResult matchAndRewrite(UnifiableCastOp op, PatternRewriter &rewriter) const override {
    Type inputType = op.getInput().getType();
    Value result = op.getResult();
    Type resultType = result.getType();
    if (typesUnify(inputType, resultType) || !allMemberWritesAcceptType(result, inputType)) {
      return failure();
    }
    bool canUse = canUseScalarizedValueAsType(
        resultType, inputType, tracker_, "UpdateUnifiableCastResultFromInput"
    );
    if (!canUse) {
      return failure();
    }
    rewriter.modifyOpInPlace(op, [&result, &inputType]() { result.setType(inputType); });
    LLVM_DEBUG(
        llvm::dbgs() << "[UpdateUnifiableCastResultFromInput] updated result type in " << op << '\n'
    );
    return success();
  }
};

/// Update the type of MemberWriteOp value based on updated types from MemberDefOp.
class UpdateMemberWriteValFromDef final : public OpRewritePattern<MemberWriteOp> {
  ConversionTracker &tracker_;
  ModuleOp rootModule_;
  bool &hadConflict_;
  mutable DenseMap<Value, Value> typedNondetReplacements_;

public:
  /// Construct the member-write value propagation pattern.
  UpdateMemberWriteValFromDef(
      MLIRContext *ctx, ConversionTracker &tracker, ModuleOp rootModule, bool &hadConflict
  )
      : OpRewritePattern(ctx, 3), tracker_(tracker), rootModule_(rootModule),
        hadConflict_(hadConflict) {}

  /// Update a member write operand type from its referenced member definition.
  LogicalResult matchAndRewrite(MemberWriteOp op, PatternRewriter &rewriter) const override {
    SymbolTableCollection tables;
    auto def = op.getMemberDefOp(tables);
    if (failed(def)) {
      return failure();
    }

    Value oldValue = op.getVal();
    Type newValueType = def->get().getType();
    auto cachedReplacement = typedNondetReplacements_.find(oldValue);
    if (hadConflict_) {
      return failure();
    }
    if (auto witness = oldValue.getDefiningOp<NonDetOp>();
        witness && (oldValue.getType() != newValueType ||
                    cachedReplacement != typedNondetReplacements_.end())) {
      Value extraValue =
          cachedReplacement == typedNondetReplacements_.end() ? Value() : cachedReplacement->second;
      if (failed(
              NonDetMemberTypeVerifier(rootModule_, tracker_).verifyComponent(witness, extraValue)
          )) {
        hadConflict_ = true;
        return failure();
      }
    }
    if (cachedReplacement != typedNondetReplacements_.end()) {
      FailureOr<Value> convertedValue = materializeValueForMemberWrite(
          op.getLoc(), oldValue, newValueType, rewriter, tracker_, typedNondetReplacements_
      );
      if (failed(convertedValue)) {
        return failure();
      }
      replaceMemberWriteValue(op, oldValue, *convertedValue, rewriter, typedNondetReplacements_);
      LLVM_DEBUG(
          llvm::dbgs() << "[UpdateMemberWriteValFromDef] reused materialized value type for " << op
                       << '\n'
      );
      return success();
    }

    Type oldValueType = oldValue.getType();
    if (oldValueType == newValueType ||
        !tracker_.isLegalConversion(oldValueType, newValueType, "UpdateMemberWriteValFromDef")) {
      return failure();
    }

    // A read derives its type from its source member, whose writes can carry independent
    // concrete dimensions. Let UpdateMemberReadValFromDef refine it from that declaration;
    // a destination write must not impose its own type on the source read.
    if (oldValue.getDefiningOp<MemberReadOp>()) {
      return failure();
    }

    if (!hasUsesOutside(oldValue, op.getOperation())) {
      rewriter.modifyOpInPlace(op, [&oldValue, &newValueType]() {
        oldValue.setType(newValueType);
      });
      LLVM_DEBUG(
          llvm::dbgs() << "[UpdateMemberWriteValFromDef] updated value type in " << op << '\n'
      );
      return success();
    }

    FailureOr<Value> convertedValue = materializeValueForMemberWrite(
        op.getLoc(), oldValue, newValueType, rewriter, tracker_, typedNondetReplacements_
    );
    if (failed(convertedValue)) {
      return failure();
    }
    replaceMemberWriteValue(op, oldValue, *convertedValue, rewriter, typedNondetReplacements_);
    LLVM_DEBUG(
        llvm::dbgs() << "[UpdateMemberWriteValFromDef] materialized value type for " << op << '\n'
    );
    return success();
  }
};

/// Run all type-propagation patterns to a local fixpoint for the current iteration.
LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  if (failed(NonDetMemberTypeVerifier(modOp, tracker).verifyModule())) {
    return failure();
  }
  bool hadConflict = false;
  RewritePatternSet patterns(ctx);
  patterns.add<
      // Benefit of this one must be higher than rules that would propagate the type in the opposite
      // direction (ex: `UpdateArrayElemFromArrRead`) else the greedy conversion would not converge.
      //  benefit = 6
      UpdateInferredResultTypes, // OpTrait::InferTypeOpAdaptor (ReadArrayOp, ExtractArrayOp)
      //  benefit = 3
      UpdateFreeFuncCallOpTypes,         // CallOp, targeting non-struct functions
      UpdateFuncTypeFromReturn,          // FuncDefOp
      UpdateNewArrayElemFromWrite,       // CreateArrayOp
      UpdateArrayElemFromArrRead,        // ReadArrayOp
      UpdateArrayElemFromArrWrite,       // WriteArrayOp
      UpdateMemberDefTypeFromWrite,      // MemberDefOp
      UpdateMemberReadValFromDef,        // MemberReadOp
      UpdateUnifiableCastResultFromInput // UnifiableCastOp
      >(ctx, tracker);
  patterns.add<UpdateMemberWriteValFromDef>(ctx, tracker, modOp, hadConflict);

  LogicalResult result = applyAndFoldGreedily(modOp, tracker, std::move(patterns));
  return failure(failed(result) || hadConflict);
}
} // namespace Step6_PropagateTypes

namespace Step7_Cleanup {

/// Cleanup strategy that preserves symbols reachable from an explicit keep set plus globals.
struct FromKeepSet : public CleanupBase {
  using CleanupBase::CleanupBase;

  /// Return `true` iff the given free function or struct definition still has unresolved template
  /// symbol bindings.
  static bool hasTemplateSymbolBindings(Operation *op) {
    if (StructDefOp sdef = llvm::dyn_cast<StructDefOp>(op)) {
      return sdef.hasTemplateSymbolBindings();
    }
    if (llvm::isa<function::FuncDefOp>(op)) {
      if (TemplateOp parent = getParentOfType<TemplateOp>(op)) {
        return parent.hasConstOps<TemplateSymbolBindingOpInterface>();
      }
    }
    return false;
  }

  /// Erase all cleanup-candidate definitions that are not reachable (via calls, types, or symbol
  /// usage) from one of the given roots or from some global def (since this pass does not remove
  /// global definitions, any symbols reachable from them must not be removed).
  LogicalResult eraseUnreachableFrom(ArrayRef<SymbolOpInterface> keep) {
    // Initialize roots from the given symbol definitions.
    SetVector<SymbolOpInterface> roots(keep.begin(), keep.end());
    // Add GlobalDefOp to the set of roots.
    rootMod.walk([&roots](global::GlobalDefOp gdef) { roots.insert(gdef); });

    // Use a SymbolDefTree to find all Symbol defs reachable from one of the root nodes. Then
    // collect all Symbol uses reachable from those def nodes. These are the symbols that should
    // be preserved. All other symbol defs should be removed.
    DenseSet<Operation *> defsToKeep;
    llvm::df_iterator_default_set<const SymbolUseGraphNode *> symbolsToKeep;
    for (size_t i = 0; i < roots.size(); ++i) { // iterate for safe insertion
      SymbolOpInterface keepRoot = roots[i];
      LLVM_DEBUG({ llvm::dbgs() << "[EraseUnreachable] root: " << keepRoot << '\n'; });
      const SymbolDefTreeNode *keepRootNode = defTree.lookupNode(keepRoot);
      assert(keepRootNode && "every symbol def must be in the def tree");
      for (const SymbolDefTreeNode *reachableDefNode : llvm::depth_first(keepRootNode)) {
        LLVM_DEBUG({
          llvm::dbgs() << "[EraseUnreachable] can reach: " << reachableDefNode->getOp() << '\n';
        });
        if (SymbolOpInterface reachableDef = reachableDefNode->getOp()) {
          if (isErasableDefinition(reachableDef.getOperation())) {
            defsToKeep.insert(reachableDef.getOperation());
          }
          // Use 'depth_first_ext()' to get all symbol uses reachable from the current Symbol def
          // node. There are no uses if the node is not in the graph. Within the loop that populates
          // 'depth_first_ext()', also check if the symbol is an erasable definition and ensure it
          // is in 'roots' so the outer loop preserves all symbols reachable from it.
          if (const SymbolUseGraphNode *useGraphNodeForDef = useGraph.lookupNode(reachableDef)) {
            for (const SymbolUseGraphNode *usedSymbolNode :
                 depth_first_ext(useGraphNodeForDef, symbolsToKeep)) {
              LLVM_DEBUG({
                llvm::dbgs() << "[EraseUnreachable]   uses symbol: "
                             << usedSymbolNode->getSymbolPath() << '\n';
              });
              // Ignore struct/template parameter symbols (before doing the lookup below because it
              // would fail anyway and then cause the "failed" case to be triggered unnecessarily).
              if (usedSymbolNode->isTemplateSymbolBinding()) {
                continue;
              }
              // If `usedSymbolNode` references an erasable definition, ensure it's considered in
              // the roots so symbols reachable from its body are preserved too.
              auto lookupRes = usedSymbolNode->lookupSymbol(tables);
              if (failed(lookupRes)) {
                LLVM_DEBUG(useGraph.dumpToDotFile());
                return failure();
              }
              //  If loaded via an IncludeOp it's not in the current AST anyway so ignore.
              if (lookupRes->viaInclude()) {
                continue;
              }
              Operation *usedOp = lookupRes->get();
              if (isErasableDefinition(usedOp)) {
                SymbolOpInterface asSymbol = llvm::cast<SymbolOpInterface>(usedOp);
                bool insertRes = roots.insert(asSymbol);
                (void)insertRes; // tell compiler it's intentionally unused in release builds
                LLVM_DEBUG({
                  if (insertRes) {
                    llvm::dbgs() << "[EraseUnreachable]  found another root: " << asSymbol << '\n';
                  }
                });
              }
            }
          }
        }
      }
    }

    SmallVector<SymbolOpInterface> toErase;
    rootMod.walk([this, &defsToKeep, &symbolsToKeep, &toErase](Operation *op) {
      if (!isErasableDefinition(op) || defsToKeep.contains(op)) {
        return;
      }
      SymbolOpInterface symOp = llvm::cast<SymbolOpInterface>(op);
      const SymbolUseGraphNode *n = this->useGraph.lookupNode(symOp);
      if (!n || !symbolsToKeep.contains(n)) {
        LLVM_DEBUG(llvm::dbgs() << "[EraseUnreachable] removing: " << symOp.getNameAttr() << '\n');
        toErase.push_back(symOp);
      }
    });
    for (SymbolOpInterface symOp : toErase) {
      symOp.erase();
    }

    return success();
  }
};

} // namespace Step7_Cleanup

class PassImpl : public llzk::polymorphic::impl::FlatteningPassBase<PassImpl> {
  using Base = FlatteningPassBase<PassImpl>;
  using Base::Base;

  /// If the cleanup mode is unspecified, default to `Preimage`.
  FlatteningCleanupMode getEffectiveCleanupMode() const {
    FlatteningCleanupMode m = cleanupMode.getValue();
    return m == FlatteningCleanupMode::Unspecified ? FlatteningCleanupMode::Preimage : m;
  }

  /// Run the pass on the current module and signal failure if any flattening step fails.
  void runOnOperation() override {
    ModuleOp modOp = getOperation();
    if (failed(runOn(modOp))) {
      LLVM_DEBUG({
        // If the pass failed, dump the current IR.
        llvm::dbgs() << "=====================================================================\n";
        llvm::dbgs() << " Dumping module after failure of pass " << DEBUG_TYPE << '\n';
        modOp.print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
        llvm::dbgs() << "=====================================================================\n";
      });
      signalPassFailure();
    }
  }

  /// Execute the full flattening pipeline until it reaches a fixpoint or the iteration limit.
  inline LogicalResult runOn(ModuleOp modOp) {
    FlatteningCleanupMode effectiveCleanupMode = getEffectiveCleanupMode();
    // If the cleanup mode is set to remove anything not reachable from the main struct, do an
    // initial pass to remove things that are not reachable (as an optimization) because creating
    // an instantiated version of a struct will not cause something to become reachable that was
    // not already reachable in parameterized form.
    if (effectiveCleanupMode == FlatteningCleanupMode::MainAsRoot) {
      if (failed(eraseUnreachableFromMainStruct(modOp))) {
        return failure();
      }
    }

    // Pass Manager to run some standard cleanup passes that are always beneficial:
    // - Remove templates that contain no struct or function definitions
    // - Convert templates with no constant parameters or expressions into modules
    OpPassManager universalCleanup(ModuleOp::getOperationName());
    universalCleanup.addPass(createEmptyTemplateRemovalPass());

    // Run universal cleanup as a preliminary step to satisfy the
    // `assert(!isNullOrEmpty(paramNames))` precondition in `genClone()`.
    if (failed(runPipeline(universalCleanup, modOp))) {
      return failure();
    }

    ConversionTracker tracker(unrollIterationLimit);
    if (failed(Step1_InstantiateStructs::instantiateMainStruct(modOp, tracker))) {
      llvm::errs() << DEBUG_TYPE << " failed while instantiating the main struct\n";
      return failure();
    }

    unsigned loopCount = 0;
    do {
      ++loopCount;
      if (loopCount > iterationLimit) {
        llvm::errs() << DEBUG_TYPE << " exceeded the limit of " << iterationLimit
                     << " iterations!\n";
        return failure();
      }
      tracker.resetModifiedFlag();

      LLVM_DEBUG({
        llvm::dbgs() << "[FlatteningPass(count=" << loopCount
                     << ")] Running step 1: struct instantiation\n";
      });
      // Find calls to "compute()" that return a parameterized struct type and replace it to call an
      // instantiated version of the struct that has parameters replaced with the constant values.
      // Create the necessary instantiated/flattened struct in the same location as the original.
      if (failed(Step1_InstantiateStructs::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while instantiating structs in templates\n";
        return failure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "[FlatteningPass(count=" << loopCount
                     << ")] Running step 2: function instantiation\n";
      });
      // Instantiate calls to templated functions.
      if (failed(Step2_InstantiateFunctions::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while instantiating functions in templates\n";
        return failure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "[FlatteningPass(count=" << loopCount
                     << ")] Running step 3: loop unrolling\n";
      });
      // Unroll loops with known iterations.
      if (failed(Step3_Unroll::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while unrolling loops\n";
        return failure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "[FlatteningPass(count=" << loopCount
                     << ")] Running step 4: affine maps instantiation\n";
      });
      // Instantiate affine_map parameters of StructType and ArrayType.
      if (failed(Step4_InstantiateAffineMaps::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while instantiating `affine_map` parameters\n";
        return failure();
      }

      // Split static arrays whose affine-map element type instantiates to different concrete
      // element types at different indices.
      LLVM_DEBUG({
        llvm::dbgs() << "[FlatteningPass(count=" << loopCount
                     << ")] Running step 5: heterogeneous array scalarization\n";
      });
      if (failed(scalarizeHeterogeneousArrays(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while scalarizing heterogeneous arrays\n";
        return failure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "[FlatteningPass(count=" << loopCount
                     << ")] Running step 6: type propagation\n";
      });
      // Propagate updated types using the semantics of various ops.
      if (failed(Step6_PropagateTypes::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while propagating instantiated types\n";
        return failure();
      }

      LLVM_DEBUG(if (tracker.isModified()) {
        llvm::dbgs() << "=====================================================================\n";
        llvm::dbgs() << " Dumping module between iterations of " << DEBUG_TYPE << '\n';
        modOp.print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
        llvm::dbgs() << "=====================================================================\n";
      });
    } while (tracker.isModified());

    tracker.clearPartialFuncInstantiations();

    // Run user-selected cleanup first.
    if (failed(cleanupSwitch(modOp, tracker))) {
      return failure();
    }
    // Run universal cleanup again since no-param or param-only structs may exist now.
    if (failed(runPipeline(universalCleanup, modOp))) {
      return failure();
    }

    OpPassManager allocationCleanup(ModuleOp::getOperationName());
    allocationCleanup.addPass(createRemoveUnusedDiscardableAllocationsPass(
        RemoveUnusedDiscardableAllocationsPassOptions {
            .allocatorOpName = CreateArrayOp::getOperationName().str()
        }
    ));
    return runPipeline(allocationCleanup, modOp);
  }

  /// Perform cleanup according to the effective `cleanupMode` option.
  LogicalResult cleanupSwitch(ModuleOp modOp, const ConversionTracker &tracker) {
    FlatteningCleanupMode effectiveCleanupMode = getEffectiveCleanupMode();
    LLVM_DEBUG({ llvm::dbgs() << "[FlatteningPass] Running step 7: cleanup "; });
    switch (effectiveCleanupMode) {
    case FlatteningCleanupMode::MainAsRoot:
      LLVM_DEBUG(llvm::dbgs() << "(main as root mode)\n");
      return eraseUnreachableFromMainStruct(modOp, false);
    case FlatteningCleanupMode::ConcreteAsRoot:
      LLVM_DEBUG(llvm::dbgs() << "(concrete definitions mode)\n");
      return eraseUnreachableFromConcreteDefinitions(modOp);
    case FlatteningCleanupMode::Preimage:
      LLVM_DEBUG(llvm::dbgs() << "(preimage mode)\n");
      return erasePreimageOfInstantiations(modOp, tracker);
    case FlatteningCleanupMode::Disabled:
      LLVM_DEBUG(llvm::dbgs() << "(disabled)\n");
      return success();
    case FlatteningCleanupMode::Unspecified:
      llvm_unreachable("`getEffectiveCleanupMode()` cannot give `Unspecified`");
    }
    llvm_unreachable("unknown cleanup mode");
  }

  /// Erase parameterized definitions that were replaced with concrete instantiations.
  LogicalResult erasePreimageOfInstantiations(ModuleOp rootMod, const ConversionTracker &tracker) {
    // TODO: The names from getInstantiatedDefinitionNames() are NOT guaranteed to be paths from the
    // "top root" and they also do not indicate a root module so there could be ambiguity. This is a
    // broader problem in the FlatteningPass itself so let's just assume, for now, that these are
    // paths from the "top root". See [LLZK-286].
    FromEraseSet cleaner(
        rootMod, getAnalysis<SymbolDefTree>(), getAnalysis<SymbolUseGraph>(),
        tracker.getInstantiatedDefinitionNames()
    );
    LogicalResult res = cleaner.eraseUnusedDefinitions();
    if (succeeded(res)) {
      LLVM_DEBUG(llvm::dbgs() << "[Cleanup(preimage)] success\n";);
      // Warn about any definitions that were instantiated but still have uses elsewhere.
      const SymbolUseGraph *useGraph = nullptr;
      rootMod->walk([this, &cleaner, &useGraph](Operation *walkedOp) {
        SymbolOpInterface op = llvm::dyn_cast<SymbolOpInterface>(walkedOp);
        if (!op || !cleaner.getTryToEraseSet().contains(op)) {
          return;
        }
        // If needed, rebuild use graph to reflect deletions.
        if (!useGraph) {
          useGraph = &getAnalysis<SymbolUseGraph>();
        }
        // If the op has any users, report the warning.
        if (useGraph->lookupNode(op)->hasPredecessor()) {
          op.emitWarning("Parameterized definition still has uses!").report();
        }
      });
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[Cleanup(preimage)] failed\n";);
    }
    return res;
  }

  /// Erase cleanup candidates that are unreachable from any concrete definition or global.
  LogicalResult eraseUnreachableFromConcreteDefinitions(ModuleOp rootMod) {
    SmallVector<SymbolOpInterface> roots;
    rootMod.walk([&roots](Operation *op) {
      if (isErasableDefinition(op) && !Step7_Cleanup::FromKeepSet::hasTemplateSymbolBindings(op)) {
        roots.push_back(llvm::cast<SymbolOpInterface>(op));
      }
    });

    Step7_Cleanup::FromKeepSet cleaner(
        rootMod, getAnalysis<SymbolDefTree>(), getAnalysis<SymbolUseGraph>()
    );
    return cleaner.eraseUnreachableFrom(roots);
  }

  /// Erase cleanup candidates that are unreachable from the `llzk.main` struct or globals.
  LogicalResult eraseUnreachableFromMainStruct(ModuleOp rootMod, bool emitWarning = true) {
    Step7_Cleanup::FromKeepSet cleaner(
        rootMod, getAnalysis<SymbolDefTree>(), getAnalysis<SymbolUseGraph>()
    );
    FailureOr<SymbolLookupResult<StructDefOp>> mainOpt =
        getMainInstanceDef(cleaner.tables, rootMod.getOperation());
    if (failed(mainOpt)) {
      return failure();
    }
    SymbolLookupResult<StructDefOp> main = mainOpt.value();
    if (emitWarning && !main) {
      // Emit warning if there is no main specified because all cleanup-candidate definitions not
      // reachable from global defs may be removed.
      rootMod.emitWarning()
          .append(
              "using option '", cleanupMode.getArgStr(), '=',
              stringifyFlatteningCleanupMode(FlatteningCleanupMode::MainAsRoot), "' with no \"",
              MAIN_ATTR_NAME,
              "\" attribute on the top-level module may remove all cleanup-candidate definitions!"
          )
          .report();
    }
    SmallVector<SymbolOpInterface> roots;
    if (main) {
      roots.push_back(*main);
    }
    return cleaner.eraseUnreachableFrom(roots);
  }
};

} // namespace
