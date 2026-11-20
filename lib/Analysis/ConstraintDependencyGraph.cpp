//===-- ConstraintDependencyGraph.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/ConstraintDependencyGraph.h"

#include "llzk/Analysis/SourceRefLattice.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/Util/ArrayTypeHelper.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/Hash.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/Support/Debug.h>

#include <algorithm>
#include <map>
#include <numeric>

#define DEBUG_TYPE "llzk-cdg"

using namespace mlir;

namespace llzk {

using namespace array;
using namespace component;
using namespace constrain;
using namespace function;
using namespace pod;

namespace {

static bool isInMaybeSkippedScfRegion(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent != nullptr; parent = parent->getParentOp()) {
    if (llvm::isa<FuncDefOp>(parent)) {
      return false;
    }
    if (llvm::isa<
            scf::ForOp, scf::IfOp, scf::WhileOp, scf::IndexSwitchOp, scf::ParallelOp,
            scf::ForallOp>(parent)) {
      return true;
    }
  }
  return false;
}

/// Return the function whose storage effects can influence this operation.
/// Operations outside functions share the top-level storage scope.
static Operation *getStorageScope(Operation *op) {
  if (auto function = op->getParentOfType<FuncDefOp>()) {
    return function.getOperation();
  }
  while (op->getParentOp() != nullptr) {
    op = op->getParentOp();
  }
  return op;
}

/// Return whether `address` is rooted at storage allocated anew within `loop`.
/// Such storage cannot carry a write from a previous iteration to the current one.
static bool isAllocatedWithinLoop(const SourceRef &address, Operation *loop) {
  auto root = address.getRoot();
  if (failed(root)) {
    return false;
  }
  auto result = llvm::dyn_cast<OpResult>(*root);
  if (!result || !llvm::isa<CreateArrayOp, NewPodOp, CallOp, ReadPodOp, ReadArrayOp, MemberReadOp>(
                     result.getOwner()
                 )) {
    return false;
  }
  return loop->isAncestor(result.getOwner());
}

/// Mutable aggregates are observed at their use; scalar SSA values retain their definition.
static bool isAggregateType(Type type) { return llvm::isa<ArrayType, StructType, PodType>(type); }

/// Return whether a value can carry shared component identity, including through value aggregates.
static bool containsComponentHandle(Type type) {
  // An unresolved type variable may be instantiated with a component handle or an aggregate
  // containing one, so its storage effects must also be treated conservatively.
  if (llvm::isa<StructType, polymorphic::TypeVarType>(type)) {
    return true;
  }
  if (auto array = llvm::dyn_cast<ArrayType>(type)) {
    return containsComponentHandle(array.getElementType());
  }
  if (auto pod = llvm::dyn_cast<PodType>(type)) {
    return llvm::any_of(pod.getRecords(), [](auto record) {
      return containsComponentHandle(record.getType());
    });
  }
  return false;
}

/// PODs and arrays carry independently mutable payload; component values are identity handles.
static bool isMutableAggregateType(Type type) { return llvm::isa<ArrayType, PodType>(type); }

static inline bool hasRangeIndex(const SourceRef &ref) {
  return llvm::any_of(ref.getPath(), [](const SourceRefIndex &index) {
    return index.isIndexRange();
  });
}

/// Return `ref` with one path component replaced, preserving the suffix after it.
static FailureOr<SourceRef>
replacePathIndex(const SourceRef &ref, size_t index, const SourceRefIndex &replacement) {
  ensure(index < ref.getPath().size(), "SourceRef path index is out of bounds");
  SourceRef result = ref;
  for (size_t i = index; i < ref.getPath().size(); ++i) {
    auto parent = result.getParentPrefix();
    ensure(succeeded(parent), "could not get SourceRef parent while replacing path index");
    result = *parent;
  }
  auto child = result.createChild(replacement);
  ensure(succeeded(child), "could not create SourceRef replacement path index");
  result = *child;
  for (const SourceRefIndex &suffix : ref.getPath().drop_front(index + 1)) {
    child = result.createChild(suffix);
    ensure(succeeded(child), "could not restore SourceRef suffix after replacing path index");
    result = *child;
  }
  return result;
}

/// Intersect two overlapping selections, retaining a concrete index when either is concrete.
static SourceRefIndex intersectIndex(const SourceRefIndex &lhs, const SourceRefIndex &rhs) {
  ensure(lhs.overlaps(rhs), "index intersection requires overlapping selections");
  if (!lhs.isIndexRange()) {
    return lhs;
  }
  if (!rhs.isIndexRange()) {
    return rhs;
  }
  auto [lhsLow, lhsHigh] = lhs.getIndexRange();
  auto [rhsLow, rhsHigh] = rhs.getIndexRange();
  auto high = lhs.hasUnboundedUpperBound()   ? rhsHigh
              : rhs.hasUnboundedUpperBound() ? lhsHigh
                                             : std::min(lhsHigh, rhsHigh);
  return SourceRefIndex(std::max(lhsLow, rhsLow), high);
}

/// Return disjoint slabs of `rangeAddress` outside an overlapping point or rectangle.
/// Earlier dimensions are restricted to the intersection before splitting the next dimension.
static SmallVector<SourceRef>
subtractAddress(const SourceRef &rangeAddress, const SourceRef &removedAddress) {
  ensure(rangeAddress.overlaps(removedAddress), "addresses must overlap before subtraction");

  SmallVector<SourceRef> result;
  const auto rangePath = rangeAddress.getPath();
  const auto removedPath = removedAddress.getPath();
  SourceRef intersection = rangeAddress;
  for (auto [index, rangeIndex] : llvm::enumerate(rangePath)) {
    if (!rangeIndex.isIndexRange()) {
      continue;
    }
    SourceRefIndex common = intersectIndex(rangeIndex, removedPath[index]);
    const auto [low, high] = rangeIndex.getIndexRange();
    auto [commonLow, commonHigh] = common.isIndexRange()
                                       ? common.getIndexRange()
                                       : std::make_pair(common.getIndex(), common.getIndex() + 1);
    if (low < commonLow) {
      result.push_back(*replacePathIndex(intersection, index, SourceRefIndex(low, commonLow)));
    }
    if (!common.hasUnboundedUpperBound() &&
        (rangeIndex.hasUnboundedUpperBound() || commonHigh < high)) {
      result.push_back(*replacePathIndex(intersection, index, SourceRefIndex(commonHigh, high)));
    }
    intersection = *replacePathIndex(intersection, index, common);
  }
  return result;
}

static bool isNonSingletonArrayWriteTarget(const SourceRefLatticeValue &writeTargets) {
  if (writeTargets.isArray()) {
    return llvm::any_of(
        llvm::seq<size_t>(0, writeTargets.getArraySize()), [&writeTargets](size_t i) {
      return isNonSingletonArrayWriteTarget(writeTargets.getElemFlatIdx(i));
    }
    );
  }
  return !writeTargets.isSingleValue() ||
         llvm::any_of(writeTargets.getScalarValue(), hasRangeIndex);
}

/// Return the point address of a statically shaped array element at `flatIndex`.
static SourceRef getArrayElementAddress(const SourceRef &root, size_t flatIndex) {
  auto arrayType = llvm::dyn_cast<ArrayType>(root.getType());
  ensure(arrayType && arrayType.hasStaticShape(), "array element address requires static shape");
  ArrayIndexGen indexGen = ArrayIndexGen::from(arrayType);
  auto indices = indexGen.delinearize(checkedCast<int64_t>(flatIndex), root.getType().getContext());
  ensure(indices.has_value(), "could not delinearize array element address");
  SourceRef address = root;
  for (Attribute attr : *indices) {
    auto child =
        address.createChild(SourceRefIndex(DynamicAPInt(llvm::cast<IntegerAttr>(attr).getValue())));
    ensure(succeeded(child), "could not create array element address");
    address = *child;
  }
  return address;
}

/// Infer the half-open index bounds of a statically bounded structured loop.
static std::optional<std::pair<DynamicAPInt, DynamicAPInt>> getStaticLoopIndexRange(Value index) {
  if (auto toIndex = index.getDefiningOp<cast::FeltToIndexOp>()) {
    index = toIndex.getValue();
  }
  auto blockArg = llvm::dyn_cast<BlockArgument>(index);
  if (!blockArg) {
    return std::nullopt;
  }

  if (auto forOp = llvm::dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
    if (blockArg != forOp.getInductionVar()) {
      return std::nullopt;
    }
    auto lower = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto upper = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    if (lower && upper) {
      return std::pair(DynamicAPInt(lower.value()), DynamicAPInt(upper.value()));
    }
    return std::nullopt;
  }

  auto whileOp = llvm::dyn_cast<scf::WhileOp>(blockArg.getOwner()->getParentOp());
  if (!whileOp || blockArg.getOwner() != &whileOp.getAfter().front()) {
    return std::nullopt;
  }
  unsigned argumentNumber = blockArg.getArgNumber();
  if (argumentNumber >= whileOp.getBeforeArguments().size() ||
      argumentNumber >= whileOp.getInits().size()) {
    return std::nullopt;
  }

  auto lower = whileOp.getInits()[argumentNumber].getDefiningOp<felt::FeltConstantOp>();
  auto cmp = whileOp.getConditionOp().getCondition().getDefiningOp<boolean::CmpOp>();
  Value beforeArgument = whileOp.getBeforeArguments()[argumentNumber];
  if (!lower || !cmp || cmp.getPredicate() != boolean::FeltCmpPredicate::LT ||
      cmp.getLhs() != beforeArgument ||
      whileOp.getConditionOp().getArgs()[argumentNumber] != beforeArgument) {
    return std::nullopt;
  }
  auto upper = cmp.getRhs().getDefiningOp<felt::FeltConstantOp>();
  if (!upper) {
    return std::nullopt;
  }
  auto upperBound = upper.getValue().getReducedValue();
  if (!upperBound) {
    // A fieldless literal is not a bound until its field representative is known.
    return std::nullopt;
  }
  // The backedge need not be monotone. The condition proves an upper bound, while a felt's
  // representative is nonnegative regardless of the initializer or recurrence.
  return std::pair(DynamicAPInt(0), *upperBound);
}

/// Select constant indices, known loop ranges, or whole dimensions from operand states.
static std::vector<SourceRefIndex> getArrayAccessIndices(
    ArrayAccessOpInterface access,
    llvm::function_ref<const SourceRefLatticeValue &(Value)> getIndexState
) {
  std::vector<SourceRefIndex> indices;
  ArrayType arrayType = access.getArrRef().getType();
  for (auto [i, index] : llvm::enumerate(access.getIndices())) {
    int64_t dimSize = arrayType.getDimSize(llzk::checkedCast<unsigned>(i));
    auto constantIndex = getIndexState(index).getConstantValue();
    if (succeeded(constantIndex)) {
      indices.emplace_back(*constantIndex);
    } else if (
        auto bounds = getStaticLoopIndexRange(index); bounds && !ShapedType::isDynamic(dimSize)
    ) {
      // A guarded access may be valid even when the loop exceeds the array's bounds.
      auto lower = std::clamp(bounds->first, DynamicAPInt(0), DynamicAPInt(dimSize));
      auto upper = std::clamp(bounds->second, lower, DynamicAPInt(dimSize));
      indices.emplace_back(lower, upper);
    } else {
      indices.push_back(SourceRefIndex::forArrayDimension(dimSize));
    }
  }
  return indices;
}

/// Recognize a complete elementwise copy with an unconditional write and a unit-step induction.
/// More general loops retain the existing conservative storage dependencies.
static WriteArrayOp getFullyOverwrittenLoopArrayWrite(Value value) {
  auto read = value.getDefiningOp<ReadArrayOp>();
  auto whileResult = read ? llvm::dyn_cast<OpResult>(read.getArrRef()) : OpResult();
  auto whileOp = whileResult ? whileResult.getDefiningOp<scf::WhileOp>() : scf::WhileOp();
  if (!whileOp || whileResult.getResultNumber() >= whileOp.getInits().size()) {
    return nullptr;
  }

  Value initializer = whileOp.getInits()[whileResult.getResultNumber()];
  auto arrayType = llvm::dyn_cast<ArrayType>(initializer.getType());
  if (!initializer.getDefiningOp<NonDetOp>() || !arrayType || !arrayType.hasStaticShape() ||
      arrayType.getShape().size() != 1) {
    return nullptr;
  }

  unsigned resultNumber = whileResult.getResultNumber();
  Value carried = whileOp.getAfter().front().getArgument(whileResult.getResultNumber());
  if (whileOp.getConditionOp().getArgs()[resultNumber] !=
          whileOp.getBeforeArguments()[resultNumber] ||
      whileOp.getYieldOp().getOperand(resultNumber) != carried) {
    return nullptr;
  }
  WriteArrayOp coveringWrite;
  for (auto write : whileOp.getAfter().front().getOps<WriteArrayOp>()) {
    if (write.getArrRef() != carried || write.getIndices().size() != 1 || coveringWrite) {
      return nullptr;
    }
    auto range = getStaticLoopIndexRange(write.getIndices().front());
    auto indexCast = write.getIndices().front().getDefiningOp<cast::FeltToIndexOp>();
    auto index = indexCast ? llvm::dyn_cast<BlockArgument>(indexCast.getValue()) : BlockArgument();
    if (!range || range->first != 0 || range->second != arrayType.getDimSize(0) || !index ||
        index.getOwner() != &whileOp.getAfter().front()) {
      return nullptr;
    }
    auto initial = whileOp.getInits()[index.getArgNumber()].getDefiningOp<felt::FeltConstantOp>();
    auto next =
        whileOp.getYieldOp().getOperand(index.getArgNumber()).getDefiningOp<felt::AddFeltOp>();
    if (!initial || initial.getValue().getReducedValueOrRaw() != 0 || !next) {
      return nullptr;
    }
    Value stepValue = next.getLhs() == index   ? next.getRhs()
                      : next.getRhs() == index ? next.getLhs()
                                               : Value();
    auto step =
        stepValue ? stepValue.getDefiningOp<felt::FeltConstantOp>() : felt::FeltConstantOp();
    auto source = write.getRvalue().getDefiningOp<ReadArrayOp>();
    if (!step || step.getValue().getReducedValueOrRaw() != 1 || !source ||
        source.getArrRef() == carried || !llvm::equal(source.getIndices(), write.getIndices())) {
      return nullptr;
    }
    coveringWrite = write;
  }
  bool hasOtherWrites = false;
  whileOp.getAfter().walk([&](WriteArrayOp write) { hasOtherWrites |= write != coveringWrite; });
  return hasOtherWrites ? WriteArrayOp() : coveringWrite;
}

/// Recognize the field-independent literals zero and one, including fieldless constants.
static bool isBooleanLiteral(const SourceRef &ref, int64_t literal) {
  if (auto value = ref.getConstant(); succeeded(value)) {
    if (auto constant = value->getDefiningOp<felt::FeltConstantOp>()) {
      return constant.getValue().getReducedValueOrRaw() == literal;
    }
  }
  auto index = ref.getConstantIndexValue();
  return succeeded(index) && *index == literal;
}

/// Recognize a literal zero written to this array before the loop.
static bool isZeroInitializerWrite(const SourceRef &ref, Value initializer) {
  auto value = ref.getConstant();
  if (!isBooleanLiteral(ref, 0) || failed(value)) {
    return false;
  }
  return llvm::any_of(value->getUsers(), [initializer](Operation *user) {
    auto write = llvm::dyn_cast<WriteArrayOp>(user);
    return write && write.getArrRef() == initializer;
  });
}

/// Collect POD record positions used to distinguish indexed aggregate alternatives.
static llvm::SmallVector<std::pair<uint64_t, uint64_t>>
getAggregateAlternativeOrdinals(const SourceRef &ref) {
  llvm::SmallVector<std::pair<uint64_t, uint64_t>> result;
  auto root = ref.getRoot();
  if (failed(root)) {
    return result;
  }
  Type currentType = root->getType();
  for (const SourceRefIndex &index : ref.getPath()) {
    if (index.isMember()) {
      currentType = index.getMember().getType();
      continue;
    }
    if (index.isPodRecord()) {
      auto podType = llvm::dyn_cast<PodType>(currentType);
      if (!podType) {
        continue;
      }
      auto records = podType.getRecords();
      for (auto [ordinal, record] : llvm::enumerate(records)) {
        if (record.getName() == index.getPodRecordNameAttr()) {
          result.emplace_back(ordinal, records.size());
          currentType = record.getType();
          break;
        }
      }
      continue;
    }
    if (index.isIndex() || index.isIndexRange()) {
      if (auto arrayType = llvm::dyn_cast<ArrayType>(currentType)) {
        currentType = arrayType.getElementType();
      }
    }
  }
  return result;
}

/// Check whether a POD alternative matches the selected aggregate element.
static bool isCompatibleIndexedAlternative(const SourceRef &candidate, const SourceRef &selection) {
  llvm::SmallVector<std::pair<uint64_t, uint64_t>> selectedIndices;
  auto root = selection.getRoot();
  if (failed(root)) {
    return true;
  }
  Type currentType = root->getType();
  for (const SourceRefIndex &index : selection.getPath()) {
    if (index.isMember()) {
      currentType = index.getMember().getType();
      continue;
    }
    if (index.isPodRecord()) {
      if (auto podType = llvm::dyn_cast<PodType>(currentType)) {
        for (auto record : podType.getRecords()) {
          if (record.getName() == index.getPodRecordNameAttr()) {
            currentType = record.getType();
            break;
          }
        }
      }
      continue;
    }
    if (auto arrayType = llvm::dyn_cast<ArrayType>(currentType)) {
      if (index.isIndex()) {
        selectedIndices.emplace_back(
            static_cast<uint64_t>(static_cast<int64_t>(index.getIndex())), arrayType.getDimSize(0)
        );
      }
      currentType = arrayType.getElementType();
    }
  }

  for (auto [ordinal, alternativeCount] : getAggregateAlternativeOrdinals(candidate)) {
    for (auto [selected, dimensionSize] : selectedIndices) {
      if (alternativeCount == dimensionSize && ordinal != selected) {
        return false;
      }
    }
  }
  return true;
}

/// Recognize a zero initializer used only as the identity of a loop-carried sum.
static std::optional<SourceRef> getNeutralWhileInitializer(Value value) {
  auto result = llvm::dyn_cast<OpResult>(value);
  auto whileOp =
      result ? llvm::dyn_cast_if_present<scf::WhileOp>(result.getDefiningOp()) : scf::WhileOp();
  if (!whileOp) {
    return std::nullopt;
  }

  unsigned resultNumber = result.getResultNumber();
  if (resultNumber >= whileOp.getInits().size() ||
      resultNumber >= whileOp.getYieldOp().getNumOperands() ||
      whileOp.getConditionOp().getArgs()[resultNumber] !=
          whileOp.getBeforeArguments()[resultNumber]) {
    return std::nullopt;
  }
  Value init = whileOp.getInits()[resultNumber];
  auto zero = init.getDefiningOp<felt::FeltConstantOp>();
  if (!zero || zero.getValue().getReducedValueOrRaw() != 0) {
    return std::nullopt;
  }

  Value carried = whileOp.getAfter().front().getArgument(resultNumber);
  Value yielded = whileOp.getYieldOp().getOperand(resultNumber);
  auto add = yielded.getDefiningOp<felt::AddFeltOp>();
  if (!add || !llvm::is_contained(add->getOperands(), carried)) {
    return std::nullopt;
  }
  return SourceRef(zero);
}

} // namespace

/* SourceRefAnalysis */

/// Solver-owned state for aggregate storage dependencies discovered by `SourceRefAnalysis`.
///
/// The ordinary `SourceRefLattice` records the storage addresses represented by each SSA value.
/// This state separately records copied values written to those addresses and aliases for
/// identity-bearing component handles. It is anchored before the top-level operation so
/// static dependency queries can retrieve it from the solver without retaining a pointer to the
/// analysis instance. The state is created lazily when the analysis first records a storage write.
class SourceRefAnalysis::StorageState : public AnalysisState {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StorageState)

  using AnalysisState::AnalysisState;

  /// Notification state for a storage query or a function's caller-visible effects.
  /// Its revision changes only when a recorded write or alias actually changes.
  class ScopeState : public AnalysisState {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScopeState)
    using AnalysisState::AnalysisState;

    void advance() { ++revision; }

    /// A query needs only one pending visit; it will observe the latest storage revision.
    void consume() {
      coalesce = true;
      pending = false;
    }

    void print(raw_ostream &os) const override { os << "storage revision " << revision; }

  protected:
    void onUpdate(DataFlowSolver *dataflowSolver) const override {
      if (coalesce && pending) {
        return;
      }
      pending = coalesce;
      AnalysisState::onUpdate(dataflowSolver);
    }

  private:
    size_t revision = 0;
    bool coalesce = false;
    mutable bool pending = false;
  };

  /// Associate this state with the top-level operation whose writes it models.
  void setTop(Operation *op, DataFlowSolver &dataflowSolver) {
    solver = &dataflowSolver;
    ensure(top == nullptr || top == op, "storage state cannot span top-level operations");
    if (top == nullptr) {
      op->walk([this](Operation *nested) {
        operationOrder.try_emplace(nested, operations.size());
        operations.push_back(nested);
        auto scope = nested->getParentOfType<FuncDefOp>();
        scopeBegins.try_emplace(scope.getOperation(), operationOrder.lookup(nested));
      });
    }
    top = op;
  }

  /// Track operations whose snapshots must be refreshed when earlier effects change.
  void registerConsumer(Operation *op) { consumers[getStorageScope(op)].insert(op); }

  /// Resolve known storage writes transitively, preserving unwritten and cyclic addresses.
  SourceRefLatticeValue
  resolveDependencies(const SourceRefLatticeValue &addresses, Operation *before) const;

  /// Follow component-handle assignments without substituting the component's member values.
  SourceRefLatticeValue
  resolveHandles(const SourceRefLatticeValue &handles, Operation *before) const;

  /// Translate callee dependencies using the caller's contents at entry to the call.
  SourceRefLatticeValue translateCallDependencies(
      const SourceRefLatticeValue &value, const TranslationMap &translation, Operation *call
  ) const;

  /// Record a storage write for later dependency queries.
  void recordStorageWrite(
      Operation *op, size_t writeIndex, const SourceRefLatticeValue &addresses,
      const SourceRefLatticeValue &value, Type valueType, bool mayBeSkipped = false,
      bool seedUnwrittenAlternative = false
  );

  /// Rebase an identity-bearing component handle to the storage receiving it.
  void recordAggregateAlias(
      Operation *op, size_t aliasIndex, const SourceRefLatticeValue &source,
      const SourceRefLatticeValue &target, bool mayBeSkipped = false
  );

  /// Record writes through shared component handles; array/POD arguments are independent copies.
  void recordCalleeStorageWrites(
      CallOpInterface call, FuncDefOp callee, const TranslationMap &translation
  );

  /// Give returned arrays/PODs independent storage, including returned arguments and slices.
  /// Component handles inside the returned value retain their identity.
  SourceRefLatticeValue recordReturnedAggregate(
      OpResult result, ArrayRef<std::pair<Operation *, SourceRefLatticeValue>> returnValues,
      const TranslationMap &translation
  );

  void print(raw_ostream &os) const override { os << "SourceRefAnalysis::StorageState"; }

private:
  /// A call translation applied after resolving a copied leaf in its original scope.
  struct CopyTranslation {
    TranslationMap arguments;
    Operation *call;
    std::optional<std::pair<SourceRef, SourceRef>> rebase;

    bool operator==(const CopyTranslation &rhs) const {
      return call == rhs.call && rebase == rhs.rebase && arguments.size() == rhs.arguments.size() &&
             llvm::all_of(arguments, [&](const auto &entry) {
        return llvm::is_contained(rhs.arguments, entry);
      });
    }
  };

  /// An elementwise copy whose last `depth` path components correspond to the destination's
  /// last `depth` components. Prefix selections (including a selected matrix row) stay fixed.
  /// `before` names the source observation point, never the state of a later destination read.
  struct DeferredCopy {
    SourceRef source;
    Operation *before;
    size_t depth;
    SmallVector<CopyTranslation, 0> translations;

    bool operator==(const DeferredCopy &) const = default;
  };

  /// Resolved dependencies and delayed elementwise alternatives at a storage location.
  struct StorageValue {
    SourceRefLatticeValue dependencies;
    SmallVector<DeferredCopy, 0> copies;

    StorageValue() = default;
    StorageValue(SourceRefLatticeValue value) : dependencies(std::move(value)) {}

    /// Join alternatives without resolving or duplicating a deferred copy.
    void update(const StorageValue &rhs) {
      (void)dependencies.update(rhs.dependencies);
      for (const DeferredCopy &copy : rhs.copies) {
        if (!llvm::is_contained(copies, copy)) {
          copies.push_back(copy);
        }
      }
    }

    bool operator==(const StorageValue &) const = default;
  };

  // libstdc++'s variant move path calls std::get after emplacing the same alternative.
  // The generated move assignment cannot reach the reported bad_variant_access.
  // NOLINTNEXTLINE(bugprone-exception-escape)
  struct StorageWrite {
    SourceRefLatticeValue addresses;
    StorageValue value;
    bool mayBeSkipped;
    bool seedUnwrittenAlternative;
    /// All ranged dimensions belong to the copied payload, not an unknown destination index.
    bool coversRange = false;

    bool operator==(const StorageWrite &) const = default;
  };

  // As above, the variant move path's std::get follows a matching emplace.
  // NOLINTNEXTLINE(bugprone-exception-escape)
  struct AggregateAlias {
    SourceRefLatticeValue source;
    SourceRefLatticeValue target;
    bool mayBeSkipped;

    bool operator==(const AggregateAlias &) const = default;
  };

  // NOLINTNEXTLINE(bugprone-exception-escape)
  struct MaterializedStorage {
    DenseMap<SourceRef, StorageValue> values;
    DenseSet<SourceRef> skippedFirstWrites;
    /// Ranges whose every element has contents, including conditional alternatives.
    DenseSet<SourceRef> coveredRanges;

    /// Addresses in `values`, grouped by their storage root for dependency lookups.
    DenseMap<Value, SmallVector<SourceRef>> valuesByRoot;

    void materializeOperation(
        const StorageState &state, Operation *op, TranslationMap &aliases, bool forceMayBeSkipped,
        Operation *replayedLoop = nullptr
    );

  private:
    /// Add or remove an address from the root index when `values` changes.
    void indexAddress(const SourceRef &address);
    void unindexAddress(const SourceRef &address);
    void eraseValue(const SourceRef &address);

    /// Remove the portion of earlier writes superseded by a definite point or range write.
    void invalidateOverlappingWrites(const SourceRef &address);

    void applyWrite(
        const SourceRef &address, const StorageValue &value, bool maySkip,
        bool seedUnwrittenAlternative, bool coversRange
    );

    void materializeWrite(
        const SourceRef &address, const StorageValue &value, bool maySkip,
        bool seedUnwrittenAlternative, bool resolveSelfReference, const TranslationMap &aliases,
        bool coversRange = false
    );

    void materializeAddressedWrite(
        const SourceRefLatticeValue &addresses, const StorageValue &value, bool maySkip,
        bool seedUnwrittenAlternative, const TranslationMap &aliases, Operation *replayedLoop,
        bool coversRange
    );
  };

  /// Storage and alias state immediately before an operation in preorder.
  ///
  /// Snapshots contain only the ordinary program-order replay. Loop backedge
  /// effects are query-specific and are applied after retrieving a snapshot.
  struct MaterializedSnapshot {
    TranslationMap aliases;
    MaterializedStorage storage;
  };

  /// Per-query snapshots and recursion guards, qualified by source observation point.
  struct ResolutionState {
    DenseMap<Operation *, std::unique_ptr<MaterializedSnapshot>> snapshots;
    DenseSet<std::pair<Operation *, SourceRef>> active;
  };

  /// Bound replay work while avoiding a full state copy at every query point.
  static constexpr size_t checkpointStride = 64;

  /// Apply all known aggregate-storage aliases to a lattice value.
  static SourceRefLatticeValue
  canonicalize(const SourceRefLatticeValue &value, const TranslationMap &aliases);

  /// Apply aggregate aliases recorded for `op` to `aliases`.
  void applyAggregateAliases(
      Operation *op, TranslationMap &aliases, bool forceMayBeSkipped = false
  ) const;

  /// Materialize storage and aliases at `before`, replaying from a sparse checkpoint.
  MaterializedSnapshot materializeSnapshot(Operation *before) const;

  /// Discard checkpoints whose program-order replay includes `op`.
  void invalidateCheckpointsFrom(Operation *op);

  static SourceRefLatticeValue projectChild(
      const SourceRefLatticeValue &value, const SourceRef &storedAddress,
      const SourceRef &readAddress
  );

  SourceRefLatticeValue resolve(
      const SourceRefLatticeValue &input, const TranslationMap &aliases,
      const MaterializedStorage &storage, Operation *before, ResolutionState &state,
      bool handlesOnly = false
  ) const;

  /// Resolve a selection in a cached snapshot, sharing recursion guards across copies and calls.
  SourceRefLatticeValue resolveAt(
      const SourceRefLatticeValue &input, Operation *before, ResolutionState &state,
      bool handlesOnly = false
  ) const;

  /// Project a destination selection into a deferred copy before observing its source.
  SourceRefLatticeValue resolveCopy(
      const DeferredCopy &copy, const SourceRef &storedAddress, const SourceRef &readAddress,
      ResolutionState &state
  ) const;

  SourceRefLatticeValue translateCallDependencies(
      const SourceRefLatticeValue &value, const TranslationMap &translation, Operation *call,
      ResolutionState &state
  ) const;

  /// Translate resolved values now and attach call translations to delayed alternatives.
  StorageValue translateCopyValue(
      const StorageValue &value, const TranslationMap &translation, Operation *call,
      std::optional<std::pair<SourceRef, SourceRef>> rebase = std::nullopt
  );

  /// Select writes to component members, resolving handles stored in copied arrays/PODs.
  /// Replacing a handle slot itself is a local value write, not a write to its component.
  SourceRefLatticeValue
  getSharedWriteTargets(const SourceRefLatticeValue &addresses, Operation *before) const;

  void recordCalleeStorageWritesImpl(
      Operation *op, const TranslationMap &translation, const TranslationMap &calleeAliases,
      SmallVectorImpl<StorageWrite> &translatedWrites, bool callMayBeSkipped, Operation *callSite
  );

  /// Translate caller-visible aggregate aliases from a callee operation.
  void recordCalleeAggregateAliasesImpl(
      Operation *op, const TranslationMap &translation,
      SmallVectorImpl<AggregateAlias> &translatedAliases, bool callMayBeSkipped
  ) const;

  /// Expand a mutable copy into independent leaf snapshots, preserving component handles.
  void appendStorageCopy(
      Operation *op, const SourceRefLatticeValue &addresses, const SourceRefLatticeValue &value,
      Type type, bool mayBeSkipped, bool seedUnwrittenAlternative,
      const MaterializedSnapshot &snapshot, SmallVectorImpl<StorageWrite> &writes, size_t depth = 0,
      bool hasCopiedRange = false
  ) const;

  /// Revisit storage consumers and callers when this operation's effects change.
  void notifyChanged(Operation *op) {
    DenseSet<Operation *> notified;
    notifyChanged(op, notified);
  }

  /// Propagate capture invalidation through callers even when their copy descriptors agree.
  void notifyChanged(Operation *op, DenseSet<Operation *> &notified) {
    if (!notified.insert(op).second) {
      return;
    }
    invalidateCheckpointsFrom(op);
    auto notify = [&](Operation *anchorOp) {
      auto *state = solver->getOrCreateState<ScopeState>(solver->getProgramPointBefore(anchorOp));
      state->advance();
      solver->propagateIfChanged(state, ChangeResult::Change);
    };
    Operation *scope = getStorageScope(op);
    notify(scope);
    for (Operation *consumer : consumers[scope]) {
      bool affected = operationOrder.lookup(op) < operationOrder.lookup(consumer);
      // Earlier observations can still see this write on a subsequent loop iteration.
      for (Operation *parent = op->getParentOp(); !affected && parent && parent != scope;
           parent = parent->getParentOp()) {
        if (!llvm::isa<LoopLikeOpInterface>(parent) || !parent->isAncestor(consumer)) {
          continue;
        }
        // A fresh allocation or copied read cannot carry its own contents into the
        // previous iteration. Only writes to persistent storage need a backward revisit.
        for (const auto &[index, writes] : storageWrites[op]) {
          affected |= llvm::any_of(writes, [&](const StorageWrite &write) {
            return llvm::any_of(write.addresses.foldToScalar(), [&](const SourceRef &address) {
              return !isAllocatedWithinLoop(address, parent);
            });
          });
        }
        affected |= !aggregateAliases[op].empty();
      }
      if (affected) {
        notify(consumer);
      }
    }
    if (auto callers = deferredCallers.find(scope); callers != deferredCallers.end()) {
      for (Operation *caller : callers->second) {
        notifyChanged(caller, notified);
      }
    }
  }

  /// Call sites whose delayed values observe storage in the keyed function scope.
  DenseMap<Operation *, llvm::SetVector<Operation *>> deferredCallers;

  /// Operations to revisit when storage effects change, grouped by `getStorageScope`.
  DenseMap<Operation *, llvm::SetVector<Operation *>> consumers;

  /// Owning solver used to propagate storage changes through `ScopeState` notifications.
  DataFlowSolver *solver = nullptr;

  /// Top-level operation whose storage effects and operation traversal this state models.
  Operation *top = nullptr;

  /// Storage writes grouped by source operation and stable initializer index.
  /// Grouping preserves initializer indices when a mutable copy expands into leaf writes.
  DenseMap<Operation *, std::map<size_t, SmallVector<StorageWrite>>> storageWrites;

  /// Component-handle alias events in replay order for each source operation.
  DenseMap<Operation *, SmallVector<AggregateAlias>> aggregateAliases;

  /// Operation-to-index lookup for the traversal recorded in `operations` by `setTop`.
  /// Used to order storage effects, identify affected consumers, and locate checkpoints.
  DenseMap<Operation *, size_t> operationOrder;

  /// Operations in the traversal order recorded by `setTop`, used for storage replay.
  SmallVector<Operation *> operations;

  /// First traversal index for each enclosing function, or nullptr for operations outside one.
  /// Bounds the initial replay needed to materialize that scope's snapshots.
  DenseMap<Operation *, size_t> scopeBegins;

  /// Sparse snapshots keyed by enclosing function (nullptr outside functions), then traversal
  /// index. Each snapshot precedes its indexed operation and excludes query-specific loop
  /// backedge effects. Function-local snapshots include other functions' effects only through
  /// translated calls, avoiding copies of unrelated allocations for each query.
  mutable DenseMap<Operation *, std::map<size_t, MaterializedSnapshot>> checkpoints;
};

const SourceRefAnalysis::Lattice *SourceRefAnalysis::getLattice(DataFlowSolver &solver, Value val) {
  return solver.lookupState<Lattice>(val);
}

SourceRefLatticeValue SourceRefAnalysis::getValueState(DataFlowSolver &solver, Value val) {
  if (const auto *state = getLattice(solver, val)) {
    return state->getValue();
  }
  return SourceRefLattice::getDefaultValue(val);
}

/// Retain the initial component at an unresolved aggregate path as a value, not a live slot.
static SourceRefLatticeValue captureComponentHandles(const SourceRefLatticeValue &value) {
  if (value.isArray()) {
    SourceRefLatticeValue captured(value.getArrayShape());
    for (size_t i = 0; i < value.getArraySize(); ++i) {
      (void)captured.getElemFlatIdx(i).setValue(captureComponentHandles(value.getElemFlatIdx(i)));
    }
    return captured;
  }
  SourceRefLatticeValue captured;
  for (const SourceRef &ref : value.getScalarValue()) {
    (void)captured.insert(llvm::isa<StructType>(ref.getType()) ? ref.captureComponent() : ref);
  }
  return captured;
}

/// Resolve one SSA dependency expression, sharing repeated subexpressions within the query.
static SourceRefLatticeValue getObservedDependencies(
    DataFlowSolver &solver, Value val, DenseMap<Value, SourceRefLatticeValue> &cache
) {
  if (auto found = cache.find(val); found != cache.end()) {
    return found->second;
  }
  auto compute = [&]() -> SourceRefLatticeValue {
    if (Operation *def = val.getDefiningOp(); def && !isAggregateType(val.getType())) {
      // Calls already translate scalar dependencies from their return sites. Resolving that
      // snapshot again could replace an unwritten source with a later write in the caller.
      if (llvm::isa<CallOpInterface>(def)) {
        return SourceRefAnalysis::getValueState(solver, val);
      }
      // The raw lattice deliberately erases intermediate arithmetic operations. Follow their
      // SSA operands here so a read used after an overwrite keeps its original observation point.
      if (def->getNumRegions() == 0 &&
          !llvm::isa<ArrayAccessOpInterface, PodAccessOpInterface, MemberRefOpInterface>(def) &&
          failed(SourceRefLattice::getSourceRef(val))) {
        SourceRefLatticeValue dependencies;
        for (Value operand : def->getOperands()) {
          (void)dependencies.update(getObservedDependencies(solver, operand, cache));
        }
        return dependencies;
      }
    }
    // Region-branch result lattices merge the references carried by their exiting regions, but a
    // storage-backed yielded value must be resolved at the region terminator rather than before
    // the enclosing operation. Resolving it before the region branch would omit writes performed
    // earlier in the region. Repetitive region branches retain the existing loop-carried model.
    if (auto result = llvm::dyn_cast<OpResult>(val)) {
      auto regionBranch = llvm::dyn_cast<RegionBranchOpInterface>(result.getOwner());
      if (regionBranch && !regionBranch.hasLoop()) {
        SourceRefLatticeValue yieldedDependencies;
        bool foundYield = false;
        const unsigned resultNumber = result.getResultNumber();
        for (Region &region : regionBranch->getRegions()) {
          SmallVector<RegionSuccessor> successors;
          regionBranch.getSuccessorRegions(region, successors);
          const bool exitsToParent = llvm::any_of(successors, [](const RegionSuccessor &successor) {
            return successor.isOperation();
          });
          if (!exitsToParent) {
            continue;
          }
          for (Block &block : region) {
            Operation *terminator = block.getTerminator();
            // Follow the executable state used by sparse propagation so a dead region's
            // yield cannot reintroduce dependencies excluded from the result lattice.
            if (terminator == nullptr || !dataflow::isOperationLive(solver, terminator)) {
              continue;
            }
            auto resolveYieldedOperand = [&](ValueRange yieldedOperands) {
              if (resultNumber >= yieldedOperands.size()) {
                return;
              }
              (void)yieldedDependencies.update(
                  isAggregateType(yieldedOperands[resultNumber].getType())
                      ? SourceRefAnalysis::getDependencyState(
                            solver, yieldedOperands[resultNumber], terminator
                        )
                      : getObservedDependencies(solver, yieldedOperands[resultNumber], cache)
              );
              foundYield = true;
            };
            if (auto branch = llvm::dyn_cast<RegionBranchTerminatorOpInterface>(terminator)) {
              resolveYieldedOperand(
                  branch.getSuccessorOperands(RegionSuccessor(regionBranch.getOperation()))
              );
            } else if (terminator->hasTrait<OpTrait::ReturnLike>()) {
              resolveYieldedOperand(terminator->getOperands());
            }
          }
        }
        if (foundYield) {
          return yieldedDependencies;
        }
      }
    }

    Operation *before = val.getDefiningOp();
    if (before == nullptr) {
      before = val.getParentBlock()->getParentOp();
    } else if (isMutableAggregateType(val.getType()) && before->getNextNode()) {
      // Allocation and call-result contents become available after their defining operation.
      before = before->getNextNode();
    }
    return SourceRefAnalysis::getDependencyState(
        solver, SourceRefAnalysis::getValueState(solver, val), before
    );
  };
  SourceRefLatticeValue result = compute();
  if (std::optional<SourceRef> neutralInitializer = getNeutralWhileInitializer(val)) {
    (void)result.remove(*neutralInitializer);
    auto whileResult = llvm::cast<OpResult>(val);
    auto whileOp = llvm::cast<scf::WhileOp>(whileResult.getDefiningOp());
    Value carried = whileOp.getAfter().front().getArgument(whileResult.getResultNumber());
    auto add = whileOp.getYieldOp()
                   .getOperand(whileResult.getResultNumber())
                   .getDefiningOp<felt::AddFeltOp>();
    for (Value operand : add->getOperands()) {
      if (operand != carried) {
        (void)result.update(getObservedDependencies(solver, operand, cache));
      }
    }
  }
  if (WriteArrayOp write = getFullyOverwrittenLoopArrayWrite(val)) {
    auto read = val.getDefiningOp<ReadArrayOp>();
    auto loopResult = llvm::cast<OpResult>(read.getArrRef());
    auto loop = loopResult.getDefiningOp<scf::WhileOp>();
    Value initializer = loop.getInits()[loopResult.getResultNumber()];
    SourceRefSet writtenDependencies =
        getObservedDependencies(solver, write.getRvalue(), cache).foldToScalar();
    SourceRefLatticeValue normalized;
    auto selected = SourceRefAnalysis::getValueState(solver, val);
    for (const SourceRef &ref : result.foldToScalar()) {
      if (isZeroInitializerWrite(ref, initializer) && !writtenDependencies.contains(ref)) {
        continue;
      }
      (void)normalized.insert(
          selected.isSingleValue() && writtenDependencies.contains(ref)
              ? ref.narrowRanges(selected.getSingleValue())
              : ref
      );
    }
    result = std::move(normalized);
  }
  if (llvm::isa<StructType>(val.getType())) {
    result = captureComponentHandles(result);
  }
  cache.try_emplace(val, result);
  return result;
}

SourceRefLatticeValue SourceRefAnalysis::getDependencyState(DataFlowSolver &solver, Value val) {
  DenseMap<Value, SourceRefLatticeValue> cache;
  return getObservedDependencies(solver, val, cache);
}

SourceRefLatticeValue
SourceRefAnalysis::getDependencyState(DataFlowSolver &solver, Value val, Operation *before) {
  if (!isMutableAggregateType(val.getType())) {
    return getDependencyState(solver, val);
  }
  return getDependencyState(solver, getValueState(solver, val), before);
}

SourceRefLatticeValue SourceRefAnalysis::getDependencyState(
    DataFlowSolver &solver, const SourceRefLatticeValue &refs, Operation *before
) {
  Operation *top = before;
  while (top->getParentOp() != nullptr) {
    top = top->getParentOp();
  }
  if (const auto *state = solver.lookupState<StorageState>(solver.getProgramPointBefore(top))) {
    return state->resolveDependencies(refs, before);
  }
  return refs;
}

SourceRefLatticeValue SourceRefAnalysis::getStoredValue(Value value) {
  // Component handles, like scalars, retain the value observed at their SSA definition.
  return isMutableAggregateType(value.getType()) ? getValueState(dataflowSolver_, value)
                                                 : getDependencyState(dataflowSolver_, value);
}

SourceRefLatticeValue
SourceRefAnalysis::recordReadCopy(Operation *op, const SourceRefLatticeValue &source) {
  SourceRefLatticeValue target(SourceRef(op->getResult(0)));
  getStorageState(op)->recordStorageWrite(
      op, 0, target, source, op->getResult(0).getType(), isInMaybeSkippedScfRegion(op)
  );
  return target;
}

SourceRefAnalysis::StorageState *SourceRefAnalysis::getStorageState(Operation *op) {
  // Snapshots consumed by writes and copies must be recomputed when an earlier call's
  // effects become available, even when their raw SSA address lattices do not change.
  auto *query = getOrCreate<StorageState::ScopeState>(getProgramPointBefore(op));
  query->consume();
  addDependency(query, getProgramPointAfter(op));
  Operation *top = op;
  while (top->getParentOp() != nullptr) {
    top = top->getParentOp();
  }
  auto *state = getOrCreate<StorageState>(getProgramPointBefore(top));
  state->setTop(top, dataflowSolver_);
  state->registerConsumer(op);
  return state;
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::canonicalize(
    const SourceRefLatticeValue &value, const TranslationMap &aliases
) {
  SourceRefLatticeValue canonical = value;
  for (size_t i = 0; i < aliases.size(); ++i) {
    auto [next, changed] = canonical.replacePrefixes(aliases, /*preserveCaptures=*/true);
    canonical = std::move(next);
    if (changed == ChangeResult::NoChange) {
      break;
    }
  }
  return canonical;
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::projectChild(
    const SourceRefLatticeValue &value, const SourceRef &storedAddress, const SourceRef &readAddress
) {
  if (!readAddress.isValidPrefix(storedAddress) && !readAddress.overlaps(storedAddress)) {
    return value;
  }
  if (value.isArray()) {
    SourceRefLatticeValue result(value.getArrayShape());
    for (size_t i = 0; i < value.getArraySize(); ++i) {
      (void)result.getElemFlatIdx(i).setValue(
          projectChild(value.getElemFlatIdx(i), storedAddress, readAddress)
      );
    }
    return result;
  }

  SourceRefLatticeValue result;
  for (const SourceRef &ref : value.getScalarValue()) {
    if (ref == storedAddress && hasRangeIndex(ref)) {
      (void)result.insert(ref.narrowRanges(readAddress));
    } else if (auto translated = readAddress.translate(storedAddress, ref); succeeded(translated)) {
      (void)result.insert(*translated);
    } else {
      (void)result.insert(ref);
    }
  }
  return result;
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::resolve(
    const SourceRefLatticeValue &input, const TranslationMap &aliases,
    const MaterializedStorage &storage, Operation *before, ResolutionState &state, bool handlesOnly
) const {
  SourceRefLatticeValue addressValue = canonicalize(input, aliases);
  if (addressValue.isArray()) {
    SourceRefLatticeValue result(addressValue.getArrayShape());
    for (size_t i = 0; i < addressValue.getArraySize(); ++i) {
      (void)result.getElemFlatIdx(i).setValue(
          resolve(addressValue.getElemFlatIdx(i), aliases, storage, before, state, handlesOnly)
      );
    }
    return result;
  }

  SourceRefLatticeValue result;
  for (const SourceRef &address : addressValue.getScalarValue()) {
    if (!state.active.insert({before, address}).second) {
      if (addressValue.isSingleValue() || storage.skippedFirstWrites.contains(address)) {
        // A skipped first write includes its destination as the unwritten alternative. Preserve
        // that alternative even when the write value contributes other dependencies.
        (void)result.insert(address);
      }
      continue;
    }

    bool foundWrite = false;
    bool preservesAddress = false;
    SourceRefLatticeValue writtenValues;
    // Point writes and elementwise copies establish coverage. An unknown-index scalar write
    // only selects one runtime element and cannot cover a whole range.
    SmallVector<SourceRef> uncoveredRanges;
    if (hasRangeIndex(address)) {
      uncoveredRanges.push_back(address);
    }
    auto addressRoot = address.getRoot();
    if (succeeded(addressRoot)) {
      auto writesForRoot = storage.valuesByRoot.find(*addressRoot);
      if (writesForRoot != storage.valuesByRoot.end()) {
        for (const SourceRef &storedAddress : writesForRoot->second) {
          if (address.protectsCapturedComponent(storedAddress)) {
            continue;
          }
          if (handlesOnly && !llvm::isa<StructType>(storedAddress.getType())) {
            continue;
          }
          auto storedValue = storage.values.find(storedAddress);
          ensure(storedValue != storage.values.end(), "storage root index is out of sync");
          if (!storedAddress.overlaps(address) && !address.isValidPrefix(storedAddress)) {
            continue;
          }
          foundWrite = true;
          SourceRefLatticeValue projectedValue =
              projectChild(storedValue->second.dependencies, storedAddress, address);
          for (const DeferredCopy &copy : storedValue->second.copies) {
            (void)projectedValue.update(resolveCopy(copy, storedAddress, address, state));
          }
          if (hasRangeIndex(storedAddress) && storedAddress.overlaps(address)) {
            preservesAddress |= projectedValue.remove(address) == ChangeResult::Change;
          }
          if (llvm::isa<StructType>(storedAddress.getType())) {
            (void)writtenValues.update(projectedValue);
          } else {
            // Copied scalar dependencies are snapshots. They must not be re-read after a
            // later mutation, even if an unwritten source is represented by its address.
            (void)result.update(projectedValue);
          }

          if (!hasRangeIndex(storedAddress) || storage.coveredRanges.contains(storedAddress)) {
            SmallVector<SourceRef> nextUncoveredRanges;
            for (const SourceRef &uncovered : uncoveredRanges) {
              if (!storedAddress.overlaps(uncovered)) {
                nextUncoveredRanges.push_back(uncovered);
                continue;
              }
              auto residuals = subtractAddress(uncovered, storedAddress);
              nextUncoveredRanges.append(residuals);
            }
            uncoveredRanges = std::move(nextUncoveredRanges);
          }
        }
      }
    }
    preservesAddress |= !uncoveredRanges.empty();
    if (foundWrite) {
      SourceRefLatticeValue resolvedValues =
          resolve(writtenValues, aliases, storage, before, state, handlesOnly);
      (void)result.update(resolvedValues);
      if (preservesAddress) {
        (void)result.insert(address);
      }
    }
    if (!foundWrite) {
      (void)result.insert(address);
    }
    state.active.erase({before, address});
  }
  return handlesOnly ? captureComponentHandles(result) : result;
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::resolveDependencies(
    const SourceRefLatticeValue &addresses, Operation *before
) const {
  ResolutionState state;
  SymbolTableCollection queryTables;
  auto module = before->getParentOfType<ModuleOp>();
  // Value aggregates expose their scalar contents at this observation point. Component handles
  // remain identities: expanding them would lose the captured instance used by later member reads.
  std::function<SourceRefLatticeValue(const SourceRefLatticeValue &)> expand =
      [&](const SourceRefLatticeValue &input) {
    if (input.isArray()) {
      SourceRefLatticeValue result(input.getArrayShape());
      for (size_t i = 0; i < input.getArraySize(); ++i) {
        (void)result.getElemFlatIdx(i).setValue(expand(input.getElemFlatIdx(i)));
      }
      return result;
    }
    SourceRefLatticeValue result;
    for (const SourceRef &address : input.getScalarValue()) {
      if (!isMutableAggregateType(address.getType())) {
        (void)result.insert(address);
        continue;
      }
      SourceRefLatticeValue children;
      if (auto arrayType = llvm::dyn_cast<ArrayType>(address.getType());
          arrayType && (!arrayType.hasStaticShape() || arrayType.getNumElements() > 64)) {
        // Keep large and symbolic arrays compact while still selecting their scalar leaves.
        auto child =
            address.createChild(SourceRefIndex::forArrayDimension(arrayType.getDimSize(0)));
        ensure(succeeded(child), "could not select aggregate dependency range");
        (void)children.insert(*child);
        (void)result.update(expand(children));
        continue;
      }
      for (const SourceRef &child : address.getAllChildren(queryTables, module)) {
        (void)children.insert(child);
      }
      (void)result.update(expand(children));
    }
    return result;
  };
  return resolveAt(expand(addresses), before, state);
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::resolveHandles(
    const SourceRefLatticeValue &handles, Operation *before
) const {
  ResolutionState state;
  return resolveAt(handles, before, state, /*handlesOnly=*/true);
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::resolveAt(
    const SourceRefLatticeValue &input, Operation *before, ResolutionState &state, bool handlesOnly
) const {
  auto &entry = state.snapshots[before];
  if (!entry) {
    entry = std::make_unique<MaterializedSnapshot>(materializeSnapshot(before));
  }
  const MaterializedSnapshot &snapshot = *entry;
  return resolve(input, snapshot.aliases, snapshot.storage, before, state, handlesOnly);
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::resolveCopy(
    const DeferredCopy &copy, const SourceRef &storedAddress, const SourceRef &readAddress,
    ResolutionState &state
) const {
  SourceRef source = copy.source;
  if (source.isRooted()) {
    ensure(
        copy.depth <= source.getPath().size() && copy.depth <= storedAddress.getPath().size(),
        "copied suffix must fit both storage paths"
    );
    size_t sourceBegin = source.getPath().size() - copy.depth;
    size_t targetBegin = storedAddress.getPath().size() - copy.depth;
    for (size_t i = 0; i < copy.depth; ++i) {
      const SourceRefIndex &sourceIndex = source.getPath()[sourceBegin + i];
      if (!sourceIndex.isIndexRange()) {
        continue;
      }
      SourceRefIndex selected = intersectIndex(
          storedAddress.getPath()[targetBegin + i], readAddress.getPath()[targetBegin + i]
      );
      source = *replacePathIndex(source, sourceBegin + i, intersectIndex(sourceIndex, selected));
    }
  }
  auto value = resolveAt(
      SourceRefLatticeValue(source), copy.before, state,
      /*handlesOnly=*/llvm::isa<StructType>(source.getType())
  );
  for (const CopyTranslation &translation : copy.translations) {
    value = translateCallDependencies(value, translation.arguments, translation.call, state);
    if (translation.rebase) {
      const auto &[from, to] = *translation.rebase;
      value = value.replacePrefixes(TranslationMap {{from, SourceRefLatticeValue(to)}}).first;
    }
  }
  return value;
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::getSharedWriteTargets(
    const SourceRefLatticeValue &addresses, Operation *before
) const {
  SourceRefLatticeValue targets;
  for (const SourceRef &address : addresses.foldToScalar()) {
    for (auto parent = address.getParentPrefix(); succeeded(parent);
         parent = parent->getParentPrefix()) {
      if (!llvm::isa<StructType>(parent->getType())) {
        continue;
      }
      auto handles = resolveHandles(SourceRefLatticeValue(*parent), before);
      auto suffix = address.getSuffix(*parent);
      ensure(succeeded(suffix), "component must be a prefix of its member address");
      for (const SourceRef &handle : handles.foldToScalar()) {
        (void)targets.insert(SourceRef::appendSuffix(handle, *suffix));
      }
      break;
    }
  }
  return targets;
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::translateCallDependencies(
    const SourceRefLatticeValue &value, const TranslationMap &translation, Operation *call
) const {
  ResolutionState state;
  return translateCallDependencies(value, translation, call, state);
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::translateCallDependencies(
    const SourceRefLatticeValue &value, const TranslationMap &translation, Operation *call,
    ResolutionState &state
) const {
  SourceRefLatticeValue result;
  Operation *callee = llvm::cast<CallOpInterface>(call).resolveCallable();
  for (const SourceRef &ref : value.foldToScalar()) {
    auto [translated, changed] = SourceRefLatticeValue(ref).replacePrefixes(translation);
    auto root = ref.getRoot();
    if (changed == ChangeResult::NoChange && succeeded(root) && root->getDefiningOp() &&
        !isMutableAggregateType(ref.getType()) && !ref.isImmutableGlobal()) {
      Operation *origin =
          ref.getCallContext().empty() ? root->getDefiningOp() : ref.getCallContext().back();
      if (getStorageScope(origin) == callee) {
        // Local nondets, component handles, and unwritten storage belong to this invocation.
        // Preserve sharing across fields/results while distinguishing separate calls,
        // including outer calls forwarding an already-scoped inner result.
        translated = SourceRefLatticeValue(ref.inCallContext(call));
      }
    }
    if (changed == ChangeResult::Change && succeeded(root) && isAggregateType(root->getType())) {
      if (isMutableAggregateType(root->getType())) {
        // A callee's initial aggregate payload is observed at entry to this call. Rebind its
        // capture against the caller snapshot; captures in actual struct arguments stay frozen.
        translated =
            SourceRefLatticeValue(ref.withoutComponentCapture()).replacePrefixes(translation).first;
      }
      translated = resolveAt(translated, call, state, llvm::isa<StructType>(ref.getType()));
    }
    (void)result.update(translated);
  }
  return result;
}

SourceRefAnalysis::StorageState::StorageValue SourceRefAnalysis::StorageState::translateCopyValue(
    const StorageValue &value, const TranslationMap &translation, Operation *call,
    std::optional<std::pair<SourceRef, SourceRef>> rebase
) {
  StorageValue result(translateCallDependencies(value.dependencies, translation, call));
  if (rebase) {
    const auto &[from, to] = *rebase;
    result.dependencies =
        result.dependencies.replacePrefixes(
                               TranslationMap {{from, SourceRefLatticeValue(to)}}
        ).first;
  }
  for (DeferredCopy copy : value.copies) {
    deferredCallers[getStorageScope(copy.before)].insert(call);
    for (const CopyTranslation &step : copy.translations) {
      deferredCallers[getStorageScope(step.call)].insert(call);
    }
    copy.translations.push_back({translation, call, rebase});
    result.copies.push_back(std::move(copy));
  }
  return result;
}

void SourceRefAnalysis::StorageState::appendStorageCopy(
    Operation *op, const SourceRefLatticeValue &addresses, const SourceRefLatticeValue &value,
    Type type, bool mayBeSkipped, bool seedUnwrittenAlternative,
    const MaterializedSnapshot &snapshot, SmallVectorImpl<StorageWrite> &writes, size_t depth,
    bool hasCopiedRange
) const {
  if (auto podType = llvm::dyn_cast<PodType>(type)) {
    for (auto record : podType.getRecords()) {
      auto targets = addresses.referencePodRecord(record.getName());
      auto sources = value.referencePodRecord(record.getName());
      ensure(succeeded(targets) && succeeded(sources), "could not select copied POD record");
      appendStorageCopy(
          op, targets->first, sources->first, record.getType(), mayBeSkipped,
          seedUnwrittenAlternative, snapshot, writes, depth + 1, hasCopiedRange
      );
    }
    return;
  }
  if (auto arrayType = llvm::dyn_cast<ArrayType>(type)) {
    SmallVector<SourceRefIndex> indices;
    std::function<void(unsigned)> copyDimension = [&](unsigned dim) {
      if (dim == arrayType.getRank()) {
        auto targets = addresses.extract(indices);
        auto sources = value.extract(indices);
        ensure(succeeded(targets) && succeeded(sources), "could not select copied array element");
        appendStorageCopy(
            op, targets->first, sources->first, arrayType.getElementType(), mayBeSkipped,
            seedUnwrittenAlternative, snapshot, writes, depth + arrayType.getRank(),
            hasCopiedRange || !arrayType.hasStaticShape()
        );
        return;
      }
      int64_t size = arrayType.getDimSize(dim);
      if (ShapedType::isDynamic(size)) {
        indices.push_back(SourceRefIndex::forArrayDimension(size));
        copyDimension(dim + 1);
        indices.pop_back();
      } else {
        for (int64_t i = 0; i < size; ++i) {
          indices.emplace_back(i);
          copyDimension(dim + 1);
          indices.pop_back();
        }
      }
    };
    copyDimension(0);
    return;
  }
  if (hasCopiedRange) {
    StorageValue copied;
    for (const SourceRef &source : value.foldToScalar()) {
      copied.copies.push_back({source, op, depth, {}});
    }
    for (const SourceRef &target : addresses.foldToScalar()) {
      bool coversRange = !llvm::any_of(target.getPath().drop_back(depth), [](const auto &index) {
        return index.isIndexRange();
      });
      bool ambiguousTarget = !addresses.isSingleValue();
      writes.push_back(
          {SourceRefLatticeValue(target), copied, mayBeSkipped || ambiguousTarget,
           seedUnwrittenAlternative || ambiguousTarget, coversRange}
      );
    }
    return;
  }
  // Scalar leaves capture the source contents at the copy operation. Struct leaves remain
  // handles, so following a member through the copied handle still reaches the same instance.
  ResolutionState state;
  writes.push_back(
      {addresses,
       resolve(value, snapshot.aliases, snapshot.storage, op, state, llvm::isa<StructType>(type)),
       mayBeSkipped, seedUnwrittenAlternative}
  );
}

void SourceRefAnalysis::StorageState::recordStorageWrite(
    Operation *op, size_t writeIndex, const SourceRefLatticeValue &addresses,
    const SourceRefLatticeValue &value, Type valueType, bool mayBeSkipped,
    bool seedUnwrittenAlternative
) {
  SmallVector<StorageWrite> writes;
  if (isMutableAggregateType(valueType)) {
    // All leaves observe the same pre-copy state; materialize it once for large arrays.
    MaterializedSnapshot snapshot = materializeSnapshot(op);
    appendStorageCopy(
        op, addresses, value, valueType, mayBeSkipped, seedUnwrittenAlternative, snapshot, writes
    );
  } else {
    // Scalar RHS values have already been observed at their SSA definition.
    writes.push_back({addresses, value, mayBeSkipped, seedUnwrittenAlternative});
  }
  auto &recorded = storageWrites[op][writeIndex];
  if (recorded != writes) {
    recorded = std::move(writes);
    notifyChanged(op);
  }
}

void SourceRefAnalysis::StorageState::recordCalleeStorageWritesImpl(
    Operation *op, const TranslationMap &translation, const TranslationMap &calleeAliases,
    SmallVectorImpl<StorageWrite> &translatedWrites, bool callMayBeSkipped, Operation *callSite
) {
  auto writes = storageWrites.find(op);
  if (writes == storageWrites.end()) {
    return;
  }
  for (const auto &[index, group] : writes->second) {
    for (const StorageWrite &write : group) {
      // A local component handle may have been assigned to a callee argument before this
      // write. Canonicalize through aliases already active at the write point so
      // the resulting caller-visible write is retained instead of being dropped
      // with the callee-local address.
      SourceRefLatticeValue canonicalAddresses =
          getSharedWriteTargets(canonicalize(write.addresses, calleeAliases), op);
      auto [addresses, addressChange] = canonicalAddresses.translate(translation);
      if (addressChange == ChangeResult::NoChange || addresses.foldToScalar().empty()) {
        continue;
      }
      addresses = getSharedWriteTargets(addresses, callSite);
      // Resolve storage-backed values at the callee write point before translating its arguments.
      // For example, a value read from a locally-created POD is initially rooted in that POD, but
      // resolves to the argument used to initialize the record. `replacePrefixes` alone cannot
      // translate that callee-local root.
      const bool aggregateWrite = llvm::any_of(write.addresses.foldToScalar(), [](const auto &ref) {
        return isAggregateType(ref.getType());
      });
      StorageValue resolvedValue = write.value;
      if (aggregateWrite) {
        resolvedValue.dependencies = resolveDependencies(resolvedValue.dependencies, op);
      }
      // Unlike addresses, values may legitimately include constants. Keep such
      // sources while replacing every reference rooted in a callee argument.
      StorageValue value = translateCopyValue(resolvedValue, translation, callSite);
      const bool translatedWriteMayBeSkipped =
          callMayBeSkipped || (write.coversRange ? !addresses.isSingleValue()
                                                 : isNonSingletonArrayWriteTarget(addresses));
      translatedWrites.push_back({
          std::move(addresses),
          std::move(value),
          write.mayBeSkipped || translatedWriteMayBeSkipped,
          write.seedUnwrittenAlternative || translatedWriteMayBeSkipped,
          write.coversRange,
      });
    }
  }
}

void SourceRefAnalysis::StorageState::recordCalleeAggregateAliasesImpl(
    Operation *op, const TranslationMap &translation,
    SmallVectorImpl<AggregateAlias> &translatedAliases, bool callMayBeSkipped
) const {
  auto aliases = aggregateAliases.find(op);
  if (aliases == aggregateAliases.end()) {
    return;
  }
  for (const AggregateAlias &alias : aliases->second) {
    // Both sides must be visible to the caller. An alias rooted in callee-local
    // storage only models the callee's temporary state and cannot observe a
    // later caller write through its source.
    auto [source, sourceChange] = alias.source.translate(translation);
    auto [target, targetChange] = getSharedWriteTargets(alias.target, op).translate(translation);
    if (sourceChange == ChangeResult::NoChange || targetChange == ChangeResult::NoChange ||
        source.foldToScalar().empty() || target.foldToScalar().empty()) {
      continue;
    }
    translatedAliases.push_back({
        std::move(source),
        std::move(target),
        alias.mayBeSkipped || callMayBeSkipped,
    });
  }
}

void SourceRefAnalysis::StorageState::recordCalleeStorageWrites(
    CallOpInterface call, FuncDefOp callee, const TranslationMap &translation
) {
  SmallVector<StorageWrite> translatedWrites;
  SmallVector<AggregateAlias> translatedAliases;
  if (Region *callableRegion = callee.getCallableRegion()) {
    const bool callMayBeSkipped = isInMaybeSkippedScfRegion(call.getOperation());
    TranslationMap calleeAliases;
    // Region walking is preorder, matching the storage state's program-order
    // replay. This includes writes nested in control-flow regions, whose
    // `mayBeSkipped` flags preserve their conditional semantics.
    callableRegion->walk([&](Operation *op) {
      recordCalleeStorageWritesImpl(
          op, translation, calleeAliases, translatedWrites, callMayBeSkipped, call.getOperation()
      );
      recordCalleeAggregateAliasesImpl(op, translation, translatedAliases, callMayBeSkipped);
      applyAggregateAliases(op, calleeAliases, callMayBeSkipped);
    });
  }

  // Recompute the complete summary on every solver revisit. This prevents
  // duplicate events and lets operand lattice refinements update the call-site
  // translation without changing its position in program order.
  auto &recordedWrites = storageWrites[call.getOperation()][0];
  auto &recordedAliases = aggregateAliases[call.getOperation()];
  if (recordedWrites != translatedWrites || recordedAliases != translatedAliases) {
    recordedWrites = std::move(translatedWrites);
    recordedAliases = std::move(translatedAliases);
    notifyChanged(call.getOperation());
  }
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::recordReturnedAggregate(
    OpResult result, ArrayRef<std::pair<Operation *, SourceRefLatticeValue>> returnValues,
    const TranslationMap &translation
) {
  Operation *call = result.getOwner();
  SmallVector<StorageWrite> initializers;
  SourceRef root(result);
  auto arrayType = llvm::dyn_cast<ArrayType>(result.getType());
  SourceRefLatticeValue references;
  if (arrayType && arrayType.hasStaticShape()) {
    references = SourceRefLatticeValue(arrayType.getShape());
  }

  for (const auto &[returnSite, returned] : returnValues) {
    MaterializedSnapshot snapshot = materializeSnapshot(returnSite);
    auto translateLeaf = [&](const SourceRefLatticeValue &sources, const SourceRef &target) {
      SourceRefLatticeValue leafReferences;
      for (const SourceRef &source : sources.foldToScalar()) {
        (void)leafReferences.insert(target);
        SmallVector<StorageWrite> copied;
        appendStorageCopy(
            returnSite, SourceRefLatticeValue(target), SourceRefLatticeValue(source),
            target.getType(), /*mayBeSkipped=*/true, /*seedUnwrittenAlternative=*/false, snapshot,
            copied
        );
        for (StorageWrite &write : copied) {
          // An uninitialized callee allocation becomes this call's allocation. Keep that
          // identity distinct across calls, even when their initializer dependencies agree.
          write.value = translateCopyValue(write.value, translation, call, {{source, target}});
          initializers.push_back(std::move(write));
        }
      }
      return leafReferences;
    };
    if (arrayType && arrayType.hasStaticShape()) {
      ArrayIndexGen indexGen = ArrayIndexGen::from(arrayType);
      for (size_t i = 0; i < references.getArraySize(); ++i) {
        auto indices = indexGen.delinearize(checkedCast<int64_t>(i), result.getContext());
        ensure(indices.has_value(), "could not delinearize returned array element");
        SmallVector<SourceRefIndex> path;
        for (Attribute attr : *indices) {
          path.emplace_back(DynamicAPInt(llvm::cast<IntegerAttr>(attr).getValue()));
        }
        auto selected = returned.extract(path);
        ensure(succeeded(selected), "could not select returned array storage");
        (void)references.getElemFlatIdx(i).update(
            translateLeaf(selected->first, getArrayElementAddress(root, i))
        );
      }
    } else {
      (void)references.update(translateLeaf(returned, root));
    }
  }
  // Group zero contains argument side effects. Each result owns a separate initializer group.
  auto &recorded = storageWrites[call][result.getResultNumber() + 1];
  if (recorded != initializers) {
    recorded = std::move(initializers);
    notifyChanged(call);
  }
  return references;
}

SourceRefAnalysis::StorageState::MaterializedSnapshot
SourceRefAnalysis::StorageState::materializeSnapshot(Operation *before) const {
  ensure(top != nullptr, "storage state must be associated with a top-level operation");

  auto target = operationOrder.find(before);
  ensure(target != operationOrder.end(), "snapshot operation must belong to storage state");

  // Materialize full checkpoints only at fixed program-order intervals. A
  // checkpoint represents state before its key operation, so replay starts at
  // that operation.
  MaterializedSnapshot snapshot;
  auto scope = before->getParentOfType<FuncDefOp>();
  Operation *scopeOp = scope.getOperation();
  auto &scopeCheckpoints = checkpoints[scopeOp];
  size_t replayBegin = scopeBegins.lookup(scopeOp) / checkpointStride * checkpointStride;
  const size_t checkpointTarget = target->second / checkpointStride * checkpointStride;
  auto nextCheckpoint = scopeCheckpoints.upper_bound(checkpointTarget);
  if (nextCheckpoint != scopeCheckpoints.begin()) {
    auto previousCheckpoint = std::prev(nextCheckpoint);
    snapshot = previousCheckpoint->second;
    replayBegin = previousCheckpoint->first;
  }

  // Build any missing checkpoints on the way to the one preceding this query.
  // Their immutable program-order state can be reused by later queries.
  for (size_t i = replayBegin; i < checkpointTarget; ++i) {
    if (operations[i]->getParentOfType<FuncDefOp>().getOperation() == scopeOp) {
      snapshot.storage.materializeOperation(
          *this, operations[i], snapshot.aliases, /*forceMayBeSkipped=*/false
      );
    }
    if ((i + 1) % checkpointStride == 0) {
      scopeCheckpoints.try_emplace(i + 1, snapshot);
    }
  }

  // The remaining replay is bounded by the checkpoint stride.
  for (size_t i = checkpointTarget; i < target->second; ++i) {
    if (operations[i]->getParentOfType<FuncDefOp>().getOperation() == scopeOp) {
      snapshot.storage.materializeOperation(
          *this, operations[i], snapshot.aliases, /*forceMayBeSkipped=*/false
      );
    }
  }

  // A read in a loop body can observe a write lexically after it from a prior iteration. Replay
  // every enclosing loop body as a weak update: the entry state remains possible (the loop may
  // execute zero times), while its writes form the loop backedge alternative.
  for (Operation *ancestor = before->getParentOp(); ancestor != nullptr;
       ancestor = ancestor->getParentOp()) {
    if (!llvm::isa<scf::ForOp, scf::WhileOp>(ancestor)) {
      continue;
    }
    (void)ancestor->walk([&ancestor, &snapshot, this](Operation *op) {
      if (op != ancestor) {
        snapshot.storage.materializeOperation(
            *this, op, snapshot.aliases, /*forceMayBeSkipped=*/true, ancestor
        );
      }
    });
  }
  return snapshot;
}

void SourceRefAnalysis::StorageState::invalidateCheckpointsFrom(Operation *op) {
  auto position = operationOrder.find(op);
  if (position == operationOrder.end()) {
    return;
  }
  // A checkpoint immediately before this operation excludes its effects, while
  // every later checkpoint includes them and must be reconstructed on a revisit.
  Operation *scope = op->getParentOfType<FuncDefOp>().getOperation();
  auto &scopeCheckpoints = checkpoints[scope];
  scopeCheckpoints.erase(scopeCheckpoints.upper_bound(position->second), scopeCheckpoints.end());
}

void SourceRefAnalysis::StorageState::MaterializedStorage::indexAddress(const SourceRef &address) {
  if (auto root = address.getRoot(); succeeded(root)) {
    valuesByRoot[*root].push_back(address);
  }
}

void SourceRefAnalysis::StorageState::MaterializedStorage::unindexAddress(
    const SourceRef &address
) {
  auto root = address.getRoot();
  if (failed(root)) {
    return;
  }
  auto entries = valuesByRoot.find(*root);
  ensure(entries != valuesByRoot.end(), "storage root index is out of sync");
  auto *addressIt = llvm::find(entries->second, address);
  ensure(addressIt != entries->second.end(), "storage root index is out of sync");
  entries->second.erase(addressIt);
  if (entries->second.empty()) {
    valuesByRoot.erase(entries);
  }
}

void SourceRefAnalysis::StorageState::MaterializedStorage::eraseValue(const SourceRef &address) {
  unindexAddress(address);
  values.erase(address);
  skippedFirstWrites.erase(address);
  coveredRanges.erase(address);
}

void SourceRefAnalysis::StorageState::MaterializedStorage::invalidateOverlappingWrites(
    const SourceRef &address
) {
  // A definite point write or whole-range copy supersedes earlier writes in its intersection.
  // Preserve the residual slabs and the original source coordinates of delayed copies.
  // SourceRef's generated move uses the same guarded libstdc++ variant move path.
  // NOLINTNEXTLINE(bugprone-exception-escape)
  struct RangeReplacement {
    SourceRef address;
    StorageValue value;
    bool wasSkippedFirstWrite;
    bool coveredRange;
  };

  SmallVector<RangeReplacement> replacements;
  SmallVector<SourceRef> erasedAddresses;
  for (const auto &[storedAddress, storedValue] : values) {
    if ((!hasRangeIndex(storedAddress) && !hasRangeIndex(address)) ||
        !storedAddress.overlaps(address)) {
      continue;
    }
    erasedAddresses.push_back(storedAddress);
    for (const SourceRef &residual : subtractAddress(storedAddress, address)) {
      // A weak write may carry its destination as the unwritten alternative. Rebase that
      // alternative to the residual range too; otherwise a later read would retain the
      // original, still-overlapping range through the value rather than the map key.
      StorageValue residualValue = storedValue;
      residualValue.dependencies =
          storedValue.dependencies
              .replacePrefixes(TranslationMap {{storedAddress, SourceRefLatticeValue(residual)}})
              .first;
      replacements.push_back(
          {residual, std::move(residualValue), skippedFirstWrites.contains(storedAddress),
           coveredRanges.contains(storedAddress)}
      );
    }
  }
  for (const SourceRef &erasedAddress : erasedAddresses) {
    eraseValue(erasedAddress);
  }
  for (const RangeReplacement &replacement : replacements) {
    auto [it, inserted] = values.try_emplace(replacement.address, replacement.value);
    if (inserted) {
      indexAddress(replacement.address);
    }
    if (!inserted) {
      it->second.update(replacement.value);
    }
    if (replacement.wasSkippedFirstWrite) {
      skippedFirstWrites.insert(replacement.address);
    }
    if (replacement.coveredRange) {
      coveredRanges.insert(replacement.address);
    }
  }
}

void SourceRefAnalysis::StorageState::MaterializedStorage::applyWrite(
    const SourceRef &address, const StorageValue &value, bool maySkip,
    bool seedUnwrittenAlternative, bool coversRange
) {
  if (seedUnwrittenAlternative) {
    // A first entry at this exact key can still have prior contents through a covering copy.
    // In particular, a loop's weak point write must not invent an uninitialized copied slot.
    SmallVector<SourceRef> uncovered {address};
    for (const SourceRef &covered : coveredRanges) {
      SmallVector<SourceRef> remaining;
      for (const SourceRef &part : uncovered) {
        if (part.overlaps(covered)) {
          remaining.append(subtractAddress(part, covered));
        } else {
          remaining.push_back(part);
        }
      }
      uncovered = std::move(remaining);
      if (uncovered.empty()) {
        break;
      }
    }
    seedUnwrittenAlternative = !uncovered.empty();
  }
  // A skipped first write can leave the storage unwritten. Seed the stored state with both
  // alternatives so reads retain the address dependency along that path.
  StorageValue initialValue = value;
  if (seedUnwrittenAlternative) {
    (void)initialValue.dependencies.insert(address);
  }
  auto [it, inserted] = values.try_emplace(address, std::move(initialValue));
  if (inserted) {
    indexAddress(address);
  }
  if (inserted && seedUnwrittenAlternative) {
    skippedFirstWrites.insert(address);
  }
  if (!inserted) {
    if (maySkip) {
      it->second.update(value);
    } else {
      it->second = value;
    }
  }
  if (coversRange && (!maySkip || !seedUnwrittenAlternative)) {
    coveredRanges.insert(address);
  }
}

void SourceRefAnalysis::StorageState::MaterializedStorage::materializeWrite(
    const SourceRef &address, const StorageValue &value, bool maySkip,
    bool seedUnwrittenAlternative, bool resolveSelfReference, const TranslationMap &aliases,
    bool coversRange
) {
  StorageValue canonicalValue = value;
  canonicalValue.dependencies = StorageState::canonicalize(value.dependencies, aliases);
  auto &dependencies = canonicalValue.dependencies;
  auto arrayType = llvm::dyn_cast<ArrayType>(address.getType());
  if (dependencies.isArray() && arrayType && arrayType.hasStaticShape() &&
      std::cmp_equal(dependencies.getArraySize(), arrayType.getNumElements())) {
    ensure(value.copies.empty(), "deferred copies must already select scalar leaves");
    for (size_t i = 0; i < dependencies.getArraySize(); ++i) {
      materializeWrite(
          getArrayElementAddress(address, i), dependencies.getElemFlatIdx(i), maySkip,
          seedUnwrittenAlternative, resolveSelfReference, aliases
      );
    }
    return;
  }
  if (llvm::isa<StructType>(address.getType()) && dependencies.isScalar() &&
      dependencies.getScalarValue().contains(address)) {
    if (auto preWrite = values.find(address); resolveSelfReference && preWrite != values.end()) {
      // The self-reference denotes the value read before this write. Substitute the
      // materialized pre-write contents so a read-modify-write retains those dependencies.
      (void)dependencies.getScalarValue().erase(address);
      canonicalValue.update(preWrite->second);
    } else {
      const bool hasPriorContents = llvm::any_of(values, [&address](const auto &entry) {
        return entry.first.isValidPrefix(address);
      });
      if (hasPriorContents || dependencies.isSingleValue()) {
        (void)dependencies.getScalarValue().erase(address);
        if (dependencies.getScalarValue().empty() && canonicalValue.copies.empty()) {
          return;
        }
      }
    }
  }
  if (!maySkip && (!hasRangeIndex(address) || coversRange)) {
    invalidateOverlappingWrites(address);
  }
  applyWrite(address, canonicalValue, maySkip, seedUnwrittenAlternative, coversRange);
}

void SourceRefAnalysis::StorageState::MaterializedStorage::materializeAddressedWrite(
    const SourceRefLatticeValue &addresses, const StorageValue &value, bool maySkip,
    bool seedUnwrittenAlternative, const TranslationMap &aliases, Operation *replayedLoop,
    bool coversRange
) {
  SourceRefLatticeValue canonicalAddresses = StorageState::canonicalize(addresses, aliases);
  if (canonicalAddresses.isArray() && value.dependencies.isArray() &&
      canonicalAddresses.getArraySize() == value.dependencies.getArraySize()) {
    ensure(value.copies.empty(), "deferred copies must already select scalar leaves");
    for (size_t i = 0; i < canonicalAddresses.getArraySize(); ++i) {
      materializeAddressedWrite(
          canonicalAddresses.getElemFlatIdx(i), value.dependencies.getElemFlatIdx(i), maySkip,
          seedUnwrittenAlternative, aliases, replayedLoop, coversRange
      );
    }
    return;
  }
  for (const SourceRef &address : canonicalAddresses.foldToScalar()) {
    if (replayedLoop && isAllocatedWithinLoop(address, replayedLoop)) {
      continue;
    }
    // Canonicalization can expand one write address into several possible alias targets. Each
    // target must retain its old contents because the write affects only one at runtime.
    const bool expandedAliasTargets = !canonicalAddresses.isSingleValue();
    const bool unknownPrefix = coversRange && !value.copies.empty() &&
                               llvm::any_of(
                                   address.getPath().drop_back(value.copies.front().depth),
                                   [](const auto &index) { return index.isIndexRange(); }
                               );
    materializeWrite(
        address, value, maySkip || expandedAliasTargets || unknownPrefix,
        seedUnwrittenAlternative || expandedAliasTargets || unknownPrefix,
        /*resolveSelfReference=*/true, aliases, coversRange && !unknownPrefix
    );
  }
}

void SourceRefAnalysis::StorageState::MaterializedStorage::materializeOperation(
    const StorageState &state, Operation *op, TranslationMap &aliases, bool forceMayBeSkipped,
    Operation *replayedLoop
) {
  DenseSet<SourceRef> priorAliasSources;
  for (const auto &[source, _] : aliases) {
    priorAliasSources.insert(source);
  }
  state.applyAggregateAliases(op, aliases, forceMayBeSkipped);

  // A component-handle assignment gives the same instance another storage location. Rebase only
  // aliases introduced by this operation: applying the complete alias map here would incorrectly
  // move older writes in response to later assignments.
  SmallVector<std::pair<SourceRef, SourceRefLatticeValue>> newAliases;
  for (const auto &[source, targets] : aliases) {
    if (!priorAliasSources.contains(source)) {
      newAliases.emplace_back(source, targets);
    }
  }
  for (const auto &[source, targets] : newAliases) {
    const bool mayBeSkipped =
        forceMayBeSkipped || (targets.isScalar() && targets.getScalarValue().contains(source));
    SmallVector<std::pair<SourceRef, StorageValue>> sourceContents;
    for (const auto &[address, value] : values) {
      if (address.isValidPrefix(source)) {
        sourceContents.emplace_back(address, value);
      }
    }
    for (const auto &[address, value] : sourceContents) {
      for (const SourceRef &target : targets.foldToScalar()) {
        if (replayedLoop && isAllocatedWithinLoop(target, replayedLoop)) {
          continue;
        }
        auto rebasedAddress = address.translate(source, target);
        if (succeeded(rebasedAddress)) {
          materializeWrite(
              *rebasedAddress, value, mayBeSkipped,
              /*seedUnwrittenAlternative=*/false,
              /*resolveSelfReference=*/false, aliases, coveredRanges.contains(address)
          );
        }
      }
    }
  }

  auto writes = state.storageWrites.find(op);
  if (writes == state.storageWrites.end()) {
    return;
  }
  for (const auto &[index, group] : writes->second) {
    for (const StorageWrite &write : group) {
      materializeAddressedWrite(
          write.addresses, write.value, forceMayBeSkipped || write.mayBeSkipped,
          forceMayBeSkipped || write.seedUnwrittenAlternative, aliases, replayedLoop,
          write.coversRange
      );
    }
  }
}

void SourceRefAnalysis::StorageState::recordAggregateAlias(
    Operation *op, size_t aliasIndex, const SourceRefLatticeValue &source,
    const SourceRefLatticeValue &target, bool mayBeSkipped
) {
  auto &aliases = aggregateAliases[op];
  AggregateAlias alias {source, target, mayBeSkipped};
  if (aliasIndex == aliases.size()) {
    aliases.push_back(std::move(alias));
    notifyChanged(op);
    return;
  }
  ensure(aliasIndex < aliases.size(), "aggregate aliases must use stable operation order");
  if (aliases[aliasIndex] != alias) {
    aliases[aliasIndex] = std::move(alias);
    notifyChanged(op);
  }
}

void SourceRefAnalysis::StorageState::applyAggregateAliases(
    Operation *op, TranslationMap &aliases, bool forceMayBeSkipped
) const {
  std::function<void(const SourceRefLatticeValue &, const SourceRefLatticeValue &, bool)> addAlias =
      [&](const SourceRefLatticeValue &source, const SourceRefLatticeValue &target,
          bool mayBeSkipped) {
    SourceRefLatticeValue canonicalSource = canonicalize(source, aliases);
    SourceRefLatticeValue canonicalTarget = canonicalize(target, aliases);
    if (canonicalSource.isArray() && canonicalTarget.isSingleValue()) {
      const SourceRef &targetRoot = canonicalTarget.getSingleValue();
      auto targetType = llvm::dyn_cast<ArrayType>(targetRoot.getType());
      if (!targetType || !targetType.hasStaticShape() ||
          !std::cmp_equal(canonicalSource.getArraySize(), targetType.getNumElements())) {
        return;
      }
      for (size_t i = 0; i < canonicalSource.getArraySize(); ++i) {
        addAlias(
            canonicalSource.getElemFlatIdx(i),
            SourceRefLatticeValue(getArrayElementAddress(targetRoot, i)), mayBeSkipped
        );
      }
      return;
    }
    if (!canonicalSource.isScalar() || !canonicalTarget.isSingleValue()) {
      return;
    }

    const SourceRef &targetRef = canonicalTarget.getSingleValue();
    for (const SourceRef &sourceRef : canonicalSource.getScalarValue()) {
      if (sourceRef == targetRef || !sourceRef.isRooted() ||
          !llvm::isa<StructType>(sourceRef.getType())) {
        continue;
      }
      auto sourceRoot = sourceRef.getRoot();
      if (failed(sourceRoot)) {
        continue;
      }
      // An SSA handle or captured initial component identifies an instance independently
      // of every slot holding it. Rebasing it onto a replaceable array/POD slot would
      // make an already captured handle follow a later assignment to that slot.
      if (llvm::isa<StructType>(sourceRoot->getType()) ||
          sourceRef.protectsCapturedComponent(sourceRef)) {
        continue;
      }
      Operation *defOp = sourceRoot->getDefiningOp();
      if (defOp == nullptr || !llvm::isa<CallOp, NewPodOp, CreateArrayOp, NonDetOp>(defOp)) {
        continue;
      }
      SourceRefLatticeValue aliasTargets = canonicalTarget;
      if (mayBeSkipped) {
        (void)aliasTargets.insert(sourceRef);
      }
      aliases.set(sourceRef, std::move(aliasTargets));
    }
  };

  auto events = aggregateAliases.find(op);
  if (events != aggregateAliases.end()) {
    for (const AggregateAlias &event : events->second) {
      addAlias(event.source, event.target, forceMayBeSkipped || event.mayBeSkipped);
    }
  }
}

FailureOr<SourceRefLatticeValue>
SourceRefAnalysis::getWriteTargetState(DataFlowSolver &solver, Operation *op) {
  llvm::SmallDenseMap<Value, SourceRefLatticeValue, 4> operandVals;
  for (Value operand : op->getOperands()) {
    operandVals[operand] = getValueState(solver, operand);
  }

  SymbolTableCollection tables;
  if (auto memberRefOp = llvm::dyn_cast<MemberRefOpInterface>(op)) {
    if (!memberRefOp.isRead()) {
      auto memberOpRes = memberRefOp.getMemberDefOp(tables);
      ensure(succeeded(memberOpRes), "could not find member write");
      auto componentIt = operandVals.find(memberRefOp.getComponent());
      ensure(componentIt != operandVals.end(), "missing component lattice for member write");
      auto memberValsRes = componentIt->second.referenceMember(memberOpRes.value());
      ensure(succeeded(memberValsRes), "could not create SourceRef child for member write");
      return memberValsRes->first;
    }
  }

  if (auto podAccessOp = llvm::dyn_cast<PodAccessOpInterface>(op)) {
    if (!podAccessOp.isRead()) {
      auto podIt = operandVals.find(podAccessOp.getPodRef());
      ensure(podIt != operandVals.end(), "missing pod lattice for pod write");
      auto podValsRes = podIt->second.referencePodRecord(podAccessOp.getRecordNameAttr());
      ensure(succeeded(podValsRes), "could not create SourceRef child for pod write");
      return podValsRes->first;
    }
  }

  if (auto arrayAccessOp = llvm::dyn_cast<ArrayAccessOpInterface>(op)) {
    if (llvm::isa<WriteArrayOp, InsertArrayOp>(arrayAccessOp)) {
      auto array = arrayAccessOp.getArrRef();
      auto it = operandVals.find(array);
      ensure(it != operandVals.end(), "improperly constructed operandVals map");
      const auto &currVals = it->second;

      auto indices =
          getArrayAccessIndices(arrayAccessOp, [&](Value index) -> const SourceRefLatticeValue & {
        auto idxIt = operandVals.find(index);
        ensure(idxIt != operandVals.end(), "improperly constructed operandVals map");
        return idxIt->second;
      });

      auto newValsRes = currVals.extract(indices);
      ensure(succeeded(newValsRes), "could not create SourceRef child for array access");
      auto [newVals, _] = *newValsRes;
      if (llvm::isa<WriteArrayOp>(arrayAccessOp)) {
        ensure(newVals.isScalar(), "array write must produce a scalar value");
      }
      return newVals;
    }
  }

  return failure();
}

void SourceRefAnalysis::setToEntryState(Lattice *lattice) {
  if (Value value = lattice->getAnchor()) {
    if (auto arg = llvm::dyn_cast<BlockArgument>(value)) {
      Operation *parent = arg.getOwner()->getParentOp();
      if (llvm::isa_and_present<RegionBranchOpInterface>(parent) &&
          llvm::isa<ArrayType, StructType, PodType>(value.getType())) {
        // Region-branch arguments are aliases of their incoming aggregate storage. Giving them a
        // fresh root would make loop-carried writes unstable and would discard that identity.
        propagateIfChanged(lattice, lattice->setValue(SourceRefLatticeValue()));
        return;
      }
    }
    // Forward-defined callees may already have callers waiting on their argument state.
    propagateIfChanged(lattice, lattice->setValue(SourceRefLattice::getDefaultValue(value)));
  }
}

LogicalResult SourceRefAnalysis::visitOperation(
    Operation *op, ArrayRef<const Lattice *> operands, ArrayRef<Lattice *> results
) {
  LLVM_DEBUG(llvm::dbgs() << "SourceRefAnalysis::visitOperation: " << *op << '\n');
  DenseMap<Value, const Lattice *> operandVals;
  for (auto [operand, lattice] : llvm::zip(op->getOperands(), operands)) {
    operandVals[operand] = lattice;
  }
  if (auto memberRefOp = llvm::dyn_cast<MemberRefOpInterface>(op)) {
    auto memberOpRes = memberRefOp.getMemberDefOp(tables);
    ensure(succeeded(memberOpRes), "could not find member read");
    // A copied struct handle keeps the component observed when it was read, even
    // if its former array/POD slot is overwritten before this member access.
    // Subscribe to storage updates so late callee summaries also refresh that identity.
    (void)getStorageState(op);
    auto memberValsRes =
        getStoredValue(memberRefOp.getComponent()).referenceMember(memberOpRes.value());
    ensure(succeeded(memberValsRes), "could not create SourceRef child for member reference");
    if (memberRefOp.isRead()) {
      auto [memberVals, _] = *memberValsRes;
      if (isMutableAggregateType(op->getResult(0).getType())) {
        memberVals = recordReadCopy(op, memberVals);
      }
      propagateIfChanged(results.front(), results.front()->setValue(memberVals));
    } else {
      auto [memberVals, _] = *memberValsRes;
      auto writeOp = llvm::cast<MemberWriteOp>(op);
      SourceRefLatticeValue writeValue = getStoredValue(writeOp.getVal());
      if (llvm::isa<ArrayType, StructType, PodType>(writeOp.getVal().getType())) {
        // A component value that merges multiple storage roots selects exactly
        // one root at runtime. Model the write as weak for every candidate so
        // the unselected roots retain their pre-write contents.
        const bool mayBeSkipped = isInMaybeSkippedScfRegion(op) || !memberVals.isSingleValue();
        getStorageState(op)->recordStorageWrite(
            op, /*writeIndex=*/0, memberVals, writeValue, writeOp.getVal().getType(), mayBeSkipped,
            /*seedUnwrittenAlternative=*/mayBeSkipped
        );
        getStorageState(op)->recordAggregateAlias(
            op, /*aliasIndex=*/0, writeValue, memberVals, mayBeSkipped
        );
      }
    }
    return success();
  }

  if (auto podAccessOp = llvm::dyn_cast<PodAccessOpInterface>(op)) {
    auto podValsRes = operandVals.at(podAccessOp.getPodRef())
                          ->getValue()
                          .referencePodRecord(podAccessOp.getRecordNameAttr());
    ensure(succeeded(podValsRes), "could not create SourceRef child for pod reference");
    if (podAccessOp.isRead()) {
      auto [podVals, _] = *podValsRes;
      if (isMutableAggregateType(op->getResult(0).getType())) {
        podVals = recordReadCopy(op, podVals);
      }
      propagateIfChanged(results.front(), results.front()->setValue(podVals));
    } else {
      auto [podVals, _] = *podValsRes;
      auto writeOp = llvm::cast<WritePodOp>(op);
      SourceRefLatticeValue writeValue = getStoredValue(writeOp.getValue());
      // A POD value that merges multiple storage roots selects exactly one root at runtime.
      // Model the write as weak for every candidate so the unselected roots retain their
      // pre-write contents.
      const bool mayBeSkipped = isInMaybeSkippedScfRegion(op) || !podVals.isSingleValue();
      getStorageState(op)->recordStorageWrite(
          op, /*writeIndex=*/0, podVals, writeValue, writeOp.getValue().getType(), mayBeSkipped,
          /*seedUnwrittenAlternative=*/mayBeSkipped
      );
      if (llvm::isa<ArrayType, StructType, PodType>(writeOp.getValue().getType())) {
        getStorageState(op)->recordAggregateAlias(
            op, /*aliasIndex=*/0, writeValue, podVals, mayBeSkipped
        );
      }
    }
    return success();
  }

  if (auto arrayAccessOp = llvm::dyn_cast<ArrayAccessOpInterface>(op)) {
    if (llvm::isa<WriteArrayOp, InsertArrayOp>(arrayAccessOp)) {
      SourceRefLatticeValue writeTargets = arraySubdivisionOpUpdate(arrayAccessOp, operandVals);
      Value rvalue = op->getOperands().back();
      SourceRefLatticeValue writeValue = getStoredValue(rvalue);
      const bool mayBeSkipped =
          isInMaybeSkippedScfRegion(op) || isNonSingletonArrayWriteTarget(writeTargets);
      getStorageState(op)->recordStorageWrite(
          op, /*writeIndex=*/0, writeTargets, writeValue, rvalue.getType(), mayBeSkipped,
          /*seedUnwrittenAlternative=*/mayBeSkipped
      );
      if (llvm::isa<ArrayType, StructType, PodType>(rvalue.getType())) {
        getStorageState(op)->recordAggregateAlias(
            op, /*aliasIndex=*/0, writeValue, writeTargets, mayBeSkipped
        );
      }
    }
    if (!results.empty()) {
      auto newVals = arraySubdivisionOpUpdate(arrayAccessOp, operandVals);
      if (llvm::isa<ReadArrayOp>(op) && isMutableAggregateType(op->getResult(0).getType())) {
        newVals = recordReadCopy(op, newVals);
      }
      propagateIfChanged(results.front(), results.front()->setValue(newVals));
    }
    return success();
  }

  if (auto createArray = llvm::dyn_cast<CreateArrayOp>(op)) {
    auto createArrayRes = createArray.getResult();
    SourceRef arrayRoot(llvm::cast<OpResult>(createArrayRes));
    auto arrayType = createArray.getType();

    if (arrayType.hasStaticShape()) {
      SourceRefLatticeValue newArrayValue(arrayType.getShape());
      for (size_t i = 0; i < newArrayValue.getArraySize(); ++i) {
        (void)newArrayValue.getElemFlatIdx(i).setValue(
            SourceRefLatticeValue(getArrayElementAddress(arrayRoot, i))
        );
      }
      propagateIfChanged(results.front(), results.front()->setValue(newArrayValue));
    } else {
      SourceRefLatticeValue newArrayValue(arrayRoot);
      propagateIfChanged(results.front(), results.front()->setValue(newArrayValue));
    }

    const auto &elements = createArray.getElements();
    if (elements.empty()) {
      return success();
    }

    const bool mayBeSkipped = isInMaybeSkippedScfRegion(op);
    StorageState *state = getStorageState(op);
    size_t aggregateAliasIndex = 0;
    for (auto [i, element] : llvm::enumerate(elements)) {
      SourceRef elementAddress = getArrayElementAddress(arrayRoot, i);
      SourceRefLatticeValue elementAddressValue(elementAddress);
      SourceRefLatticeValue elementValue = getStoredValue(element);
      state->recordStorageWrite(
          op, i, elementAddressValue, elementValue, element.getType(), mayBeSkipped
      );
      if (llvm::isa<ArrayType, StructType, PodType>(element.getType())) {
        state->recordAggregateAlias(
            op, aggregateAliasIndex++, elementValue, elementAddressValue, mayBeSkipped
        );
      }
    }
    return success();
  }

  if (auto newPod = llvm::dyn_cast<NewPodOp>(op)) {
    auto newPodValue = SourceRefLattice::getDefaultValue(newPod.getResult());
    propagateIfChanged(results.front(), results.front()->setValue(newPodValue));
    SourceRef podRoot(llvm::cast<OpResult>(newPod.getResult()));
    size_t aggregateAliasIndex = 0;
    for (auto [idx, record] : llvm::enumerate(newPod.getInitializedRecordValues())) {
      auto recordRef =
          podRoot.createChild(SourceRefIndex(StringAttr::get(op->getContext(), record.name)));
      ensure(succeeded(recordRef), "could not create initialized POD record SourceRef");
      SourceRefLatticeValue recordValue = getStoredValue(record.value);
      StorageState *state = getStorageState(op);
      SourceRefLatticeValue recordAddress(*recordRef);
      const bool mayBeSkipped = isInMaybeSkippedScfRegion(op);
      state->recordStorageWrite(
          op, idx, recordAddress, recordValue, record.value.getType(), mayBeSkipped
      );
      if (llvm::isa<ArrayType, StructType, PodType>(record.value.getType())) {
        state->recordAggregateAlias(
            op, aggregateAliasIndex++, recordValue, recordAddress, mayBeSkipped
        );
      }
    }
    return success();
  }

  if (auto structNewOp = llvm::dyn_cast<CreateStructOp>(op)) {
    auto newStructValue = SourceRefLattice::getDefaultValue(structNewOp.getResult());
    propagateIfChanged(results.front(), results.front()->setValue(newStructValue));
    return success();
  }

  auto updated = fallbackOpUpdate(op, operandVals, results);
  for (Lattice *result : results) {
    propagateIfChanged(result, updated);
  }
  return success();
}

LogicalResult SourceRefAnalysis::validateInput(ModuleOp module) {
  auto result = module.walk([](Operation *op) -> WalkResult {
    auto call = llvm::dyn_cast<CallOpInterface>(op);
    if (!call || llvm::none_of(call.getArgOperands().getTypes(), containsComponentHandle)) {
      return WalkResult::advance();
    }
    auto callable = llvm::dyn_cast_if_present<CallableOpInterface>(call.resolveCallable());
    if (callable && callable.getCallableRegion()) {
      return WalkResult::advance();
    }
    return op->emitOpError(
        "cannot analyze an opaque call with component handles in its arguments; "
        "callee storage effects are unknown"
    );
  });
  return failure(result.wasInterrupted());
}

void SourceRefAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const Lattice *> operandLattices,
    ArrayRef<Lattice *> resultLattices
) {
  auto callable = dyn_cast_if_present<CallableOpInterface>(call.resolveCallable());
  if (!callable || !callable.getCallableRegion()) {
    // Call is truly external
    for (auto [result, lattice] : llvm::zip(call->getResults(), resultLattices)) {
      auto resultRef = SourceRefLattice::getSourceRef(result);
      ensure(succeeded(resultRef), "could not create external call SourceRef");
      SourceRefLatticeValue resultValue(*resultRef);
      propagateIfChanged(lattice, lattice->setValue(resultValue));
    }
    return;
  }
  // Call is to a defined function with a body, but it's treated as external so we
  // can translate the results based on the arguments.
  auto funcOpRes = resolveCallable<FuncDefOp>(tables, call);
  if (failed(funcOpRes) || !*funcOpRes) {
    // Other callable operations, such as verif.contract, can have a body
    // without defining function storage effects to translate. In particular,
    // resultless calls have no value lattice to update either.
    return;
  }

  TranslationMap storageTranslation;
  FuncDefOp funcOp = funcOpRes->get();
  // Storage effects are independent of return-value lattices, especially for void calls.
  // Subscribe before copying the summary so forward definitions and nested calls converge.
  (void)getOrCreateFor<StorageState::ScopeState>(
      getProgramPointAfter(call.getOperation()), getProgramPointBefore(funcOp.getOperation())
  );
  for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
    SourceRefLatticeValue argumentValue =
        static_cast<const Lattice *>(operandLattices[i])->getValue();
    const bool isAggregate =
        llvm::isa<ArrayType, StructType, PodType>(call->getOperand(i).getType());
    if (!isAggregate) {
      argumentValue = getDependencyState(dataflowSolver_, call->getOperand(i));
    }
    SourceRef argumentRef(funcOp.getArgument(i));
    // Aggregate value lattices are shaped by their elements. Storage effects,
    // however, need the aggregate root so a callee selection is appended only
    // once (e.g. `%arg[0]`, never `%arg[0][0]`).
    if (llvm::isa<StructType>(call->getOperand(i).getType())) {
      storageTranslation.set(argumentRef, getDependencyState(dataflowSolver_, call->getOperand(i)));
    } else if (isAggregate) {
      auto blockArgument = llvm::dyn_cast<BlockArgument>(call->getOperand(i));
      const bool isRegionArgument =
          blockArgument &&
          llvm::isa_and_present<RegionBranchOpInterface>(blockArgument.getOwner()->getParentOp());
      auto storageArgumentRef = SourceRefLattice::getSourceRef(call->getOperand(i));
      // Region arguments inherit storage from incoming values, as in setToEntryState.
      // A fresh block-argument root would discard those contents at the call boundary.
      if (!isRegionArgument && succeeded(storageArgumentRef)) {
        storageTranslation.set(argumentRef, SourceRefLatticeValue(*storageArgumentRef));
      } else if (argumentValue.isScalar()) {
        // Scalar lattices name the selected aggregate, including every incoming
        // storage alternative for a region argument.
        storageTranslation.set(argumentRef, argumentValue);
      } else if (auto arrayType = llvm::dyn_cast<ArrayType>(call->getOperand(i).getType())) {
        // Propagated arrays and array.extract values have one lattice element per
        // selected leaf. Recover the selected subarray address by removing the
        // dimensions belonging to the argument type from every leaf. This keeps
        // the extract path while avoiding a duplicated callee index during
        // translation (e.g. `%matrix[1]` + callee `[0]`, not `%matrix[1][0][0]`).
        SourceRefLatticeValue storageArgumentValue;
        for (const SourceRef &leaf : argumentValue.foldToScalar()) {
          SourceRef selectedAggregate = leaf;
          bool recovered = true;
          for (int64_t dim = 0; dim < arrayType.getRank(); ++dim) {
            auto parent = selectedAggregate.getParentPrefix();
            if (failed(parent)) {
              recovered = false;
              break;
            }
            selectedAggregate = *parent;
          }
          if (recovered) {
            (void)storageArgumentValue.insert(selectedAggregate);
          }
        }
        if (!storageArgumentValue.foldToScalar().empty()) {
          storageTranslation.set(argumentRef, std::move(storageArgumentValue));
        }
      }
    } else {
      storageTranslation.set(argumentRef, argumentValue);
    }
  }
  getStorageState(call.getOperation())->recordCalleeStorageWrites(call, funcOp, storageTranslation);

  if (resultLattices.empty()) {
    // Any shared component effects have been recorded above. Array/POD writes
    // remain local, and there are no result values to translate back to the caller.
    return;
  }

  const auto *predecessors = getOrCreateFor<mlir::dataflow::PredecessorState>(
      getProgramPointAfter(call), getProgramPointAfter(call)
  );
  // If not all return sites are known, then conservatively assume we can't
  // reason about the data-flow.
  if (!predecessors->allPredecessorsKnown()) {
    setAllToEntryStates(resultLattices);
    return;
  }
  const auto returnSites = predecessors->getKnownPredecessors();

  for (auto [result, resultLattice] : llvm::zip(call->getResults(), resultLattices)) {
    if (isMutableAggregateType(result.getType())) {
      SmallVector<std::pair<Operation *, SourceRefLatticeValue>> returnValues;
      for (Operation *returnSite : returnSites) {
        const auto *returned = getLatticeElementFor(
            getProgramPointAfter(call.getOperation()),
            returnSite->getOperand(result.getResultNumber())
        );
        returnValues.emplace_back(returnSite, returned->getValue());
      }
      SourceRefLatticeValue references =
          getStorageState(call.getOperation())
              ->recordReturnedAggregate(result, returnValues, storageTranslation);
      propagateIfChanged(resultLattice, resultLattice->setValue(references));
      continue;
    }
    SourceRefLatticeValue combined;
    unsigned resultNum = result.getResultNumber();
    for (Operation *returnSite : returnSites) {
      (void)getLatticeElementFor(
          getProgramPointAfter(call.getOperation()), returnSite->getOperand(resultNum)
      );
      // Struct handles and scalars both retain the observation at their SSA definition.
      // Re-reading an unresolved handle's slot at the return would follow later overwrites.
      SourceRefLatticeValue returnDependencies =
          getDependencyState(dataflowSolver_, returnSite->getOperand(resultNum));
      auto translatedVal = getStorageState(call.getOperation())
                               ->translateCallDependencies(
                                   returnDependencies, storageTranslation, call.getOperation()
                               );
      (void)combined.update(translatedVal);
    }
    propagateIfChanged(resultLattice, resultLattice->setValue(combined));
  }
}

ChangeResult SourceRefAnalysis::fallbackOpUpdate(
    Operation *op, const OperandValues &operandVals, ArrayRef<Lattice *> results
) {
  auto updated = ChangeResult::NoChange;
  for (auto [res, lattice] : llvm::zip(op->getResults(), results)) {
    auto cur = SourceRefLattice::getDefaultValue(res);
    for (const auto &[_, opVal] : operandVals) {
      (void)cur.update(opVal->getValue());
    }
    updated |= lattice->setValue(cur);
  }
  return updated;
}

SourceRefLatticeValue SourceRefAnalysis::arraySubdivisionOpUpdate(
    ArrayAccessOpInterface arrayAccessOp, const OperandValues &operandVals
) {
  auto array = arrayAccessOp.getArrRef();
  auto it = operandVals.find(array);
  ensure(it != operandVals.end(), "improperly constructed operandVals map");
  const auto &currVals = it->second->getValue();

  auto indices =
      getArrayAccessIndices(arrayAccessOp, [&](Value index) -> const SourceRefLatticeValue & {
    auto idxIt = operandVals.find(index);
    ensure(idxIt != operandVals.end(), "improperly constructed operandVals map");
    return idxIt->second->getValue();
  });

  auto newValsRes = currVals.extract(indices);
  ensure(succeeded(newValsRes), "could not create SourceRef child for array access");
  auto [newVals, _] = *newValsRes;
  if (llvm::isa<ReadArrayOp, WriteArrayOp>(arrayAccessOp)) {
    ensure(newVals.isScalar(), "array read/write must produce a scalar value");
  }
  return newVals;
}

/* ConstraintDependencyGraph */

FailureOr<ConstraintDependencyGraph> ConstraintDependencyGraph::compute(
    ModuleOp m, StructDefOp s, DataFlowSolver &solver, AnalysisManager &am,
    const CDGAnalysisContext &ctx
) {
  ConstraintDependencyGraph cdg(m, s, ctx);
  if (cdg.computeConstraints(solver, am).failed()) {
    return failure();
  }
  return cdg;
}

void ConstraintDependencyGraph::dump() const { print(llvm::errs()); }

/// Print all constraints. Any element that is unconstrained is omitted.
void ConstraintDependencyGraph::print(llvm::raw_ostream &os) const {
  // the EquivalenceClasses::iterator is sorted, but the EquivalenceClasses::member_iterator is
  // not guaranteed to be sorted. So, we will sort members before printing them.
  // We also want to add the constant values into the printing.
  std::set<std::set<SourceRef>> sortedSets;
  for (const auto *entry : signalSets) {
    if (!entry->isLeader()) {
      continue;
    }

    auto members = signalSets.members(*entry);
    std::set<SourceRef> sortedMembers(members.begin(), members.end());

    // We only want to print sets with a size > 1, because size == 1 means the
    // signal is not in a constraint.
    if (sortedMembers.size() > 1) {
      sortedSets.insert(sortedMembers);
    }
  }
  // Add the constants in separately.
  for (const auto &[ref, constSet] : constantSets) {
    if (constSet.empty()) {
      continue;
    }
    std::set<SourceRef> sortedMembers(constSet.begin(), constSet.end());
    sortedMembers.insert(ref);
    sortedSets.insert(sortedMembers);
  }

  os << "ConstraintDependencyGraph { ";

  llvm::interleave(sortedSets, os, [&](const auto &members) {
    os << "\n    { ";
    llvm::interleaveComma(members, os);
    os << " }";
  }, ",");
  if (!sortedSets.empty()) {
    os << '\n';
  }

  os << "}\n";
}

LogicalResult
ConstraintDependencyGraph::computeConstraints(DataFlowSolver &solver, AnalysisManager &am) {
  // Fetch the constrain function. This is a required feature for all LLZK structs.
  FuncDefOp constrainFnOp = structDef.getConstrainFuncOp();
  ensure(
      constrainFnOp,
      "malformed struct " + Twine(structDef.getName()) + " must define a constrain function"
  );

  /**
   * Now, given the analysis, construct the CDG:
   * - Union all references based on solver results.
   * - Union all references based on nested dependencies.
   */

  // - Union all constraints from the analysis
  // This requires iterating over all of the emit operations
  constrainFnOp.walk([this, &solver](Operation *op) {
    if (!dataflow::isOperationLive(solver, op)) {
      return;
    }

    for (Value operand : op->getOperands()) {
      for (const SourceRefLatticeValue &state :
           {SourceRefAnalysis::getValueState(solver, operand),
            SourceRefAnalysis::getDependencyState(solver, operand)}) {
        for (const SourceRef &ref : state.foldToScalar()) {
          ref2Val[ref].insert(operand);
        }
      }
    }
    for (Value result : op->getResults()) {
      for (const SourceRefLatticeValue &state :
           {SourceRefAnalysis::getValueState(solver, result),
            SourceRefAnalysis::getDependencyState(solver, result)}) {
        for (const SourceRef &ref : state.foldToScalar()) {
          ref2Val[ref].insert(result);
        }
      }
    }
    auto writeTargetState = SourceRefAnalysis::getWriteTargetState(solver, op);
    if (succeeded(writeTargetState)) {
      for (const SourceRef &ref : writeTargetState->foldToScalar()) {
        ref2Val[ref].insert(op);
      }
    }
    if (isa<EmitEqualityOp, EmitContainmentOp>(op)) {
      this->walkConstrainOp(solver, op);
    }
  });

  if (!ctx.runIntraproceduralAnalysis()) {
    /**
     * Step two of the analysis is to traverse all of the constrain calls.
     * This is the nested analysis, basically.
     * Constrain functions don't return, so we don't need to compute "values" from
     * the call. We just need to see what constraints are generated here, and
     * add them to the transitive closures.
     */
    auto fnCallWalker = [this, &solver, &am](CallOp fnCall) mutable {
      if (!dataflow::isOperationLive(solver, fnCall.getOperation())) {
        return;
      }
      auto res = resolveCallable<FuncDefOp>(tables, fnCall);
      ensure(succeeded(res), "could not resolve constrain call");

      auto fn = res->get();
      if (!fn.isStructConstrain()) {
        return;
      }
      // Nested
      auto calledStruct = fn.getOperation()->getParentOfType<StructDefOp>();
      SourceRefRemappings translations;

      // Map fn parameters to args in the call op
      for (unsigned i = 0; i < fn.getNumArguments(); i++) {
        SourceRef prefix(fn.getArgument(i));
        Value operand = fnCall.getOperand(i);
        SourceRefLatticeValue val;
        if (isMutableAggregateType(operand.getType())) {
          // Keep the storage address until the complete child path is translated.
          // Resolving array elements here and again below would re-read captured
          // scalar dependencies after subsequent writes to their original slots.
          val = SourceRefAnalysis::getValueState(solver, operand);
        } else {
          // Scalars and struct handles retain their original SSA observation.
          val = SourceRefAnalysis::getDependencyState(solver, operand);
        }
        translations.emplace_back(prefix, val);
      }
      auto &childAnalysis =
          am.getChildAnalysis<ConstraintDependencyGraphStructAnalysis>(calledStruct);
      if (!childAnalysis.constructed(ctx)) {
        ensure(
            succeeded(childAnalysis.runAnalysis(solver, am, {.runIntraprocedural = false})),
            "could not construct CDG for child struct"
        );
      }
      auto translatedCDG = childAnalysis.getResult(ctx).translate(
          translations,
          [&solver, call = fnCall.getOperation()](const SourceRef &prefix, const SourceRef &ref) {
        // Only aggregate arguments expose storage paths to resolve at the call.
        // Scalar mappings already contain the dependencies of their captured value.
        if (isAggregateType(prefix.getType())) {
          const SourceRef entryRef =
              isMutableAggregateType(prefix.getType()) ? ref.withoutComponentCapture() : ref;
          return SourceRefAnalysis::getDependencyState(
              solver, SourceRefLatticeValue(entryRef), call
          );
        }
        return SourceRefLatticeValue(ref);
      }
      );
      // Update the refMap with the translation
      const auto &translatedRef2Val = translatedCDG.getRef2Val();
      ref2Val.insert(translatedRef2Val.begin(), translatedRef2Val.end());

      // Now, union sets based on the translation
      // We should be able to just merge what is in the translatedCDG to the current CDG
      auto &tSets = translatedCDG.signalSets;
      for (const auto *entry : tSets) {
        if (!entry->isLeader()) {
          continue;
        }
        auto leader = entry->getData();
        for (const SourceRef &member : tSets.members(*entry)) {
          signalSets.unionSets(leader, member);
        }
      }
      for (const auto &[source, targets] : translatedCDG.dependencyEdges) {
        dependencyEdges[source].insert(targets.begin(), targets.end());
      }
      for (const auto &[source, targets] : translatedCDG.conditionalDependencyEdges) {
        conditionalDependencyEdges[source].insert(targets.begin(), targets.end());
      }
      auto &translatedAliases = translatedCDG.directAliases;
      for (const auto *leaderIt : translatedAliases) {
        if (!leaderIt->isLeader()) {
          continue;
        }
        const SourceRef &leader = leaderIt->getData();
        for (const SourceRef &member : translatedAliases.members(*leaderIt)) {
          directAliases.unionSets(leader, member);
        }
      }
      // And update the constant sets
      for (auto &[ref, constSet] : translatedCDG.constantSets) {
        constantSets[ref].insert(constSet.begin(), constSet.end());
      }
    };
    constrainFnOp.walk(fnCallWalker);
  }

  return success();
}

void ConstraintDependencyGraph::walkConstrainOp(
    mlir::DataFlowSolver &solver, mlir::Operation *emitOp
) {
  std::vector<SourceRef> signalUsages, constUsages;
  std::vector<SourceRef> conditionalUsages;
  llvm::SmallVector<SourceRefSet, 2> operandSignals;

  // Some frontends preserve a witness-only conditional in @constrain as an empty scf.if marker.
  // Associate that marker's condition with the next emitted constraint in the block.
  for (Operation *previous = emitOp->getPrevNode(); previous != nullptr;
       previous = previous->getPrevNode()) {
    if (llvm::isa<EmitEqualityOp, EmitContainmentOp>(previous)) {
      break;
    }
    auto ifOp = llvm::dyn_cast<scf::IfOp>(previous);
    if (!ifOp) {
      continue;
    }
    auto hasOnlyTerminator = [](Region &region) {
      return region.hasOneBlock() && region.front().without_terminator().empty();
    };
    if (!hasOnlyTerminator(ifOp.getThenRegion()) || !hasOnlyTerminator(ifOp.getElseRegion())) {
      continue;
    }
    for (const SourceRef &conditionRef :
         SourceRefAnalysis::getDependencyState(solver, ifOp.getCondition()).foldToScalar()) {
      if (conditionRef.isConstant() || conditionRef.isTemplateConstant()) {
        constUsages.push_back(conditionRef);
      } else if (!conditionRef.isBlockArgument() || *conditionRef.getInputNum() == 0) {
        conditionalUsages.push_back(conditionRef);
      }
    }
  }

  for (auto operand : emitOp->getOperands()) {
    auto latticeVal = SourceRefAnalysis::getDependencyState(solver, operand, emitOp);
    std::optional<SourceRef> neutralInitializer = getNeutralWhileInitializer(operand);
    SourceRefSet currentOperandSignals;
    for (const auto &ref : latticeVal.foldToScalar()) {
      if (ref.isConstant() || ref.isTemplateConstant()) {
        if (!neutralInitializer || ref != *neutralInitializer) {
          constUsages.push_back(ref);
        }
      } else {
        signalUsages.push_back(ref);
      }
    }
    for (const SourceRef &ref : SourceRefAnalysis::getValueState(solver, operand).foldToScalar()) {
      if (!ref.isConstant() && !ref.isTemplateConstant()) {
        currentOperandSignals.insert(ref);
      }
    }
    operandSignals.push_back(std::move(currentOperandSignals));
  }

  auto isStorageIdentity = [](Value operand) {
    return llvm::isa<BlockArgument>(operand) ||
           llvm::isa_and_present<MemberReadOp, ReadArrayOp, ReadPodOp>(operand.getDefiningOp());
  };
  if (llvm::isa<EmitEqualityOp>(emitOp) && operandSignals.size() == 2 &&
      llvm::all_of(emitOp->getOperands(), isStorageIdentity) && operandSignals[0].size() == 1 &&
      operandSignals[1].size() == 1) {
    const SourceRef &lhs = *operandSignals[0].begin();
    const SourceRef &rhs = *operandSignals[1].begin();
    directAliases.unionSets(lhs, rhs);
  }

  // Compute a transitive closure over the signals.
  if (!signalUsages.empty()) {
    for (const SourceRef &lhs : signalUsages) {
      for (const SourceRef &rhs : signalUsages) {
        if (lhs != rhs) {
          dependencyEdges[lhs].insert(rhs);
        }
      }
    }
    auto leader = signalSets.getOrInsertLeaderValue(signalUsages.front());
    for (const SourceRef &signal : llvm::drop_begin(signalUsages)) {
      signalSets.unionSets(leader, signal);
    }
  }
  for (const SourceRef &conditionRef : conditionalUsages) {
    for (const SourceRef &signalRef : signalUsages) {
      if (conditionRef != signalRef) {
        conditionalDependencyEdges[conditionRef].insert(signalRef);
        conditionalDependencyEdges[signalRef].insert(conditionRef);
      }
    }
  }
  // Also update constant references for each value.
  for (auto &sig : signalUsages) {
    constantSets[sig].insert(constUsages.begin(), constUsages.end());
  }
}

ConstraintDependencyGraph ConstraintDependencyGraph::translate(
    SourceRefRemappings translation,
    const std::function<SourceRefLatticeValue(const SourceRef &, const SourceRef &)> &resolve
) const {
  ConstraintDependencyGraph res(mod, structDef, ctx);
  auto translate = [&translation,
                    &resolve](const SourceRef &elem) -> FailureOr<std::vector<SourceRef>> {
    std::vector<SourceRef> refs;
    auto appendRef = [&](const SourceRef &prefix, const SourceRef &ref) {
      if (!resolve) {
        refs.push_back(ref);
        return;
      }
      SourceRefLatticeValue resolved = resolve(prefix, ref);
      auto folded = resolved.foldToScalar();
      refs.insert(refs.end(), folded.begin(), folded.end());
    };
    for (auto &[prefix, vals] : translation) {
      if (!elem.isValidPrefix(prefix)) {
        continue;
      }

      if (vals.isArray()) {
        // Try to index into the array
        auto suffix = elem.getSuffix(prefix);
        ensure(succeeded(suffix), "failure is nonsensical, we already checked for valid prefix");

        auto resolvedValsRes = vals.extract(suffix.value());
        ensure(succeeded(resolvedValsRes), "could not create SourceRef child while resolving refs");
        auto [resolvedVals, _] = *resolvedValsRes;
        auto folded = resolvedVals.foldToScalar();
        for (const SourceRef &ref : folded) {
          appendRef(prefix, ref);
        }
      } else {
        for (const auto &replacement : vals.getScalarValue()) {
          auto translated = elem.translate(prefix, replacement);
          if (succeeded(translated)) {
            appendRef(prefix, translated.value());
          }
        }
      }
    }
    if (refs.empty()) {
      return failure();
    }
    return refs;
  };

  for (const auto *entry : signalSets) {
    if (!entry->isLeader()) {
      continue;
    }
    // translate everything in this set first
    std::vector<SourceRef> translatedSignals, translatedConsts;
    for (const SourceRef &member : signalSets.members(*entry)) {
      auto translated = translate(member);
      if (mlir::failed(translated)) {
        continue;
      }
      for (const auto &ref : *translated) {
        if (ref.isConstant()) {
          translatedConsts.push_back(ref);
        } else {
          translatedSignals.push_back(ref);
        }
      }
      // Also add the constants from the original CDG
      if (auto it = constantSets.find(member); it != constantSets.end()) {
        const auto &origConstSet = it->second;
        translatedConsts.insert(translatedConsts.end(), origConstSet.begin(), origConstSet.end());
      }
    }

    if (translatedSignals.empty()) {
      continue;
    }

    // Now we can insert the translated signals
    auto leader = translatedSignals.front();
    res.signalSets.insert(leader);
    for (const SourceRef &signal : llvm::drop_begin(translatedSignals)) {
      res.signalSets.insert(signal);
      res.signalSets.unionSets(leader, signal);
    }

    // And update the constant references
    for (auto &ref : translatedSignals) {
      res.constantSets[ref].insert(translatedConsts.begin(), translatedConsts.end());
    }
  }

  // Translate direct and conditional edges with the same filtering.
  auto translateEdges = [&translate](const auto &edges, auto &translatedEdges) {
    for (const auto &[source, targets] : edges) {
      auto translatedSources = translate(source);
      if (failed(translatedSources)) {
        continue;
      }
      for (const SourceRef &target : targets) {
        auto translatedTargets = translate(target);
        if (failed(translatedTargets)) {
          continue;
        }
        for (const SourceRef &translatedSource : *translatedSources) {
          for (const SourceRef &translatedTarget : *translatedTargets) {
            if (!translatedSource.isConstant() && !translatedTarget.isConstant() &&
                translatedSource != translatedTarget) {
              translatedEdges[translatedSource].insert(translatedTarget);
            }
          }
        }
      }
    }
  };
  translateEdges(dependencyEdges, res.dependencyEdges);

  for (const auto *leaderIt : directAliases) {
    if (!leaderIt->isLeader()) {
      continue;
    }
    std::vector<SourceRef> translatedAliases;
    for (auto memberIt = directAliases.member_begin(*leaderIt);
         memberIt != directAliases.member_end(); ++memberIt) {
      auto translated = translate(*memberIt);
      if (failed(translated)) {
        continue;
      }
      llvm::copy_if(
          *translated, std::back_inserter(translatedAliases),
          [](const SourceRef &translatedRef) { return !translatedRef.isConstant(); }
      );
    }
    if (translatedAliases.empty()) {
      continue;
    }
    const SourceRef &leader = translatedAliases.front();
    res.directAliases.insert(leader);
    for (const SourceRef &alias : llvm::drop_begin(translatedAliases)) {
      res.directAliases.unionSets(leader, alias);
    }
  }

  translateEdges(conditionalDependencyEdges, res.conditionalDependencyEdges);

  // Translate ref2Val as well
  for (const auto &[ref, vals] : ref2Val) {
    auto translationRes = translate(ref);
    if (succeeded(translationRes)) {
      for (const auto &translatedRef : *translationRes) {
        res.ref2Val[translatedRef].insert(vals.begin(), vals.end());
      }
    }
  }

  return res;
}

SourceRefSet ConstraintDependencyGraph::getConstrainingValues(const SourceRef &ref) const {
  SourceRefSet res;
  SourceRefSet visited;
  llvm::SmallVector<SourceRef> worklist = {ref};
  bool hasBooleanZero = false;
  bool hasBooleanOne = false;
  for (const auto &[source, constants] : constantSets) {
    if (!source.overlaps(ref)) {
      continue;
    }
    for (const SourceRef &constant : constants) {
      hasBooleanZero |= isBooleanLiteral(constant, 0);
      hasBooleanOne |= isBooleanLiteral(constant, 1);
    }
  }
  const bool followConditionalDependencies = hasBooleanZero && hasBooleanOne;
  const bool includeDescendants = llvm::isa<ArrayType, StructType, PodType>(ref.getType());

  // Overlapping aggregate ranges can bridge otherwise distinct equivalence sets. Follow those
  // bridges transitively so that, for example, `out[2] == child.out[2]` and a child constraint on
  // `child.out[0:N]` connect `out[2]` to the child's inputs.
  while (!worklist.empty()) {
    auto currRef = mlir::FailureOr<SourceRef>(worklist.pop_back_val());
    while (mlir::succeeded(currRef)) {
      if (!visited.insert(*currRef).second) {
        break;
      }

      for (const auto &[source, targets] : dependencyEdges) {
        if ((!source.overlaps(*currRef) &&
             !(includeDescendants && source.isValidPrefix(*currRef))) ||
            !isCompatibleIndexedAlternative(source, ref)) {
          continue;
        }
        if (auto constIt = constantSets.find(source); constIt != constantSets.end()) {
          res.insert(constIt->second.begin(), constIt->second.end());
        }
        for (const SourceRef &target : targets) {
          if (!isCompatibleIndexedAlternative(target, ref)) {
            continue;
          }
          SourceRef narrowed = target.narrowRanges(*currRef);
          if (!visited.contains(narrowed)) {
            worklist.push_back(narrowed);
          }
          if (!narrowed.overlaps(ref)) {
            res.insert(narrowed);
          }
          auto constIt = constantSets.find(target);
          if (constIt != constantSets.end()) {
            res.insert(constIt->second.begin(), constIt->second.end());
          }
        }
      }
      if (followConditionalDependencies) {
        for (const auto &[source, targets] : conditionalDependencyEdges) {
          if ((!source.overlaps(*currRef) &&
               !(includeDescendants && source.isValidPrefix(*currRef))) ||
              !isCompatibleIndexedAlternative(source, ref)) {
            continue;
          }
          for (const SourceRef &target : targets) {
            if (!isCompatibleIndexedAlternative(target, ref)) {
              continue;
            }
            SourceRef narrowed = target.narrowRanges(*currRef);
            if (!visited.contains(narrowed)) {
              worklist.push_back(narrowed);
            }
            if (!narrowed.overlaps(ref)) {
              res.insert(narrowed);
            }
          }
        }
      }
      for (const auto &[constantSource, constants] : constantSets) {
        if ((constantSource.overlaps(*currRef) ||
             (includeDescendants && constantSource.isValidPrefix(*currRef))) &&
            isCompatibleIndexedAlternative(constantSource, ref)) {
          res.insert(constants.begin(), constants.end());
        }
      }
      // Constraints on an aggregate also constrain its scalar descendants.
      auto parent = currRef->getParentPrefix();
      if (includeDescendants && succeeded(parent) && ref.isValidPrefix(*parent) && *parent != ref) {
        currRef = failure();
      } else {
        currRef = parent;
      }
    }
  }

  DenseMap<SourceRef, SourceRef> preferredAliases;
  auto root = ref.getRoot();
  StructDefOp queriedStruct =
      succeeded(root) ? root->getParentRegion()->getParentOfType<StructDefOp>() : nullptr;
  auto rankAlias = [queriedStruct](const SourceRef &candidate) {
    if (candidate.isBlockArgument() && *candidate.getInputNum() > 0) {
      auto candidateRoot = candidate.getRoot();
      auto candidateStruct = succeeded(candidateRoot)
                                 ? candidateRoot->getParentRegion()->getParentOfType<StructDefOp>()
                                 : nullptr;
      return candidateStruct == queriedStruct ? 0 : 1;
    }
    if (candidate.isCreateStructOp()) {
      return 2;
    }
    if (candidate.isBlockArgument()) {
      return 3;
    }
    return 4;
  };
  for (const auto *leaderIt : directAliases) {
    if (!leaderIt->isLeader()) {
      continue;
    }
    auto memberIt = directAliases.member_begin(*leaderIt);
    SourceRef preferred = *memberIt;
    for (; memberIt != directAliases.member_end(); ++memberIt) {
      if (rankAlias(*memberIt) < rankAlias(preferred) ||
          (rankAlias(*memberIt) == rankAlias(preferred) && *memberIt < preferred)) {
        preferred = *memberIt;
      }
    }
    for (memberIt = directAliases.member_begin(*leaderIt); memberIt != directAliases.member_end();
         ++memberIt) {
      preferredAliases.try_emplace(*memberIt, preferred);
    }
  }

  SourceRefSet normalized;
  if (ctx.runIntraproceduralAnalysis() &&
      llvm::any_of(res, [](const SourceRef &resultRef) { return resultRef.isNonDetOp(); })) {
    SourceRefSet equivalentInputs;
    for (const auto *candidate : signalSets) {
      const SourceRef &candidateRef = candidate->getData();
      if (candidateRef.isBlockArgument() && *candidateRef.getInputNum() > 0) {
        equivalentInputs.insert(candidateRef);
      }
    }
    res.insert(equivalentInputs.begin(), equivalentInputs.end());
  }

  for (const SourceRef &resultRef : res) {
    auto aliasIt = preferredAliases.find(resultRef);
    if (aliasIt == preferredAliases.end()) {
      normalized.insert(resultRef);
    } else if (!aliasIt->second.overlaps(ref)) {
      normalized.insert(aliasIt->second);
    } else if (!resultRef.overlaps(ref)) {
      normalized.insert(resultRef);
    }
  }

  const bool hasInputDependency = llvm::any_of(normalized, [](const SourceRef &resultRef) {
    return resultRef.isBlockArgument() && *resultRef.getInputNum() > 0;
  });
  if (hasInputDependency) {
    auto isLogicalComponentPath = [](const SourceRef &resultRef) {
      if (!resultRef.isBlockArgument() || *resultRef.getInputNum() != 0) {
        return false;
      }
      return llvm::any_of(resultRef.getPath(), [](const SourceRefIndex &index) {
        return index.isMember() && llvm::isa<StructType>(index.getMember().getType());
      });
    };
    const bool queryIsPodStorage = llvm::isa<PodType>(ref.getType());
    const bool hasLogicalComponentPath =
        !queryIsPodStorage && llvm::any_of(normalized, isLogicalComponentPath);
    SourceRefSet terminalDependencies;
    auto isTerminal = [&](const SourceRef &resultRef) {
      if (resultRef.isConstant() || (isLogicalComponentPath(resultRef) && !queryIsPodStorage)) {
        return true;
      }
      if (!resultRef.isBlockArgument() || *resultRef.getInputNum() == 0) {
        return false;
      }
      return !hasLogicalComponentPath || llvm::none_of(resultRef.getPath(), [](const auto &index) {
        return index.isIndexRange();
      });
    };
    for (const SourceRef &resultRef : normalized) {
      if (isTerminal(resultRef)) {
        terminalDependencies.insert(resultRef);
      }
    }
    normalized = std::move(terminalDependencies);
  }

  auto isStrictRangeSubset = [](const SourceRef &candidate, const SourceRef &other) {
    auto candidateRoot = candidate.getRoot();
    auto otherRoot = other.getRoot();
    if (failed(candidateRoot) || failed(otherRoot) || *candidateRoot != *otherRoot ||
        candidate.getPath().size() != other.getPath().size()) {
      return false;
    }

    bool isStrict = false;
    for (auto [candidateIndex, otherIndex] : llvm::zip(candidate.getPath(), other.getPath())) {
      if (candidateIndex == otherIndex) {
        continue;
      }
      if (!candidateIndex.isIndexRange() || !otherIndex.isIndexRange()) {
        return false;
      }
      auto [candidateLower, candidateUpper] = candidateIndex.getIndexRange();
      auto [otherLower, otherUpper] = otherIndex.getIndexRange();
      if (candidateLower < otherLower || candidateUpper > otherUpper) {
        return false;
      }
      isStrict = true;
    }
    return isStrict;
  };
  SourceRefSet minimalRanges;
  for (const SourceRef &candidate : normalized) {
    if (llvm::none_of(normalized, [&](const SourceRef &other) {
      return isStrictRangeSubset(candidate, other);
    })) {
      minimalRanges.insert(candidate);
    }
  }
  return minimalRanges;
}

/* ConstraintDependencyGraphModuleAnalysis */

LogicalResult ConstraintDependencyGraphModuleAnalysis::validateInput(ModuleOp module) const {
  if (failed(SourceRefAnalysis::validateInput(module))) {
    return failure();
  }
  auto result = module.walk([](Operation *op) -> WalkResult {
    if (op->getName().getDialectNamespace() == "cf") {
      return op->emitOpError(
          "constraint dependency graph analysis requires SCF control flow; "
          "operations from the 'cf' dialect are unsupported"
      );
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

/* ConstraintDependencyGraphStructAnalysis */

LogicalResult ConstraintDependencyGraphStructAnalysis::runAnalysis(
    DataFlowSolver &solver, AnalysisManager &moduleAnalysisManager, const CDGAnalysisContext &ctx
) {
  auto result = ConstraintDependencyGraph::compute(
      getModule(), getStruct(), solver, moduleAnalysisManager, ctx
  );
  if (failed(result)) {
    return failure();
  }
  setResult(ctx, std::move(*result));
  return success();
}

} // namespace llzk
