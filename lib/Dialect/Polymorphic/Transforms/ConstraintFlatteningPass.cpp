//===-- ConstraintFlatteningPass.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "ConstraintBindings.h"
#include "ConstraintDiagnostics.h"
#include "ConstraintScalarFacts.h"
#include "ConstraintStorage.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Polymorphic/Util/StaticLoop.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/TypeSwitch.h>

namespace llzk::polymorphic {
#define GEN_PASS_DEF_CONSTRAINTFLATTENINGPASS
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h.inc"
} // namespace llzk::polymorphic

using namespace mlir;
using namespace llzk;
using namespace llzk::component;
using namespace llzk::function;
using namespace llzk::polymorphic;
using namespace llzk::polymorphic::detail;

namespace {
/// Expand structural control flow while leaving ordinary expressions as SSA.
class ConstraintElaborator {
  ModuleOp module;
  StructDefOp main;
  OpBuilder builder;
  SymbolTableCollection tables;
  /// Detached output is committed only after expansion and validation succeed.
  Region output;
  /// Scalar facts requested by control flow and family coordinates only.
  ConstraintScalarFacts facts;
  std::optional<ConstraintStorage> storage;
  ConstraintBindings bindings;
  /// Live unresolved scalar storage reads are rejected after dead SSA removal.
  llvm::DenseSet<Operation *> nonSignalReads;
  SmallVector<FuncDefOp> activeCalls;

  static bool aggregate(Type type) { return isa<array::ArrayType, pod::PodType>(type); }

  /// Scalar-typed source operands must already have emitted SSA bindings.
  static Value scalar(Value source, const Environment &frame) {
    auto found = frame.find(source);
    assert(found != frame.end() && "missing source binding");
    const auto *value = std::get_if<Value>(&found->second);
    assert(value && *value && "expected a scalar SSA binding");
    return *value;
  }

  /// Materialize only immutable storage access chains ending in a scalar leaf.
  FailureOr<Value> readStorage(Value value, ArrayRef<Attribute> path, Type type, Operation *site) {
    while (!path.empty()) {
      if (auto arrayType = dyn_cast<array::ArrayType>(value.getType())) {
        unsigned rank = arrayType.getShape().size();
        if (path.size() < rank) {
          return constraintError(site, "storage path does not identify a scalar leaf");
        }
        SmallVector<Value> indices;
        for (Attribute step : path.take_front(rank)) {
          auto index = dyn_cast<IntegerAttr>(step);
          if (!index) {
            return constraintError(site, "invalid array storage path");
          }
          indices.push_back(
              arith::ConstantIndexOp::create(builder, site->getLoc(), index.getInt())
          );
        }
        value = array::ReadArrayOp::create(
            builder, site->getLoc(), arrayType.getElementType(), value, indices
        );
        path = path.drop_front(rank);
        continue;
      }
      auto podType = dyn_cast<pod::PodType>(value.getType());
      auto name = dyn_cast<StringAttr>(path.front());
      if (!podType || !name) {
        return constraintError(site, "invalid POD storage path");
      }
      auto recordType = podType.getRecord(name.getValue(), [site] {
        return site->emitError(CONSTRAINT_DIAGNOSTIC_PREFIX);
      });
      if (failed(recordType)) {
        return failure();
      }
      value = pod::ReadPodOp::create(builder, site->getLoc(), *recordType, value, name.getValue());
      path = path.drop_front();
    }
    if (value.getType() != type) {
      return constraintError(site, "storage leaf type mismatch");
    }
    auto info = storage->describe(value, site);
    if (failed(info)) {
      return failure();
    }
    if (!isa<StructType>(type) && !info->signal) {
      nonSignalReads.insert(value.getDefiningOp());
    }
    return value;
  }

  FailureOr<Value> load(AccessPath path, Operation *site) {
    return bindings.load(
        path, site, [this](Value root, ArrayRef<Attribute> pathSteps, Type type, Operation *at) {
      return readStorage(root, pathSteps, type, at);
    }
    );
  }

  /// Copy scalar bindings under a new root; never emit aggregate copies.
  FailureOr<BoundValue> copy(BoundValue source, Operation *site, Value input = {}) {
    auto *path = std::get_if<AccessPath>(&source);
    if (!path) {
      return source;
    }
    auto target = bindings.create(path->type, input);
    if (failed(transfer(target, *path, site))) {
      return failure();
    }
    return BoundValue(target);
  }

  LogicalResult transfer(AccessPath target, AccessPath source, Operation *site) {
    return bindings.transfer(
        target, source, site,
        [this](Value root, ArrayRef<Attribute> pathSteps, Type type, Operation *at) {
      return readStorage(root, pathSteps, type, at);
    }
    );
  }

  /// Assign a scalar binding or snapshot an aggregate subtree.
  LogicalResult assign(AccessPath target, BoundValue source, Operation *site) {
    if (auto *value = std::get_if<Value>(&source)) {
      bindings.store(target, *value);
      return success();
    }
    return transfer(target, std::get<AccessPath>(source), site);
  }

  /// Resolve and bounds-check every array coordinate before accessing bindings.
  FailureOr<AccessPath>
  arrayPath(Value source, ValueRange indices, Environment &frame, Operation *site) {
    auto path = std::get<AccessPath>(frame.lookup(source));
    for (Value index : indices) {
      auto coordinate = facts.integer(scalar(index, frame));
      if (!coordinate) {
        return constraintError(site, "array path requires compile-time indices");
      }
      auto selected = bindings.select(path, builder.getIndexAttr(*coordinate), site);
      if (failed(selected)) {
        return failure();
      }
      path = *selected;
    }
    return path;
  }

  /// Resolve a named POD record without reconstructing the POD value.
  FailureOr<AccessPath> podPath(Value source, StringRef name, Environment &frame, Operation *site) {
    auto path = std::get<AccessPath>(frame.lookup(source));
    return bindings.select(path, builder.getStringAttr(name), site);
  }

  /// Aggregate reads have value semantics: a new writable root gets the bindings.
  LogicalResult readPath(Value result, AccessPath path, Environment &frame, Operation *site) {
    if (aggregate(path.type)) {
      auto copied = copy(BoundValue(path), site);
      if (failed(copied)) {
        return failure();
      }
      frame[result] = *copied;
      return success();
    }
    auto value = load(path, site);
    if (failed(value)) {
      return failure();
    }
    frame[result] = *value;
    return success();
  }

  /// Writes change local destination bindings, not the referenced scalar values.
  LogicalResult writePath(AccessPath path, Value value, Environment &frame, Operation *site) {
    if (failed(bindings.writable(path, site))) {
      return failure();
    }
    return assign(path, frame.lookup(value), site);
  }

  /// Allocate only a root and initialize its row-major element bindings.
  LogicalResult arrayNew(array::CreateArrayOp op, Environment &frame) {
    auto path = bindings.create(op.getType());
    frame[op.getResult()] = path;
    if (op.getElements().empty()) {
      return success();
    }
    unsigned next = 0;
    return bindings.elements(path, op, [this, &op, &frame, &next](AccessPath item) {
      return assign(item, frame.lookup(op.getElements()[next++]), op);
    });
  }

  /// Initialize named records under a fresh local root.
  LogicalResult podNew(pod::NewPodOp op, Environment &frame) {
    auto path = bindings.create(op.getType());
    frame[op.getResult()] = path;
    for (auto record : op.getInitializedRecordValues()) {
      auto field = podPath(op.getResult(), record.name, frame, op);
      if (failed(field) || failed(assign(*field, frame.lookup(record.value), op))) {
        return failure();
      }
    }
    return success();
  }

  /// Preserve original component reads; aggregate results become immutable paths.
  LogicalResult memberRead(MemberReadOp op, Environment &frame) {
    if (op.getTableOffset()) {
      return constraintError(op, "member read requires concrete non-column component storage");
    }
    IRMapping mapping;
    mapping.map(op.getComponent(), scalar(op.getComponent(), frame));
    auto emitted = llvm::cast<MemberReadOp>(builder.clone(*op, mapping));
    auto info = storage->describe(emitted.getResult(), op);
    if (failed(info)) {
      return failure();
    }
    if (aggregate(op.getType())) {
      frame[op.getResult()] = bindings.create(info->memberType, {}, emitted.getResult());
    } else {
      frame[op.getResult()] = emitted.getResult();
      if (!isa<StructType>(op.getType()) && !info->signal) {
        nonSignalReads.insert(emitted);
      }
    }
    return success();
  }

  /// Constant aggregate globals initialize local bindings, not emitted storage.
  LogicalResult globalRead(global::GlobalReadOp op, Environment &frame) {
    auto definition = lookupTopLevelSymbol<global::GlobalDefOp>(tables, op.getNameRefAttr(), op);
    if (failed(definition) || !definition->get().isConstant() ||
        !definition->get().getInitialValue()) {
      return constraintError(op, "global read requires an initialized constant");
    }
    Attribute initial = definition->get().getInitialValue();
    if (!aggregate(op.getType())) {
      auto value = constant(initial, op);
      if (failed(value)) {
        return failure();
      }
      frame[op.getResult()] = *value;
      return success();
    }
    auto values = dyn_cast<ArrayAttr>(initial);
    if (!values || !isa<array::ArrayType>(op.getType())) {
      return constraintError(op, "unsupported aggregate global initializer");
    }
    auto path = bindings.create(op.getType());
    frame[op.getResult()] = path;
    unsigned next = 0;
    return bindings.walk(path, op, [this, &op, values, &next](AccessPath item) -> LogicalResult {
      if (next >= values.size()) {
        return constraintError(op, "global initializer does not match aggregate leaves");
      }
      auto value = constant(values[next++], op);
      if (failed(value)) {
        return failure();
      }
      bindings.store(item, *value);
      return success();
    });
  }

  /// Materialize a typed scalar initializer as an SSA constant.
  FailureOr<Value> constant(Attribute attribute, Operation *site) {
    if (auto felt = dyn_cast<felt::FeltConstAttr>(attribute)) {
      return Value(felt::FeltConstantOp::create(builder, site->getLoc(), felt));
    }
    if (auto integer = dyn_cast<IntegerAttr>(attribute)) {
      return Value(arith::ConstantOp::create(builder, site->getLoc(), integer));
    }
    return constraintError(site, "unsupported scalar global initializer");
  }

  /// Remap scalar operands after the operation's handler has checked support.
  FailureOr<IRMapping> scalarMapping(Operation &op, const Environment &frame) {
    IRMapping mapping;
    for (Value operand : op.getOperands()) {
      if (aggregate(operand.getType())) {
        return constraintError(&op, "unsupported aggregate operand");
      }
      mapping.map(operand, scalar(operand, frame));
    }
    return mapping;
  }

  /// Only these scalar field expressions can remain in flattened constraints.
  /// Purity or dialect membership alone does not establish field semantics.
  static bool isFieldExpression(Operation &op) {
    return isa<
        felt::FeltConstantOp, felt::AddFeltOp, felt::SubFeltOp, felt::MulFeltOp, felt::NegFeltOp>(
        op
    );
  }

  /// Recognize integer computations by shape, independently of their dialect.
  /// These may be evaluated temporarily, but are never approved output operations.
  static bool isIntegerExpression(Operation &op) {
    return !op.getNumRegions() && op.getNumResults() && isMemoryEffectFree(&op) &&
           llvm::all_of(op.getResultTypes(), [](Type type) {
      return isa<IndexType, IntegerType>(type);
    });
  }

  /// Evaluate pure integer results when possible; symbolic results must become dead.
  LogicalResult emitStructural(Operation &op, IRMapping &mapping, Environment &frame) {
    Operation *emitted = builder.clone(op, mapping);
    for (auto [source, result] : llvm::zip_equal(op.getResults(), emitted->getResults())) {
      if (auto constant = dyn_cast_or_null<IntegerAttr>(facts.known(result))) {
        frame[source] = Value(arith::ConstantOp::create(builder, op.getLoc(), constant));
      } else {
        frame[source] = result;
      }
    }
    // Keep the temporary operation until final dead-code removal: other folded
    // results may still depend on it, and the facts cache retains its SSA keys.
    return success();
  }

  /// Evaluate non-native field arithmetic under the given environment.
  /// Require every result to fold to a field constant.
  LogicalResult evaluateNonNative(Operation &op, Environment &frame) {
    if (op.getNumRegions() || !op.getNumResults() || !isMemoryEffectFree(&op) ||
        !llvm::all_of(op.getResultTypes(), [](Type type) { return isa<felt::FeltType>(type); })) {
      return constraintError(&op, "lower non-native operations before constraint flattening");
    }
    auto mapping = scalarMapping(op, frame);
    if (failed(mapping)) {
      return failure();
    }
    // A field left shift with a known amount is multiplication by 2^amount.
    // Fold that factor using the shift's own modular semantics, including large
    // unsigned shift representatives. Right shifts retain integer semantics.
    if (auto shift = dyn_cast<felt::ShlFeltOp>(op); shift &&
                                                    !facts.known(mapping->lookup(shift.getLhs())) &&
                                                    facts.known(mapping->lookup(shift.getRhs()))) {
      Value lhs = mapping->lookup(shift.getLhs());
      auto one = felt::FeltConstAttr::get(
          op.getContext(), llvm::DynamicAPInt(1), llvm::cast<felt::FeltType>(shift.getType())
      );
      mapping->map(shift.getLhs(), felt::FeltConstantOp::create(builder, op.getLoc(), one));
      Operation *factor = builder.clone(op, *mapping);
      auto constant = dyn_cast_or_null<felt::FeltConstAttr>(facts.known(factor->getResult(0)));
      if (!constant) {
        return constraintError(&op, "lower non-native operations before constraint flattening");
      }
      Value multiplier = felt::FeltConstantOp::create(builder, op.getLoc(), constant);
      frame[shift.getResult()] =
          Value(felt::MulFeltOp::create(builder, op.getLoc(), lhs, multiplier));
      return success();
    }
    Operation *emitted = builder.clone(op, *mapping);
    for (auto [source, result] : llvm::zip_equal(op.getResults(), emitted->getResults())) {
      auto constant = dyn_cast_or_null<felt::FeltConstAttr>(facts.known(result));
      if (!constant || constant.getType() != result.getType()) {
        return constraintError(&op, "lower non-native operations before constraint flattening");
      }
      frame[source] = Value(felt::FeltConstantOp::create(builder, op.getLoc(), constant));
    }
    // As with structural computations, keep cached SSA keys alive until DCE.
    return success();
  }

  /// Emit explicitly supported constraints, or evaluate temporary integer facts.
  LogicalResult emit(Operation &op, Environment &frame) {
    if (auto assertion = dyn_cast<boolean::AssertOp>(op)) {
      auto condition = facts.integer(scalar(assertion.getCondition(), frame));
      return condition && *condition ? success()
                                     : constraintError(&op, "assertion must be statically true");
    }
    bool equality = isa<constrain::EmitEqualityOp>(op);
    if (equality && !isa<felt::FeltType>(op.getOperand(0).getType())) {
      return constraintError(&op, "equality constraints require scalar felt operands");
    }
    bool structural = isIntegerExpression(op);
    if (!(equality || structural || isFieldExpression(op) || isa<cast::IntToFeltOp>(op))) {
      if (op.hasTrait<function::NotFieldNative>() || isa<felt::DivFeltOp>(op)) {
        return evaluateNonNative(op, frame);
      }
      if (isa<constrain::EmitContainmentOp>(op)) {
        return constraintError(
            &op, "unsupported operation 'constrain.in': containment constraints require lowering "
                 "before constraint flattening"
        );
      }
      if (llvm::any_of(op.getResultTypes(), [](Type type) { return isa<FloatType>(type); })) {
        return constraintError(
            &op, "unsupported: floating-point results are not supported in flattened constraints"
        );
      }
      return constraintError(
          &op, "unsupported operation '" + op.getName().getStringRef() +
                   "': expected a supported field expression or pure integer computation"
      );
    }
    auto mapping = scalarMapping(op, frame);
    if (failed(mapping)) {
      return failure();
    }
    if (structural) {
      return emitStructural(op, *mapping, frame);
    }
    Operation *emitted = builder.clone(op, *mapping);
    for (auto [source, result] : llvm::zip_equal(op.getResults(), emitted->getResults())) {
      frame[source] = result;
    }
    return success();
  }

  static void bind(ValueRange arguments, ArrayRef<BoundValue> values, Environment &frame) {
    for (auto [argument, value] : llvm::zip_equal(arguments, values)) {
      frame[argument] = value;
    }
  }

  /// Expand one defined function using fresh SSA bindings. Aggregate arguments
  /// retain identity, but their roots are protected from writes during the call.
  LogicalResult invoke(
      FuncDefOp callee, ArrayRef<BoundValue> arguments, Operation *site,
      SmallVectorImpl<BoundValue> &results
  ) {
    if (!callee || !llvm::hasSingleElement(callee.getBody()) || isInTemplate(callee)) {
      return constraintError(site, "callee must be concrete, defined, and single-block");
    }
    if (llvm::is_contained(activeCalls, callee)) {
      return constraintError(site, "recursive calls are unsupported");
    }
    if (callee.getNumArguments() != arguments.size()) {
      return constraintError(site, "callee argument arity mismatch");
    }
    if (activeCalls.size() >= 256) {
      return constraintError(site, "call nesting limit exceeded");
    }
    Environment frame;
    for (auto [parameter, argument] : llvm::zip_equal(callee.getArguments(), arguments)) {
      frame[parameter] = argument;
      if (const auto *path = std::get_if<AccessPath>(&argument)) {
        bindings.protect(*path);
      }
    }
    activeCalls.push_back(callee);
    auto status = expand(callee.getBody().front(), frame, results);
    activeCalls.pop_back();
    for (const auto &argument : arguments) {
      if (const auto *path = std::get_if<AccessPath>(&argument)) {
        bindings.unprotect(*path);
      }
    }
    return status;
  }

  /// Resolve constrain calls on the concrete component and expand their bodies.
  LogicalResult call(CallOp op, Environment &frame) {
    SmallVector<BoundValue> arguments;
    for (Value argument : op.getArgOperands()) {
      arguments.push_back(frame.lookup(argument));
    }
    FuncDefOp target;
    if (!arguments.empty() && op.calleeIsStructConstrain()) {
      auto info = storage->describe(std::get<Value>(arguments.front()), op);
      if (failed(info)) {
        return failure();
      }
      auto owner = info->definition;
      target = owner ? owner.getConstrainFuncOp() : FuncDefOp();
    } else {
      auto found = lookupTopLevelSymbol<FuncDefOp>(tables, op.getCalleeAttr(), op);
      if (failed(found)) {
        return failure();
      }
      target = found->get();
    }
    SmallVector<BoundValue> results;
    if (failed(invoke(target, arguments, op, results))) {
      return failure();
    }
    if (results.size() != op.getNumResults()) {
      return constraintError(op, "callee result arity mismatch");
    }
    bind(op.getResults(), results, frame);
    return success();
  }

  /// Select the known branch without generating conditional output.
  LogicalResult branch(scf::IfOp op, Environment &frame) {
    auto condition = facts.integer(scalar(op.getCondition(), frame));
    if (!condition) {
      return constraintError(
          op, "symbolic i1 condition requires an explicit field selector representation"
      );
    }
    auto &region = *condition ? op.getThenRegion() : op.getElseRegion();
    SmallVector<BoundValue> results;
    if (!region.empty() && failed(expand(region.front(), frame, results))) {
      return failure();
    }
    bind(op.getResults(), results, frame);
    return success();
  }

  /// Expand signed static iterations with independent SSA bindings.
  LogicalResult forLoop(scf::ForOp op, Environment &frame) {
    auto lower = facts.integer(scalar(op.getLowerBound(), frame));
    auto upper = facts.integer(scalar(op.getUpperBound(), frame));
    auto step = facts.integer(scalar(op.getStep(), frame));
    if (!lower || !upper || !step || *step <= 0 || op.getUnsignedCmp()) {
      return constraintError(op, "requires signed static for bounds and a positive step");
    }
    SmallVector<BoundValue> carried;
    for (Value value : op.getInitArgs()) {
      carried.push_back(frame.lookup(value));
    }
    for (int64_t i = *lower; i < *upper;) {
      frame[op.getInductionVar()] = Value(
          arith::ConstantOp::create(
              builder, op.getLoc(), builder.getIntegerAttr(op.getInductionVar().getType(), i)
          )
      );
      bind(op.getRegionIterArgs(), carried, frame);
      carried.clear();
      if (failed(expand(*op.getBody(), frame, carried))) {
        return failure();
      }
      if (!hasNextStaticIteration(i, *upper, *step)) {
        break;
      }
      i += *step;
    }
    bind(op.getResults(), carried, frame);
    return success();
  }

  /// Recheck the scalar condition and carry SSA yields between iterations.
  LogicalResult whileLoop(scf::WhileOp op, Environment &frame) {
    SmallVector<BoundValue> carried;
    for (Value value : op.getInits()) {
      carried.push_back(frame.lookup(value));
    }
    while (true) {
      bind(op.getBeforeArguments(), carried, frame);
      SmallVector<BoundValue> condition;
      if (failed(expand(op.getBefore().front(), frame, condition))) {
        return failure();
      }
      auto test = facts.integer(std::get<Value>(condition.front()));
      if (!test) {
        return constraintError(op, "while condition must be statically known");
      }
      carried.assign(condition.begin() + 1, condition.end());
      if (!*test) {
        break;
      }
      bind(op.getAfterArguments(), carried, frame);
      carried.clear();
      if (failed(expand(op.getAfter().front(), frame, carried))) {
        return failure();
      }
    }
    bind(op.getResults(), carried, frame);
    return success();
  }

  /// Scalar array reads and subarray snapshots share concrete path resolution.
  template <typename ReadOp> LogicalResult arrayRead(ReadOp op, Environment &frame) {
    auto path = arrayPath(op.getArrRef(), op.getIndices(), frame, op);
    if (failed(path)) {
      return failure();
    }
    return readPath(op.getResult(), *path, frame, op);
  }

  /// Scalar and subtree assignments share destination checks and binding updates.
  template <typename WriteOp> LogicalResult arrayWrite(WriteOp op, Environment &frame) {
    auto path = arrayPath(op.getArrRef(), op.getIndices(), frame, op);
    if (failed(path)) {
      return failure();
    }
    return writePath(*path, op.getRvalue(), frame, op);
  }

  /// Resolve a record using its scalar binding or value-copy snapshot.
  LogicalResult podRead(pod::ReadPodOp op, Environment &frame) {
    auto path = podPath(op.getPodRef(), op.getRecordName(), frame, op);
    if (failed(path)) {
      return failure();
    }
    return readPath(op.getResult(), *path, frame, op);
  }

  /// Replace a writable record binding or subtree.
  LogicalResult podWrite(pod::WritePodOp op, Environment &frame) {
    auto path = podPath(op.getPodRef(), op.getRecordName(), frame, op);
    if (failed(path)) {
      return failure();
    }
    return writePath(*path, op.getValue(), frame, op);
  }

  /// Obtain static dimensions directly from the bound subtree type.
  LogicalResult arrayLength(array::ArrayLengthOp op, Environment &frame) {
    auto path = std::get<AccessPath>(frame.lookup(op.getArrRef()));
    auto type = dyn_cast<array::ArrayType>(path.type);
    auto dimension = facts.integer(scalar(op.getDim(), frame));
    if (!type || !dimension || *dimension < 0 || uint64_t(*dimension) >= type.getShape().size()) {
      return constraintError(op, "array dimension must be statically known and in bounds");
    }
    int64_t size = type.getShape()[*dimension];
    if (size < 0) {
      return constraintError(op, "aggregate shape must be static");
    }
    frame[op.getResult()] = Value(arith::ConstantIndexOp::create(builder, op.getLoc(), size));
    return success();
  }

  /// Select a binding using a compile-time Boolean condition.
  LogicalResult select(arith::SelectOp op, Environment &frame) {
    auto condition = facts.integer(scalar(op.getCondition(), frame));
    if (!condition) {
      return constraintError(
          op, "symbolic select requires an explicit field selector representation"
      );
    }
    frame[op.getResult()] = frame.lookup(*condition ? op.getTrueValue() : op.getFalseValue());
    return success();
  }

  /// Each storage or control operation has one focused handler. Ordinary scalar
  /// expressions use the common emitter and never see aggregate operands.
  LogicalResult dispatch(Operation &op, Environment &frame) {
    return llvm::TypeSwitch<Operation *, LogicalResult>(&op)
        .Case<scf::IfOp>([this, &frame](auto control) { return branch(control, frame); })
        .Case<scf::ForOp>([this, &frame](auto control) { return forLoop(control, frame); })
        .Case<scf::WhileOp>([this, &frame](auto control) { return whileLoop(control, frame); })
        .Case<CallOp>([this, &frame](auto function) { return call(function, frame); })
        .Case<arith::SelectOp>([this, &frame](auto choice) { return select(choice, frame); })
        .Case<UnifiableCastOp>([&frame](auto conversion) {
      frame[conversion.getResult()] = frame.lookup(conversion.getInput());
      return success();
    })
        .Case<global::GlobalReadOp>([this, &frame](auto read) { return globalRead(read, frame); })
        .Case<MemberReadOp>([this, &frame](auto read) { return memberRead(read, frame); })
        .Case<MemberWriteOp>([](auto write) {
      return constraintError(write, "cannot write to original component storage");
    })
        .Case<array::CreateArrayOp>([this, &frame](auto allocation) {
      return arrayNew(allocation, frame);
    })
        .Case<NonDetOp>([this, &frame](auto allocation) {
      if (!aggregate(allocation.getType())) {
        return emit(*allocation, frame);
      }
      // Uninitialized aggregate leaves may be overwritten but never read.
      frame[allocation.getResult()] = bindings.create(allocation.getType());
      return success();
    })
        .Case<pod::NewPodOp>([this, &frame](auto allocation) { return podNew(allocation, frame); })
        .Case<array::ReadArrayOp, array::ExtractArrayOp>([this, &frame](auto read) {
      return arrayRead(read, frame);
    })
        .Case<array::WriteArrayOp, array::InsertArrayOp>([this, &frame](auto write) {
      return arrayWrite(write, frame);
    })
        .Case<pod::ReadPodOp>([this, &frame](auto read) { return podRead(read, frame); })
        .Case<pod::WritePodOp>([this, &frame](auto write) { return podWrite(write, frame); })
        .Case<array::ArrayLengthOp>([this, &frame](auto length) {
      return arrayLength(length, frame);
    }).Default([this, &frame](Operation *ordinary) { return emit(*ordinary, frame); });
  }

  /// Visit source operations once per execution; all ordinary values remain SSA.
  LogicalResult expand(Block &block, Environment &frame, SmallVectorImpl<BoundValue> &results) {
    for (Operation &op : block) {
      if (isa<ReturnOp, scf::YieldOp, scf::ConditionOp>(op)) {
        for (Value value : op.getOperands()) {
          results.push_back(frame.lookup(value));
        }
        return success();
      }
      if (failed(dispatch(op, frame))) {
        return failure();
      }
    }
    return constraintError(block.getParentOp(), "missing supported terminator");
  }

  /// Check every surviving operation against the flattened-output contract.
  /// This check does not depend on how an operation reached the output region.
  LogicalResult validateOperation(Operation &op) {
    if (op.getNumRegions()) {
      return constraintError(&op, "regions are unsupported in flattened constraints");
    }
    if (isFieldExpression(op) || isa<constrain::EmitEqualityOp>(op)) {
      auto fieldType = [](Type type) { return isa<felt::FeltType>(type); };
      if (llvm::all_of(op.getOperandTypes(), fieldType) &&
          llvm::all_of(op.getResultTypes(), fieldType)) {
        return success();
      }
      return constraintError(
          &op, "constraint expressions require scalar felt operands and results"
      );
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      Type type = constant.getType();
      if (type.isIndex() || type.isInteger(1)) {
        return success();
      }
    }
    if (auto conversion = dyn_cast<cast::IntToFeltOp>(op)) {
      Attribute literal;
      if (matchPattern(conversion.getValue(), m_Constant(&literal)) && isa<IntegerAttr>(literal)) {
        return success();
      }
      return constraintError(&op, "field conversion requires a compile-time integer");
    }
    if (isa<array::ReadArrayOp, array::ExtractArrayOp>(op)) {
      ValueRange indices = llvm::cast<array::ArrayAccessOpInterface>(op).getIndices();
      for (Value index : indices) {
        if (!index.getDefiningOp<arith::ConstantOp>()) {
          return constraintError(&op, "array path requires compile-time indices");
        }
      }
      return success();
    }
    if (auto read = dyn_cast<MemberReadOp>(op)) {
      if (read.getTableOffset()) {
        return constraintError(&op, "member read requires concrete non-column component storage");
      }
      return success();
    }
    if (isa<pod::ReadPodOp>(op)) {
      return success();
    }
    if (isIntegerExpression(op)) {
      return constraintError(&op, "structural operation requires a compile-time value");
    }
    return constraintError(
        &op, "unsupported operation in flattened constraints: '" + op.getName().getStringRef() + "'"
    );
  }

  /// Remove dead facts, enforce the output contract, and reject non-signal reads.
  LogicalResult validateOutput() {
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(output.front()))) {
      if (op.use_empty() &&
          (isMemoryEffectFree(&op) ||
           isa<MemberReadOp, array::ReadArrayOp, array::ExtractArrayOp, pod::ReadPodOp>(op))) {
        op.erase();
      }
    }
    for (Operation &op : output.front()) {
      if (failed(validateOperation(op))) {
        return failure();
      }
      if (nonSignalReads.contains(&op)) {
        return constraintError(
            &op, "live unresolved non-signal scalar storage; algebraic elimination is required"
        );
      }
    }
    return success();
  }

  /// Verify in the original function scope and restore its body on failure.
  LogicalResult commit(FuncDefOp source) {
    Region original;
    original.takeBody(source.getBody());
    source.getBody().takeBody(output);
    if (failed(verify(source))) {
      output.takeBody(source.getBody());
      source.getBody().takeBody(original);
      return failure();
    }
    return success();
  }

public:
  explicit ConstraintElaborator(ModuleOp root)
      : module(root), builder(root.getContext()), facts(root.getContext()),
        bindings(root.getContext()) {}

  /// Preserve main's signature and commit only the complete validated body.
  LogicalResult run() {
    auto root = getRootStructDef(tables, module);
    if (failed(root)) {
      return failure();
    }
    if (!*root) {
      return constraintError(module, "requires llzk.main");
    }
    main = root->get();
    storage.emplace(module, main, builder, tables, facts);
    auto source = main.getConstrainFuncOp();
    if (isInTemplate(main) || !source || !llvm::hasSingleElement(source.getBody())) {
      return constraintError(main, "main must have a concrete single-block constrain function");
    }
    output.push_back(new Block());
    SmallVector<BoundValue> arguments;
    for (auto [i, argument] : llvm::enumerate(source.getArguments())) {
      Value emitted = output.front().addArgument(argument.getType(), argument.getLoc());
      if (aggregate(argument.getType())) {
        arguments.push_back(bindings.create(argument.getType(), argument, emitted));
      } else {
        arguments.push_back(emitted);
      }
    }
    builder.setInsertionPointToEnd(&output.front());
    SmallVector<BoundValue> results;
    if (failed(invoke(source, arguments, source, results)) || failed(validateOutput())) {
      return failure();
    }
    ReturnOp::create(builder, source.getLoc());
    return commit(source);
  }
};

class PassImpl : public llzk::polymorphic::impl::ConstraintFlatteningPassBase<PassImpl> {
  using Base = llzk::polymorphic::impl::ConstraintFlatteningPassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    if (failed(ConstraintElaborator(getOperation()).run())) {
      signalPassFailure();
    }
  }
};
} // namespace
