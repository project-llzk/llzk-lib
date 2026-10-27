//===-- R1CSLoweringPass.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-r1cs-lowering` pass.
///
//===----------------------------------------------------------------------===//

#include "r1cs/Dialect/IR/Attrs.h"
#include "r1cs/Dialect/IR/Constants.h"
#include "r1cs/Dialect/IR/Ops.h"
#include "r1cs/Dialect/IR/Types.h"
#include "r1cs/Transforms/TransformationPasses.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Transforms/LoweringUtils.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/LLZKLayout.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/Walk.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/FileUtilities.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ToolOutputFile.h>

#include <cstdint>
#include <deque>
#include <limits>
#include <memory>

// Include the generated base pass class definitions.
namespace r1cs {
#define GEN_PASS_DEF_R1CSLOWERINGPASS
#include "r1cs/Transforms/TransformationPasses.h.inc"
} // namespace r1cs

using namespace mlir;
using r1cs::WireKind;
using namespace llzk;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::component;
using namespace llzk::constrain;

#define DEBUG_TYPE "llzk-r1cs-lowering"
#define R1CS_AUXILIARY_MEMBER_PREFIX "__llzk_r1cs_lowering_pass_aux_member_"

namespace {

/// A LinearCombination is a map from a Value (like a variable or MemberRead) to a felt constant.
struct LinearCombination {
  DenseMap<Value, DynamicAPInt> terms; // variable -> coeff
  DynamicAPInt constant;

  LinearCombination() : constant() {}

  /// Add a term, preserving native coefficients for arithmetic overload selection.
  template <typename CoeffT> void addTerm(Value v, const CoeffT &coeff) {
    if (coeff == 0) {
      return;
    }
    auto [it, inserted] = terms.try_emplace(v, coeff);
    if (!inserted) {
      it->second += coeff;
    }
  }

  void negate() {
    for (auto &kv : terms) {
      kv.second = -kv.second;
    }
    constant = -constant;
  };

  /// Scale all coefficients, preserving native factors for arithmetic overload selection.
  template <typename FactorT> LinearCombination scaled(const FactorT &factor) const {
    LinearCombination result;
    if (factor == 0) {
      return result;
    }

    for (const auto &kv : terms) {
      result.terms[kv.first] = kv.second * factor;
    }
    result.constant = constant * factor;
    return result;
  }

  LinearCombination add(const LinearCombination &other) const {
    LinearCombination result(*this);

    for (const auto &kv : other.terms) {
      auto [it, inserted] = result.terms.try_emplace(kv.first, kv.second);
      if (!inserted) {
        it->second += kv.second;
      }
    }
    result.constant += other.constant;
    return result;
  }

  LinearCombination negated() const { return scaled(-1); }

  void print(raw_ostream &os) const {
    bool first = true;
    for (const auto &[val, coeff] : terms) {
      if (!first) {
        os << " + ";
      }
      first = false;
      os << coeff << '*' << val;
    }
    if (constant != 0) {
      if (!first) {
        os << " + ";
      }
      os << constant;
    }
    if (first && constant == 0) {
      os << '0';
    }
  }
};

/// A struct representing a * b = c R1CS constraint
struct R1CSConstraint {
  LinearCombination a;
  LinearCombination b;
  LinearCombination c;

  R1CSConstraint negated() const {
    R1CSConstraint result(*this);
    result.a = a.negated();
    result.c = c.negated();
    return result;
  }

  R1CSConstraint scaled(const DynamicAPInt &factor) const {
    R1CSConstraint result(*this);
    result.a = a.scaled(factor);
    result.c = c.scaled(factor);
    return result;
  }

  R1CSConstraint(const DynamicAPInt &constant) { c.constant = constant; }

  R1CSConstraint() = default;

  inline bool isLinearOnly() const { return a.terms.empty() && b.terms.empty(); }

  R1CSConstraint multiply(const R1CSConstraint &other) {
    auto isDegZero = [](const R1CSConstraint &constraint) {
      return constraint.a.terms.empty() && constraint.b.terms.empty() && constraint.c.terms.empty();
    };

    if (isDegZero(other)) {
      return this->scaled(other.c.constant);
    }
    if (isDegZero(*this)) {
      return other.scaled(this->c.constant);
    }

    if (isLinearOnly() && other.isLinearOnly()) {
      R1CSConstraint result;
      result.a = this->c;
      result.b = other.c;

      // We do NOT compute `c = a * b` because R1CS doesn't need it explicitly
      // It suffices to enforce: a * b = c

      return result;
    }
    llvm::errs() << "R1CSConstraint::multiply: Only supported for purely linear constraints.\n";
    llvm_unreachable("Invalid multiply: non-linear constraint(s) involved");
  }

  R1CSConstraint add(const R1CSConstraint &other) {

    if (isLinearOnly()) {
      R1CSConstraint result(other);
      result.c = result.c.add(this->c);
      return result;
    }
    if (other.isLinearOnly()) {
      R1CSConstraint result(*this);
      result.c = result.c.add(other.c);
      return result;
    }
    llvm::errs() << "R1CSConstraint::add: Only supported for purely linear constraints.\n";
    llvm_unreachable("Invalid add: non-linear constraint(s) involved");
  }

  void print(raw_ostream &os) const {
    os << '(';
    a.print(os);
    os << ") * (";
    b.print(os);
    os << ") = ";
    c.print(os);
  }
};

/// One association between a backend-independent layout leaf and a final R1CS wire.
struct R1CSWire {
  uint64_t layoutId;
  uint32_t wireId;
  WireKind kind;

  /// Inputs are represented by circuit block arguments rather than signal definitions.
  bool isInput() const { return kind == WireKind::PublicInput || kind == WireKind::PrivateInput; }
};

/// R1CS-ID-ordered wires and their reverse lookup by LLZK layout signal ID.
struct R1CSWireMap {
  SmallVector<R1CSWire> entries;
  SmallVector<uint32_t> wireIdsByLayout;
};

/// Convert main's flattened constraints using one wire for every layout leaf.
class PassImpl : public r1cs::impl::R1CSLoweringPassBase<PassImpl> {
  using Base = R1CSLoweringPassBase<PassImpl>;
  using Base::Base;

  StructDefOp main;
  FuncDefOp constrain;
  DenseMap<Value, unsigned> degrees;
  DenseMap<Value, ArrayAttr> paths;
  DenseMap<Value, Value> normalized;
  DenseMap<Value, Value> auxiliaries;
  DenseMap<Value, R1CSConstraint> expressions;
  DenseMap<Value, Value> linearWires;
  SmallVector<AuxAssignment> assignments;
  SmallVector<Value> wires;
  unsigned auxCounter = 0;

  /// Clear all per-module handles and caches before reusing a pass instance.
  void resetState() {
    main = StructDefOp();
    constrain = FuncDefOp();
    degrees.clear();
    paths.clear();
    normalized.clear();
    auxiliaries.clear();
    expressions.clear();
    linearWires.clear();
    assignments.clear();
    wires.clear();
    auxCounter = 0;
  }

  /// Storage reads are polynomial leaves, including reads through aggregates.
  static bool isStorageRead(Operation *op) {
    return isa_and_nonnull<MemberReadOp, array::ReadArrayOp, pod::ReadPodOp>(op);
  }

  /// Visit expression dependencies once, stopping at felt storage leaves.
  static SmallVector<Value> getExpressionOrder(Value root) {
    SmallVector<Value> order;
    SmallVector<std::pair<Value, bool>> pending {{root, false}};
    DenseSet<Value> visited;
    while (!pending.empty()) {
      auto [value, expanded] = pending.pop_back_val();
      if (expanded) {
        order.push_back(value);
        continue;
      }
      if (!visited.insert(value).second) {
        continue;
      }
      pending.emplace_back(value, true);
      Operation *op = value.getDefiningOp();
      if (op && !isStorageRead(op)) {
        for (Value operand : llvm::reverse(op->getOperands())) {
          pending.emplace_back(operand, false);
        }
      }
    }
    return order;
  }

  /// Recover a structural layout path from immutable flattened storage reads.
  FailureOr<ArrayAttr> resolveSignalPath(Value value) {
    if (auto found = paths.find(value); found != paths.end()) {
      return found->second;
    }
    OpBuilder builder(constrain);
    SmallVector<Attribute> path;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner() != &constrain.getBody().front()) {
        return constrain.emitOpError("R1CS storage must be rooted in constrain arguments");
      }
      if (arg.getArgNumber() == 0) {
        path.push_back(builder.getStringAttr(LAYOUT_MAIN_ROOT));
      } else {
        path.append(
            {builder.getStringAttr(LAYOUT_ARGUMENT_ROOT),
             builder.getI64IntegerAttr(arg.getArgNumber())}
        );
      }
    } else {
      Operation *op = value.getDefiningOp();
      Value parent;
      SmallVector<Attribute> suffix;
      if (auto read = dyn_cast<MemberReadOp>(op)) {
        if (read.getTableOffset() || !read.getMapOperands().empty()) {
          return read.emitError(
              "R1CS lowering does not support member reads with table offsets or map operands"
          );
        }
        parent = read.getComponent();
        suffix.push_back(builder.getStringAttr(read.getMemberName()));
      } else if (auto podRead = dyn_cast<pod::ReadPodOp>(op)) {
        parent = podRead.getPodRef();
        suffix.push_back(podRead.getRecordNameAttr());
      } else if (isa<array::ReadArrayOp, array::ExtractArrayOp>(op)) {
        auto access = llvm::cast<array::ArrayAccessOpInterface>(op);
        parent = access.getArrRef();
        for (Value index : access.getIndices()) {
          APInt literal;
          if (!matchPattern(index, m_ConstantInt(&literal)) || !literal.isSignedIntN(64) ||
              literal.isNegative()) {
            return op->emitError("R1CS storage paths require nonnegative constant indices");
          }
          suffix.push_back(builder.getIndexAttr(literal.getSExtValue()));
        }
      } else {
        return emitError(
            value.getLoc(), "R1CS signal read is not rooted in main storage or an input argument"
        );
      }
      auto prefix = resolveSignalPath(parent);
      if (failed(prefix)) {
        return failure();
      }
      path.append(prefix->begin(), prefix->end());
      path.append(suffix);
    }
    return paths[value] = builder.getArrayAttr(path);
  }

  /// Drop dead pure operations, then reject unsupported constraints and degrees above two.
  LogicalResult validateDegreeAtMostTwo() {
    if (failed(checkFuncBodyIsStraightLine(constrain, "R1CS lowering"))) {
      return failure();
    }
    // Remove unused infallible casts and their dead producers from the end of
    // the straight-line block. Effectful assertions are deliberately retained.
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(constrain.getBody().front()))) {
      if (isOpTriviallyDead(&op)) {
        op.erase();
      }
    }
    for (Operation &op : constrain.getBody().front()) {
      // Keep main's assertion diagnostics without changing the layout converter's
      // diagnostic for live, infallible operations it cannot lower.
      if (isa<cast::IntToFeltOp, cast::FeltToIndexOp>(op) && !isMemoryEffectFree(&op)) {
        return op.emitError() << "R1CS lowering does not support effectful '"
                              << op.getName().getStringRef() << "' operations";
      }
      if (!isa<
              EmitEqualityOp, ReturnOp, FeltConstantOp, AddFeltOp, SubFeltOp, MulFeltOp, NegFeltOp,
              MemberReadOp, array::ReadArrayOp, array::ExtractArrayOp, pod::ReadPodOp,
              arith::ConstantOp>(op)) {
        return op.emitError("unsupported operation in flattened R1CS constraints");
      }
    }
    for (auto eq : constrain.getOps<EmitEqualityOp>()) {
      for (Value root : eq->getOperands()) {
        for (Value value : getExpressionOrder(root)) {
          if (degrees.contains(value)) {
            continue;
          }
          Operation *op = value.getDefiningOp();
          unsigned degree = 0;
          if (!isa<FeltType>(value.getType())) {
            return emitError(value.getLoc(), "R1CS constraint expressions must have felt type");
          }
          if (isa<BlockArgument>(value) || isStorageRead(op)) {
            if (failed(resolveSignalPath(value))) {
              return failure();
            }
            degree = 1;
          } else if (isa<FeltConstantOp>(op)) {
            degree = 0;
          } else if (isa<AddFeltOp, SubFeltOp>(op)) {
            degree = std::max(degrees.lookup(op->getOperand(0)), degrees.lookup(op->getOperand(1)));
          } else if (isa<MulFeltOp>(op)) {
            degree = degrees.lookup(op->getOperand(0)) + degrees.lookup(op->getOperand(1));
          } else if (isa<NegFeltOp>(op)) {
            degree = degrees.lookup(op->getOperand(0));
          } else {
            return emitError(value.getLoc(), "unsupported operation in R1CS polynomial expression");
          }
          if (degree > 2) {
            return emitError(value.getLoc())
                   << "R1CS lowering requires polynomial degree at most two; found degree "
                   << degree << "; run degree lowering first";
          }
          degrees[value] = degree;
        }
      }
    }
    return success();
  }

  /// Materialize a quadratic expression as a signal with a defining equality.
  Value createAuxiliarySignal(Value expression, OpBuilder &builder) {
    if (auto found = auxiliaries.find(expression); found != auxiliaries.end()) {
      return found->second;
    }
    std::string name = R1CS_AUXILIARY_MEMBER_PREFIX + std::to_string(auxCounter++);
    auto member = addAuxMember(main, name, expression.getType());
    member.setSignal(true);
    Value read = MemberReadOp::create(
        builder, expression.getLoc(), expression.getType(), constrain.getSelfValueFromConstrain(),
        member.getNameAttr()
    );
    EmitEqualityOp::create(builder, expression.getLoc(), read, expression);
    assignments.push_back({name, expression});
    degrees[read] = 1;
    return auxiliaries[expression] = read;
  }

  /// Normalize sums of quadratic terms without lowering higher-degree input.
  Value normalizeExpression(Value root, OpBuilder &builder) {
    if (auto found = normalized.find(root); found != normalized.end()) {
      return found->second;
    }
    for (Value value : getExpressionOrder(root)) {
      if (normalized.contains(value)) {
        continue;
      }
      Operation *op = value.getDefiningOp();
      if (!op || isStorageRead(op) || isa<FeltConstantOp>(op)) {
        normalized[value] = value;
        continue;
      }
      SmallVector<Value> operands;
      for (Value operand : op->getOperands()) {
        operands.push_back(normalized.lookup(operand));
      }
      if (isa<AddFeltOp, SubFeltOp>(op) && degrees.lookup(operands[0]) == 2 &&
          degrees.lookup(operands[1]) == 2) {
        operands[0] = createAuxiliarySignal(operands[0], builder);
      }
      IRMapping mapping;
      for (auto [before, after] : llvm::zip(op->getOperands(), operands)) {
        mapping.map(before, after);
      }
      Value replacement = value;
      if (!llvm::equal(op->getOperands(), operands)) {
        replacement = builder.clone(*op, mapping)->getResult(0);
      }
      unsigned degree = degrees.lookup(operands[0]);
      if (operands.size() == 2) {
        degree = isa<MulFeltOp>(op) ? degree + degrees.lookup(operands[1])
                                    : std::max(degree, degrees.lookup(operands[1]));
      }
      degrees[replacement] = degree;
      normalized[value] = replacement;
    }
    return normalized.lookup(root);
  }

  /// Clone only the validated dependency graph needed for a compute assignment.
  FailureOr<Value> rebuildAuxiliaryExpression(Value root, OpBuilder &builder, IRMapping &mapping) {
    SmallVector<std::pair<Value, bool>> pending {{root, false}};
    while (!pending.empty()) {
      auto [value, expanded] = pending.pop_back_val();
      if (mapping.contains(value)) {
        continue;
      }
      Operation *op = value.getDefiningOp();
      if (!op) {
        return emitError(value.getLoc(), "cannot map R1CS auxiliary expression into compute");
      }
      if (expanded) {
        Operation *clone = builder.clone(*op, mapping);
        for (auto [before, after] : llvm::zip(op->getResults(), clone->getResults())) {
          mapping.map(before, after);
        }
      } else {
        pending.emplace_back(value, true);
        for (Value operand : llvm::reverse(op->getOperands())) {
          pending.emplace_back(operand, false);
        }
      }
    }
    return mapping.lookup(root);
  }

  /// Add auxiliary signal members, equations, and compute writes before layout generation.
  LogicalResult prepareAuxiliarySignals() {
    if (failed(checkForAuxMemberConflicts(main, R1CS_AUXILIARY_MEMBER_PREFIX))) {
      return failure();
    }
    SmallVector<EmitEqualityOp> equalities(constrain.getOps<EmitEqualityOp>());
    for (auto eq : equalities) {
      OpBuilder builder(eq);
      Value lhs = normalizeExpression(eq.getLhs(), builder);
      Value rhs = normalizeExpression(eq.getRhs(), builder);
      if (degrees.lookup(lhs) == 2 && degrees.lookup(rhs) == 2) {
        lhs = createAuxiliarySignal(lhs, builder);
      }
      eq->setOperands({lhs, rhs});
    }
    if (assignments.empty()) {
      return success();
    }
    auto compute = main.getComputeFuncOp();
    if (!compute || compute.isExternal()) {
      return main.emitOpError("R1CS auxiliary signals require a compute body");
    }
    if (compute.getNumArguments() + 1 != constrain.getNumArguments()) {
      return compute.emitOpError(
          "compute inputs do not match constrain inputs for R1CS auxiliaries"
      );
    }
    for (auto [input, argument] :
         llvm::zip(compute.getArguments(), llvm::drop_begin(constrain.getArguments()))) {
      if (input.getType() != argument.getType()) {
        return compute.emitOpError(
            "compute input types do not match constrain inputs for R1CS auxiliaries"
        );
      }
    }
    for (Block &block : compute.getBody()) {
      auto ret = dyn_cast<ReturnOp>(block.getTerminator());
      if (!ret) {
        continue;
      }
      if (ret->getNumOperands() != 1) {
        return ret.emitOpError("R1CS auxiliary writes require the returned component instance");
      }
      Value self = ret->getOperand(0);
      OpBuilder builder(ret);
      IRMapping mapping;
      mapping.map(constrain.getArgument(0), self);
      for (auto [input, argument] :
           llvm::zip(compute.getArguments(), llvm::drop_begin(constrain.getArguments()))) {
        mapping.map(argument, input);
      }
      for (const auto &assignment : assignments) {
        auto expression = rebuildAuxiliaryExpression(assignment.computedValue, builder, mapping);
        if (failed(expression)) {
          return failure();
        }
        MemberWriteOp::create(
            builder, ret.getLoc(), self, builder.getStringAttr(assignment.auxMemberName),
            *expression
        );
      }
    }
    return success();
  }

  /// Assign IDs after all auxiliary signals have been added to main.
  FailureOr<LLZKLayout> generateSignalLayout(ModuleOp module) {
    auto layout = buildLLZKLayout(module);
    if (failed(layout)) {
      return failure();
    }
    for (auto eq : constrain.getOps<EmitEqualityOp>()) {
      for (Value root : eq->getOperands()) {
        for (Value value : getExpressionOrder(root)) {
          if (!isa<BlockArgument>(value) && !isStorageRead(value.getDefiningOp())) {
            continue;
          }
          auto path = resolveSignalPath(value);
          if (failed(path)) {
            return failure();
          }
          if (!layout->getSignalId(*path)) {
            return emitError(value.getLoc())
                   << "R1CS storage read has no signal in the layout: " << *path;
          }
        }
      }
    }
    return layout;
  }

  /// Assign final R1CS IDs without changing the backend-independent LLZK layout.
  FailureOr<R1CSWireMap> buildR1CSWireMap(const LLZKLayout &layout) {
    if (layout.signalPaths.size() >= std::numeric_limits<uint32_t>::max()) {
      return main.emitOpError("R1CS layout exceeds the supported wire ID range");
    }
    DenseMap<WireKind, SmallVector<uint64_t>> layoutIdsByKind;
    for (auto [layoutId, path] : llvm::enumerate(layout.signalPaths)) {
      WireKind kind;
      if (llvm::cast<StringAttr>(path[0]).getValue() == LAYOUT_ARGUMENT_ROOT) {
        unsigned argument = llvm::cast<IntegerAttr>(path[1]).getInt();
        kind =
            constrain.hasArgPublicAttr(argument) ? WireKind::PublicInput : WireKind::PrivateInput;
      } else {
        auto member = main.getMemberDef(llvm::cast<StringAttr>(path[1]));
        kind = member.hasPublicAttr() ? WireKind::PublicOutput : WireKind::Internal;
      }
      layoutIdsByKind[kind].push_back(layoutId);
    }
    R1CSWireMap mapping;
    mapping.wireIdsByLayout.resize(layout.signalPaths.size());
    for (WireKind kind : r1cs::WIRE_KIND_ORDER) {
      for (uint64_t layoutId : layoutIdsByKind[kind]) {
        uint32_t wireId = mapping.entries.size() + r1cs::FIRST_SIGNAL_WIRE_ID;
        mapping.entries.push_back({layoutId, wireId, kind});
        mapping.wireIdsByLayout[layoutId] = wireId;
      }
    }
    return mapping;
  }

  /// Write each signal path and its LLZK and R1CS IDs directly from the lowering maps.
  void
  emitLayoutCSV(const LLZKLayout &layout, const R1CSWireMap &mapping, llvm::raw_ostream &output) {
    output << "access_path,llzk_layout_id,r1cs_wire_id\n";
    for (auto [layoutId, path] : llvm::enumerate(layout.signalPaths)) {
      std::string accessPath;
      llvm::raw_string_ostream stream(accessPath);
      printLLZKLayoutPath(path, layout.argumentNames, stream);
      // CSV escapes embedded quotes by doubling them, and quotes fields with commas.
      output << '"';
      for (char character : accessPath) {
        if (character == '"') {
          output << '"';
        }
        output << character;
      }
      output << "\"," << layoutId << ',' << mapping.wireIdsByLayout[layoutId] << '\n';
    }
  }

  /// Create wires from the R1CS map and retain the reverse lookup by LLZK layout ID.
  FailureOr<r1cs::CircuitDefOp>
  createR1CSWires(ModuleOp module, const LLZKLayout &layout, const R1CSWireMap &mapping) {
    OpBuilder builder(module.getBodyRegion());
    StringRef name = main.getSymName();
    if (auto origin = main->getAttrOfType<SymbolRefAttr>(polymorphic::SPECIALIZATION_ORIGIN_ATTR)) {
      name = origin.getLeafReference().getValue();
    }
    SmallVector<NamedAttribute> publicInputs;
    SmallVector<int64_t> inputLabels;
    unsigned inputCount = 0;
    for (const R1CSWire &entry : mapping.entries) {
      if (entry.isInput()) {
        if (entry.kind == WireKind::PublicInput) {
          publicInputs.emplace_back(
              builder.getStringAttr(std::to_string(inputCount)), builder.getAttr<r1cs::PublicAttr>()
          );
        }
        inputLabels.push_back(entry.wireId);
        ++inputCount;
      }
    }
    auto circuit = r1cs::CircuitDefOp::create(
        builder, main.getLoc(), name, builder.getDictionaryAttr(publicInputs)
    );
    circuit->setAttr(r1cs::INPUT_LABELS_ATTR, builder.getDenseI64ArrayAttr(inputLabels));
    builder.setInsertionPointToEnd(circuit.addEntryBlock());
    wires.resize(layout.signalPaths.size());
    for (const R1CSWire &entry : mapping.entries) {
      Value wire;
      if (entry.isInput()) {
        wire = circuit.getBody().front().addArgument(
            builder.getType<r1cs::SignalType>(), main.getLoc()
        );
      } else {
        auto visibility = entry.kind == WireKind::PublicOutput ? builder.getAttr<r1cs::PublicAttr>()
                                                               : r1cs::PublicAttr();
        wire = r1cs::SignalDefOp::create(
            builder, main.getLoc(), builder.getType<r1cs::SignalType>(),
            builder.getUI32IntegerAttr(entry.wireId), visibility
        );
      }
      wires[entry.layoutId] = wire;
    }
    return circuit;
  }

  /// Cache each normalized polynomial across all equality constraints.
  R1CSConstraint lowerExpression(Value root) {
    if (auto found = expressions.find(root); found != expressions.end()) {
      return found->second;
    }
    for (Value value : getExpressionOrder(root)) {
      if (expressions.contains(value)) {
        continue;
      }
      Operation *op = value.getDefiningOp();
      R1CSConstraint result;
      if (!op || isStorageRead(op)) {
        result.c.addTerm(value, 1);
      } else if (auto constant = dyn_cast<FeltConstantOp>(op)) {
        result.c.constant = constant.getValue().getReducedValueOrRaw();
      } else if (isa<AddFeltOp>(op)) {
        result = expressions.at(op->getOperand(0)).add(expressions.at(op->getOperand(1)));
      } else if (isa<SubFeltOp>(op)) {
        result = expressions.at(op->getOperand(0)).add(expressions.at(op->getOperand(1)).negated());
      } else if (isa<MulFeltOp>(op)) {
        result = expressions.at(op->getOperand(0)).multiply(expressions.at(op->getOperand(1)));
      } else if (isa<NegFeltOp>(op)) {
        result = expressions.at(op->getOperand(0)).negated();
      }
      expressions[value] = result;
    }
    return expressions.at(root);
  }

  /// Emit nonzero terms in increasing layout-ID order, merging repeated signal
  /// reads and reusing each wire's linear conversion. Constants are emitted first.
  Value emitLinearCombination(
      const LinearCombination &expression, const LLZKLayout &layout, OpBuilder &builder,
      Location loc
  ) {
    auto type = builder.getType<r1cs::LinearType>();
    Value result;
    if (expression.constant != 0) {
      result = r1cs::ConstOp::create(
          builder, loc, type, builder.getAttr<r1cs::FeltAttr>(expression.constant)
      );
    }
    // Distinct SSA reads can name the same signal. Combine them before sorting
    // so emission never depends on DenseMap iteration or equal-key sort order.
    DenseMap<uint64_t, DynamicAPInt> coefficientsById;
    for (const auto &[value, coefficient] : expression.terms) {
      uint64_t id = *layout.getSignalId(paths.at(value));
      auto [entry, inserted] = coefficientsById.try_emplace(id, coefficient);
      if (!inserted) {
        entry->second += coefficient;
      }
    }
    SmallVector<std::pair<uint64_t, DynamicAPInt>> terms;
    for (const auto &[id, coefficient] : coefficientsById) {
      if (coefficient != 0) {
        terms.emplace_back(id, coefficient);
      }
    }
    llvm::sort(terms, [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    for (const auto &[id, coefficient] : terms) {
      Value wire = wires[id];
      Value &linear = linearWires[wire];
      if (!linear) {
        linear = r1cs::ToLinearOp::create(builder, loc, type, wire);
      }
      Value term = linear;
      if (coefficient != 1) {
        term = r1cs::MulConstOp::create(
            builder, loc, type, term, builder.getAttr<r1cs::FeltAttr>(coefficient)
        );
      }
      result = result ? r1cs::AddOp::create(builder, loc, type, result, term).getResult() : term;
    }
    if (!result) {
      result = r1cs::ConstOp::create(
          builder, loc, type, builder.getAttr<r1cs::FeltAttr>(DynamicAPInt(0))
      );
    }
    return result;
  }

  /// Emit each equality as a product of linear combinations equal to a linear combination.
  LogicalResult lowerConstraintsToR1CS(r1cs::CircuitDefOp circuit, const LLZKLayout &layout) {
    OpBuilder builder = OpBuilder::atBlockEnd(&circuit.getBody().front());
    for (auto eq : constrain.getOps<EmitEqualityOp>()) {
      auto lhs = lowerExpression(eq.getLhs());
      auto rhs = lowerExpression(eq.getRhs());
      R1CSConstraint result;
      if (degrees.lookup(eq.getLhs()) == 2) {
        result = lhs;
        result.c = rhs.c.add(lhs.c.negated());
      } else if (degrees.lookup(eq.getRhs()) == 2) {
        result = rhs;
        result.c = lhs.c.add(rhs.c.negated());
      } else {
        result = rhs.add(lhs.negated());
      }
      Value a = emitLinearCombination(result.a, layout, builder, eq.getLoc());
      Value b = emitLinearCombination(result.b, layout, builder, eq.getLoc());
      Value c = emitLinearCombination(result.c, layout, builder, eq.getLoc());
      r1cs::ConstrainOp::create(builder, eq.getLoc(), a, b, c);
    }
    return success();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<r1cs::R1CSDialect>();
  }

  /// Check side-output paths before modifying the input module.
  LogicalResult validateOutputOptions(ModuleOp module) {
    StringRef llzkPath = outputLLZK.getValue();
    StringRef layoutPath = outputLayout.getValue();
    if (llzkPath.empty() && layoutPath.empty()) {
      return success();
    }
    if (llzkPath == "-" || layoutPath == "-") {
      return module.emitError(
          "output-llzk and output-layout require file paths, not standard output"
      );
    }
    if (!llzkPath.empty() && llzkPath == layoutPath) {
      return module.emitError("output-llzk and output-layout require distinct file paths");
    }
    return success();
  }

  /// Write the LLZK snapshot and CSV only after R1CS conversion succeeds.
  LogicalResult emitOutputs(
      ModuleOp module, StringRef llzkIR, const LLZKLayout &layout, const R1CSWireMap &mapping
  ) {
    std::string csv;
    if (!outputLayout.getValue().empty()) {
      llvm::raw_string_ostream stream(csv);
      emitLayoutCSV(layout, mapping, stream);
    }
    SmallVector<std::pair<StringRef, StringRef>> outputs;
    if (!outputLLZK.getValue().empty()) {
      outputs.emplace_back(outputLLZK.getValue(), llzkIR);
    }
    if (!outputLayout.getValue().empty()) {
      outputs.emplace_back(outputLayout.getValue(), csv);
    }
    SmallVector<std::unique_ptr<llvm::ToolOutputFile>> files;
    for (auto [path, contents] : outputs) {
      std::string error;
      auto file = mlir::openOutputFile(path, &error);
      if (!file) {
        return module.emitError() << "cannot open R1CS lowering output '" << path << "': " << error;
      }
      file->os() << contents;
      file->os().flush();
      if (file->os().has_error()) {
        std::string message = file->os().error().message();
        file->os().clear_error();
        return module.emitError() << "cannot write R1CS lowering output '" << path
                                  << "': " << message;
      }
      files.push_back(std::move(file));
    }
    for (auto &file : files) {
      file->keep();
    }
    return success();
  }

  void runOnOperation() override {
    resetState();
    ModuleOp module = getOperation();
    if (failed(validateOutputOptions(module))) {
      return signalPassFailure();
    }
    SymbolTableCollection tables;
    auto definition = getMainInstanceDef(tables, module);
    if (failed(definition) || !*definition) {
      if (succeeded(definition)) {
        module.emitError("R1CS lowering requires a concrete llzk.main");
      }
      return signalPassFailure();
    }
    main = definition->get();
    constrain = main.getConstrainFuncOp();
    if (!constrain || constrain.isExternal()) {
      main.emitOpError("R1CS lowering requires a constrain body");
      return signalPassFailure();
    }
    if (failed(validateDegreeAtMostTwo()) || failed(prepareAuxiliarySignals())) {
      return signalPassFailure();
    }
    auto layout = generateSignalLayout(module);
    if (failed(layout)) {
      return signalPassFailure();
    }
    auto wireMap = buildR1CSWireMap(*layout);
    if (failed(wireMap)) {
      return signalPassFailure();
    }
    std::string llzkIR;
    if (!outputLLZK.getValue().empty()) {
      if (failed(verify(module))) {
        return signalPassFailure();
      }
      llvm::raw_string_ostream stream(llzkIR);
      module->print(stream);
      stream << '\n';
    }
    auto circuit = createR1CSWires(module, *layout, *wireMap);
    if (failed(circuit)) {
      return signalPassFailure();
    }
    if (failed(lowerConstraintsToR1CS(*circuit, *layout))) {
      circuit->erase();
      return signalPassFailure();
    }
    for (Operation &op : llvm::make_early_inc_range(module.getBody()->getOperations())) {
      if (&op != circuit->getOperation()) {
        op.erase();
      }
    }
    module->removeAttr(MAIN_ATTR_NAME);
    module->setAttr(LANG_ATTR_NAME, StringAttr::get(module.getContext(), "r1cs"));
    // Reject invalid converted IR before committing any side outputs.
    if (failed(verify(module))) {
      return signalPassFailure();
    }
    if (failed(emitOutputs(module, llzkIR, *layout, *wireMap))) {
      return signalPassFailure();
    }
  }
};

} // namespace
