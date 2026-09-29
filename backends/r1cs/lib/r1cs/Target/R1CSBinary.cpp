//===-- R1CSBinary.cpp - R1CS binary serialization --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements binary .r1cs serialization.
///
//===----------------------------------------------------------------------===//

#include "r1cs/Target/R1CSBinary.h"

#include "r1cs/Dialect/IR/Ops.h"

#include "llzk/Util/BinaryBuffer.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/DynamicAPIntHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringMap.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

using namespace mlir;
using llzk::BinaryBuffer;

namespace {

static FailureOr<r1cs::CircuitDefOp> selectCircuit(ModuleOp moduleOp, StringRef circuitName) {
  if (!circuitName.empty()) {
    StringRef normalized = circuitName;
    normalized.consume_front("@");

    auto *symbol = SymbolTable::lookupSymbolIn(moduleOp, normalized);
    if (!symbol) {
      return moduleOp.emitOpError() << "could not find r1cs.circuit @" << normalized;
    }

    auto circuit = dyn_cast<r1cs::CircuitDefOp>(symbol);
    if (!circuit) {
      return moduleOp.emitOpError() << "symbol @" << normalized << " is not an r1cs.circuit";
    }

    return circuit;
  }

  SmallVector<r1cs::CircuitDefOp> circuits;
  for (auto circuit : moduleOp.getOps<r1cs::CircuitDefOp>()) {
    circuits.push_back(circuit);
  }

  if (circuits.empty()) {
    return moduleOp.emitOpError() << "does not contain an r1cs.circuit to export";
  }
  if (circuits.size() > 1) {
    auto diag =
        moduleOp.emitOpError("contains multiple r1cs.circuit ops; specify '--r1cs-circuit-name'");
    diag << " (available:";
    for (auto circuit : circuits) {
      diag << " @" << circuit.getSymName();
    }
    diag << ')';
    return diag;
  }

  return circuits.front();
}

/// Check that untrusted metadata fits the nonnegative signed-id representation.
static bool isLayoutIndex(IntegerAttr value) {
  return value && !value.getValue().isNegative() && value.getValue().getActiveBits() <= 63;
}

/// Print a storage path using roots that are meaningful outside the IR.
///
/// Root zero is the main circuit instance. Other roots are constrain-function
/// arguments printed as arg["name"] when named or argN otherwise. Names use
/// MLIR string literals so punctuation and escapes remain unambiguous.
static LogicalResult
printSymbolPath(ArrayAttr path, DictionaryAttr rootNames, llvm::raw_ostream &output) {
  if (path.empty()) {
    return failure();
  }
  auto root = dyn_cast<IntegerAttr>(path.getValue().front());
  if (!isLayoutIndex(root)) {
    return failure();
  }
  auto name = rootNames ? rootNames.getAs<StringAttr>(std::to_string(root.getInt())) : StringAttr();
  if (root.getInt() == 0) {
    output << "main";
  } else if (name) {
    output << "arg[";
    name.print(output);
    output << ']';
  } else {
    output << "arg" << root.getInt();
  }
  for (Attribute segment : path.getValue().drop_front()) {
    output << '[';
    if (auto member = dyn_cast<StringAttr>(segment)) {
      member.print(output);
    } else if (auto index = dyn_cast<IntegerAttr>(segment); isLayoutIndex(index)) {
      output << index.getInt();
    } else {
      return failure();
    }
    output << ']';
  }
  return success();
}

static FailureOr<llvm::APInt> parsePrime(ModuleOp moduleOp, StringRef primeText) {
  if (primeText.empty()) {
    return moduleOp.emitOpError()
           << "R1CS binary export requires a non-empty '--r1cs-prime' option";
  }
  if (!llvm::all_of(primeText, llvm::isDigit)) {
    return moduleOp.emitOpError() << "'--r1cs-prime' must be a base-10 integer";
  }

  // `APInt` requires a bit width up front when parsing from decimal text. Four
  // bits per digit is intentionally loose but always sufficient because
  // `10 < 2^4`.
  unsigned bits = std::max(1u, 4u * static_cast<unsigned>(primeText.size()));
  llvm::APInt tmp(bits, primeText, 10);
  unsigned activeBits = std::max(1u, tmp.getActiveBits());
  llvm::APInt prime = tmp.zextOrTrunc(activeBits);
  if (prime.ule(1)) {
    return moduleOp.emitOpError() << "'--r1cs-prime' must be greater than 1";
  }

  return prime;
}

/// Read the logical layout id carried by a signal in the binary export model.
static IntegerAttr getLayoutSignalId(Value signal, ArrayAttr argumentSignals) {
  if (auto argument = dyn_cast<BlockArgument>(signal)) {
    return argumentSignals && argument.getArgNumber() < argumentSignals.size()
               ? dyn_cast<IntegerAttr>(argumentSignals[argument.getArgNumber()])
               : IntegerAttr();
  }
  if (Operation *definition = signal.getDefiningOp()) {
    return definition->getAttrOfType<IntegerAttr>(r1cs::LAYOUT_SIGNAL_ATTR_NAME);
  }
  return {};
}

enum class ExportWireClass : std::uint8_t {
  ConstantOne,
  PublicOutput,
  PublicInput,
  PrivateInput,
  InternalSignal,
};

/// A canonical linear term ready for binary serialization.
///
/// Coefficients are always reduced modulo the requested prime and zero
/// coefficients are removed before these terms are materialized.
struct ExportLinearTerm {
  uint32_t wireId;
  llvm::DynamicAPInt coefficient;
};

/// A canonical linear combination sorted by ascending wire id.
struct ExportLinearCombination {
  SmallVector<ExportLinearTerm> terms;
};

struct ExportConstraint {
  ExportLinearCombination a;
  ExportLinearCombination b;
  ExportLinearCombination c;
};

struct ExportedWireInfo {
  Value signal;
  uint32_t wireId;
  uint64_t labelId;
  ExportWireClass wireClass;
};

struct ExportedCircuit {
  SmallVector<ExportedWireInfo> wires;
  SmallVector<uint64_t> wireToLabel;
  DenseMap<Value, uint32_t> wireIdsBySignal;
  SmallVector<ExportConstraint> constraints;
  uint32_t numPublicOutputs = 0;
  uint32_t numPublicInputs = 0;
  uint32_t numPrivateInputs = 0;
  uint32_t numWires = 0;
  uint64_t numLabels = 0;
};

/// Builds the in-memory representation that will later be serialized to `.r1cs`.
///
/// Key assumptions documented here because they affect the binary layout:
/// 1. `wire 0` is always the implicit constant-one wire mandated by the format.
/// 2. Exported wire ids are assigned in the order required by the spec:
///    public outputs, public inputs, private inputs, then remaining internal
///    signals.
/// 3. `r1cs.def` carries the only explicit source labels available in the IR, so
///    those labels are preserved in the exported wire-to-label map. Block
///    arguments do not have labels in the current dialect, so the exporter
///    synthesizes fresh label ids above the largest explicit `r1cs.def` label.
/// 4. `wire 0 -> label 0` is reserved for the implicit one wire. Any explicit
///    `r1cs.def` with label `0` is rejected because it would collide with that
///    reserved mapping.
class CircuitExportModelBuilder {
public:
  explicit CircuitExportModelBuilder(r1cs::CircuitDefOp circuitOp) : circuit(circuitOp) {}

  FailureOr<ExportedCircuit> build(const llvm::APInt &prime) {
    primeModulus = llzk::toDynamicAPInt(prime);
    Block &entryBlock = circuit.getBody().front();
    if (failed(assignWires())) {
      return failure();
    }

    for (auto constrainOp : entryBlock.getOps<r1cs::ConstrainOp>()) {
      FailureOr<ExportLinearCombination> a = flattenLinear(constrainOp.getA(), constrainOp);
      FailureOr<ExportLinearCombination> b = flattenLinear(constrainOp.getB(), constrainOp);
      FailureOr<ExportLinearCombination> c = flattenLinear(constrainOp.getC(), constrainOp);
      if (failed(a) || failed(b) || failed(c)) {
        return failure();
      }

      model.constraints.push_back({*a, *b, *c});
    }

    return model;
  }

  /// Build just the wire model for layout-map validation.  Layout maps must
  /// not depend on a placeholder field modulus or flatten constraints.
  FailureOr<ExportedCircuit> buildWireLayout() {
    if (failed(assignWires())) {
      return failure();
    }
    return model;
  }

private:
  struct LinearAccumulator {
    llvm::SmallDenseMap<uint32_t, llvm::DynamicAPInt, 8> coefficients;
  };

  llvm::DynamicAPInt decodeFieldElement(r1cs::FeltAttr attr) const {
    // FeltAttr stores an IntegerAttr with a signless IntegerType, so we cannot
    // use `IntegerAttr::getAPSInt()` here. The lowering constructs felt
    // literals from signed APSInts, so we explicitly recover that signed
    // interpretation before reducing modulo the export field prime.
    llvm::APSInt signedValue(attr.getValue().getValue(), false);
    return reduce(llzk::toDynamicAPInt(signedValue));
  }

  llvm::DynamicAPInt reduce(const llvm::DynamicAPInt &value) const {
    llvm::DynamicAPInt reduced = value % primeModulus;
    if (reduced < 0) {
      reduced += primeModulus;
    }
    return reduced;
  }

  FailureOr<uint64_t> assignFreshLabel(uint64_t nextLabel, StringRef kind, Location loc) const {
    if (nextLabel == std::numeric_limits<uint64_t>::max()) {
      return emitError(loc) << "ran out of label ids while assigning a " << kind << " wire";
    }
    return nextLabel;
  }

  void addWire(Value signal, uint32_t wireId, uint64_t labelId, ExportWireClass wireClass) {
    model.wires.push_back({signal, wireId, labelId, wireClass});
    model.wireIdsBySignal.try_emplace(signal, wireId);
    model.wireToLabel.push_back(labelId);
  }

  LogicalResult assignWires() {
    Block &entryBlock = circuit.getBody().front();
    SmallVector<r1cs::SignalDefOp> publicOutputs;
    SmallVector<r1cs::SignalDefOp> internalSignals;
    uint64_t nextFreshLabel = 1;
    llvm::SmallBitVector publicInputMask(entryBlock.getNumArguments(), false);
    uint32_t numPublicInputs = 0;
    auto argAttrs = circuit.getArgAttrs();

    for (BlockArgument arg : entryBlock.getArguments()) {
      Attribute attr = argAttrs ? argAttrs->get(std::to_string(arg.getArgNumber())) : Attribute();
      if (!llvm::isa_and_nonnull<r1cs::PublicAttr>(attr)) {
        continue;
      }
      publicInputMask.set(arg.getArgNumber());
      numPublicInputs++;
    }

    // Scan signal definitions first so synthesized input labels can be placed
    // strictly after the largest explicit signal label. This preserves the
    // source labels that already exist in the dialect while keeping synthesized
    // labels collision-free.
    for (auto signalDef : entryBlock.getOps<r1cs::SignalDefOp>()) {
      uint32_t label = signalDef.getLabel();
      if (label == 0) {
        return signalDef.emitOpError() << "label 0 is reserved for the implicit one wire in .r1cs";
      }

      nextFreshLabel = std::max(nextFreshLabel, static_cast<uint64_t>(label) + 1);
      if (signalDef.getPub().has_value()) {
        publicOutputs.push_back(signalDef);
      } else {
        internalSignals.push_back(signalDef);
      }
    }

    model.wireToLabel.push_back(0);

    uint32_t nextWireId = 1;
    for (auto signalDef : publicOutputs) {
      addWire(
          signalDef.getOut(), nextWireId++, static_cast<uint64_t>(signalDef.getLabel()),
          ExportWireClass::PublicOutput
      );
    }
    model.numPublicOutputs = llzk::checkedCast<uint32_t>(publicOutputs.size());

    for (BlockArgument arg : entryBlock.getArguments()) {
      if (!publicInputMask.test(arg.getArgNumber())) {
        continue;
      }

      FailureOr<uint64_t> freshLabel =
          assignFreshLabel(nextFreshLabel, "public input", arg.getLoc());
      if (failed(freshLabel)) {
        return failure();
      }
      addWire(arg, nextWireId++, *freshLabel, ExportWireClass::PublicInput);
      nextFreshLabel = *freshLabel + 1;
    }

    for (BlockArgument arg : entryBlock.getArguments()) {
      if (publicInputMask.test(arg.getArgNumber())) {
        continue;
      }

      FailureOr<uint64_t> freshLabel =
          assignFreshLabel(nextFreshLabel, "private input", arg.getLoc());
      if (failed(freshLabel)) {
        return failure();
      }
      addWire(arg, nextWireId++, *freshLabel, ExportWireClass::PrivateInput);
      nextFreshLabel = *freshLabel + 1;
    }

    uint32_t numArguments = llzk::checkedCast<uint32_t>(entryBlock.getNumArguments());
    model.numPublicInputs = numPublicInputs;
    model.numPrivateInputs = numArguments - model.numPublicInputs;

    for (auto signalDef : internalSignals) {
      addWire(
          signalDef.getOut(), nextWireId++, static_cast<uint64_t>(signalDef.getLabel()),
          ExportWireClass::InternalSignal
      );
    }

    model.numWires = nextWireId;
    // `nLabels` tracks the exported label-id space, not just the number of
    // used wires, because explicit `r1cs.def` labels may be sparse.
    model.numLabels = nextFreshLabel;
    return success();
  }

  void addReducedTerm(
      LinearAccumulator &accumulator, uint32_t wireId, const llvm::DynamicAPInt &coeff
  ) const {
    llvm::DynamicAPInt reducedCoeff = reduce(coeff);
    if (reducedCoeff == 0) {
      return;
    }

    auto it = accumulator.coefficients.find(wireId);
    if (it == accumulator.coefficients.end()) {
      accumulator.coefficients.try_emplace(wireId, reducedCoeff);
      return;
    }

    it->second = reduce(it->second + reducedCoeff);
    if (it->second == 0) {
      accumulator.coefficients.erase(it);
    }
  }

  void
  addCombination(LinearAccumulator &accumulator, const ExportLinearCombination &combination) const {
    for (const ExportLinearTerm &term : combination.terms) {
      addReducedTerm(accumulator, term.wireId, term.coefficient);
    }
  }

  void addScaledCombination(
      LinearAccumulator &accumulator, const ExportLinearCombination &combination,
      const llvm::DynamicAPInt &factor
  ) const {
    llvm::DynamicAPInt reducedFactor = reduce(factor);
    if (reducedFactor == 0) {
      return;
    }

    for (const ExportLinearTerm &term : combination.terms) {
      addReducedTerm(accumulator, term.wireId, term.coefficient * reducedFactor);
    }
  }

  ExportLinearCombination canonicalize(LinearAccumulator &&accumulator) const {
    ExportLinearCombination result;
    SmallVector<std::pair<uint32_t, llvm::DynamicAPInt>> sortedTerms;
    sortedTerms.reserve(accumulator.coefficients.size());
    for (const auto &entry : accumulator.coefficients) {
      sortedTerms.push_back(entry);
    }

    llvm::sort(sortedTerms, [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    for (const auto &[wireId, coefficient] : sortedTerms) {
      if (coefficient != 0) {
        result.terms.push_back({wireId, coefficient});
      }
    }

    return result;
  }

  const ExportLinearCombination *lookupFlattenedOperand(Value operand, Value parentValue) {
    auto it = flattenedLinearMemo.find(operand);
    if (it != flattenedLinearMemo.end()) {
      return &it->second;
    }

    failedLinearValues.insert(parentValue);
    return nullptr;
  }

  FailureOr<ExportLinearCombination> flattenLinear(Value root, Operation *user) {
    if (auto it = flattenedLinearMemo.find(root); it != flattenedLinearMemo.end()) {
      return it->second;
    }
    if (failedLinearValues.contains(root)) {
      return failure();
    }

    // The R1CS lowering pass builds left-associated `r1cs.add` chains, so recursive
    // flattening can overflow the C stack on large circuits. We use an
    // explicit post-order walk instead and memoize every visited sub-expression.
    SmallVector<std::pair<Value, bool>> stack;
    stack.push_back({root, false});

    while (!stack.empty()) {
      auto [value, expanded] = stack.pop_back_val();
      if (flattenedLinearMemo.contains(value) || failedLinearValues.contains(value)) {
        continue;
      }

      Operation *defOp = value.getDefiningOp();
      if (!defOp) {
        failedLinearValues.insert(value);
        return user->emitOpError()
               << "cannot export block-defined !r1cs.linear values; expected linear "
                  "expressions built from r1cs.{to_linear,const,add,mul_const,neg}";
      }

      if (auto toLinear = dyn_cast<r1cs::ToLinearOp>(defOp)) {
        LinearAccumulator accumulator;
        auto wireIt = model.wireIdsBySignal.find(toLinear.getInput());
        if (wireIt == model.wireIdsBySignal.end()) {
          failedLinearValues.insert(value);
          return toLinear.emitOpError()
                 << "references a signal that is not a circuit input or r1cs.def result";
        }

        addReducedTerm(accumulator, wireIt->second, llvm::DynamicAPInt(1));
        flattenedLinearMemo.try_emplace(value, canonicalize(std::move(accumulator)));
        continue;
      }

      if (auto constOp = dyn_cast<r1cs::ConstOp>(defOp)) {
        LinearAccumulator accumulator;
        // Constants are lowered onto the implicit one wire required by `.r1cs`.
        addReducedTerm(accumulator, 0, decodeFieldElement(constOp.getValue()));
        flattenedLinearMemo.try_emplace(value, canonicalize(std::move(accumulator)));
        continue;
      }

      if (!expanded) {
        stack.push_back({value, true});

        if (auto addOp = dyn_cast<r1cs::AddOp>(defOp)) {
          stack.push_back({addOp.getRhs(), false});
          stack.push_back({addOp.getLhs(), false});
          continue;
        }
        if (auto mulConstOp = dyn_cast<r1cs::MulConstOp>(defOp)) {
          stack.push_back({mulConstOp.getInput(), false});
          continue;
        }
        if (auto negOp = dyn_cast<r1cs::NegOp>(defOp)) {
          stack.push_back({negOp.getInput(), false});
          continue;
        }

        failedLinearValues.insert(value);
        return defOp->emitOpError()
               << "cannot be exported as a .r1cs linear combination; expected one of "
                  "r1cs.to_linear, r1cs.const, r1cs.add, r1cs.mul_const, or r1cs.neg";
      }

      LinearAccumulator accumulator;
      if (auto addOp = dyn_cast<r1cs::AddOp>(defOp)) {
        const ExportLinearCombination *lhs = lookupFlattenedOperand(addOp.getLhs(), value);
        const ExportLinearCombination *rhs = lookupFlattenedOperand(addOp.getRhs(), value);
        if (!lhs || !rhs) {
          return failure();
        }

        addCombination(accumulator, *lhs);
        addCombination(accumulator, *rhs);
      } else if (auto mulConstOp = dyn_cast<r1cs::MulConstOp>(defOp)) {
        const ExportLinearCombination *input = lookupFlattenedOperand(mulConstOp.getInput(), value);
        if (!input) {
          return failure();
        }

        addScaledCombination(accumulator, *input, decodeFieldElement(mulConstOp.getConstValue()));
      } else if (auto negOp = dyn_cast<r1cs::NegOp>(defOp)) {
        const ExportLinearCombination *input = lookupFlattenedOperand(negOp.getInput(), value);
        if (!input) {
          return failure();
        }

        addScaledCombination(accumulator, *input, llvm::DynamicAPInt(-1));
      } else {
        failedLinearValues.insert(value);
        return failure();
      }

      flattenedLinearMemo.try_emplace(value, canonicalize(std::move(accumulator)));
    }

    auto resultIt = flattenedLinearMemo.find(root);
    if (resultIt == flattenedLinearMemo.end()) {
      failedLinearValues.insert(root);
      return failure();
    }
    return resultIt->second;
  }

  r1cs::CircuitDefOp circuit;
  llvm::DynamicAPInt primeModulus;
  ExportedCircuit model;
  DenseMap<Value, ExportLinearCombination> flattenedLinearMemo;
  DenseSet<Value> failedLinearValues;
};

static FailureOr<uint32_t> computeFieldSizeBytes(Operation *op, const llvm::APInt &prime) {
  uint32_t minBytes = std::max(1u, (prime.getActiveBits() + 7u) / 8u);
  uint64_t roundedSize = ((static_cast<uint64_t>(minBytes) + 7u) / 8u) * 8u;
  if (!std::in_range<uint32_t>(roundedSize)) {
    return op->emitOpError() << "field size does not fit in a 32-bit header field";
  }
  return llzk::checkedCast<uint32_t>(roundedSize);
}

static void
writeSection(BinaryBuffer &fileBuffer, uint32_t sectionType, const BinaryBuffer &section) {
  fileBuffer.writeU32(sectionType);
  fileBuffer.writeU64(section.size());
  fileBuffer.writeBytes(section.bytes());
}

static LogicalResult serializeLinearCombination(
    r1cs::CircuitDefOp circuit, const ExportLinearCombination &combination, uint32_t numWires,
    uint32_t fieldSizeBytes, BinaryBuffer &buffer
) {
  buffer.writeU32(llzk::checkedCast<uint32_t>(combination.terms.size()));

  uint32_t previousWireId = 0;
  bool sawAnyTerm = false;
  for (const ExportLinearTerm &term : combination.terms) {
    if (term.wireId >= numWires) {
      return circuit.emitOpError() << "linear combination references wire " << term.wireId
                                   << " but only " << numWires << " wires were assigned";
    }
    if (sawAnyTerm && term.wireId <= previousWireId) {
      return circuit.emitOpError() << "linear combination terms must be sorted by wire id";
    }

    buffer.writeU32(term.wireId);
    buffer.writeFieldElement(fieldSizeBytes, term.coefficient);
    previousWireId = term.wireId;
    sawAnyTerm = true;
  }

  return success();
}

static FailureOr<BinaryBuffer> serializeExportedCircuit(
    r1cs::CircuitDefOp circuit, const llvm::APInt &prime, const ExportedCircuit &model
) {
  if (model.wireToLabel.size() != model.numWires) {
    return circuit.emitOpError() << "internal export error: wire-to-label map has "
                                 << model.wireToLabel.size() << " entries for " << model.numWires
                                 << " wires";
  }
  if (model.wireToLabel.empty() || model.wireToLabel.front() != 0) {
    return circuit.emitOpError() << "internal export error: wire 0 must map to label 0";
  }

  FailureOr<uint32_t> fieldSizeBytes = computeFieldSizeBytes(circuit, prime);
  if (failed(fieldSizeBytes)) {
    return failure();
  }

  BinaryBuffer headerSection;
  headerSection.writeU32(*fieldSizeBytes);
  headerSection.writeFieldElement(*fieldSizeBytes, llzk::toDynamicAPInt(prime));
  headerSection.writeU32(model.numWires);
  headerSection.writeU32(model.numPublicOutputs);
  headerSection.writeU32(model.numPublicInputs);
  headerSection.writeU32(model.numPrivateInputs);
  headerSection.writeU64(model.numLabels);
  headerSection.writeU32(llzk::checkedCast<uint32_t>(model.constraints.size()));

  BinaryBuffer constraintsSection;
  auto appendCombination = [&](const ExportLinearCombination &combination) -> LogicalResult {
    return serializeLinearCombination(
        circuit, combination, model.numWires, *fieldSizeBytes, constraintsSection
    );
  };
  for (const ExportConstraint &constraint : model.constraints) {
    if (failed(appendCombination(constraint.a)) || failed(appendCombination(constraint.b)) ||
        failed(appendCombination(constraint.c))) {
      return failure();
    }
  }

  BinaryBuffer wireToLabelSection;
  for (uint64_t labelId : model.wireToLabel) {
    wireToLabelSection.writeU64(labelId);
  }

  BinaryBuffer fileBuffer;
  fileBuffer.writeBytes({'r', '1', 'c', 's'});
  fileBuffer.writeU32(1);
  fileBuffer.writeU32(3);
  writeSection(fileBuffer, 0x01, headerSection);
  writeSection(fileBuffer, 0x02, constraintsSection);
  writeSection(fileBuffer, 0x03, wireToLabelSection);
  return fileBuffer;
}

} // namespace

LogicalResult r1cs::exportR1CSBinary(
    ModuleOp moduleOp, llvm::raw_ostream &output, StringRef prime, StringRef circuitName
) {
  FailureOr<r1cs::CircuitDefOp> selectedCircuit = selectCircuit(moduleOp, circuitName);
  if (failed(selectedCircuit)) {
    return failure();
  }

  FailureOr<llvm::APInt> parsedPrime = parsePrime(moduleOp, prime);
  if (failed(parsedPrime)) {
    return failure();
  }

  CircuitExportModelBuilder modelBuilder(*selectedCircuit);
  FailureOr<ExportedCircuit> exportedCircuit = modelBuilder.build(*parsedPrime);
  if (failed(exportedCircuit)) {
    return failure();
  }

  FailureOr<BinaryBuffer> binary =
      serializeExportedCircuit(*selectedCircuit, *parsedPrime, *exportedCircuit);
  if (failed(binary)) {
    return failure();
  }

  output.write(binary->bytes().data(), llzk::checkedCast<size_t>(binary->size()));
  return success();
}

LogicalResult
r1cs::exportLLZKLayoutMap(ModuleOp moduleOp, llvm::raw_ostream &output, StringRef circuitName) {
  FailureOr<r1cs::CircuitDefOp> selectedCircuit = selectCircuit(moduleOp, circuitName);
  if (failed(selectedCircuit)) {
    return failure();
  }

  ArrayAttr bindings = (*selectedCircuit)->getAttrOfType<ArrayAttr>(WIRE_BINDINGS_ATTR_NAME);
  if (!bindings) {
    return selectedCircuit->emitOpError()
           << "cannot export layout map: missing '" << WIRE_BINDINGS_ATTR_NAME
           << "' from direct R1CS lowering";
  }

  ArrayAttr layoutSignals = (*selectedCircuit)->getAttrOfType<ArrayAttr>(LAYOUT_SIGNALS_ATTR_NAME);
  DictionaryAttr rootNames =
      (*selectedCircuit)->getAttrOfType<DictionaryAttr>(LAYOUT_ROOT_NAMES_ATTR_NAME);
  if (!layoutSignals) {
    return selectedCircuit->emitOpError()
           << "cannot export layout map: missing '" << LAYOUT_SIGNALS_ATTR_NAME
           << "' from direct R1CS lowering";
  }
  SmallVector<std::string> paths;
  for (auto [id, attr] : llvm::enumerate(layoutSignals)) {
    auto signal = dyn_cast<DictionaryAttr>(attr);
    auto signalId = signal ? signal.getAs<IntegerAttr>("id") : IntegerAttr();
    auto path = signal ? signal.getAs<ArrayAttr>("path") : ArrayAttr();
    if (!isLayoutIndex(signalId) || !path || static_cast<uint64_t>(signalId.getInt()) != id) {
      return selectedCircuit->emitOpError()
             << "cannot export layout map: invalid layout signal " << id;
    }
    std::string rendered;
    llvm::raw_string_ostream pathOutput(rendered);
    if (failed(printSymbolPath(path, rootNames, pathOutput))) {
      return selectedCircuit->emitOpError()
             << "cannot export layout map: invalid layout signal path";
    }
    pathOutput.flush();
    paths.push_back(std::move(rendered));
  }
  for (auto [index, attr] : llvm::enumerate(bindings)) {
    auto binding = dyn_cast<DictionaryAttr>(attr);
    auto wire = binding ? binding.getAs<IntegerAttr>("wire") : IntegerAttr();
    auto path = binding ? binding.getAs<ArrayAttr>("path") : ArrayAttr();
    auto signal = binding ? binding.getAs<IntegerAttr>("signal") : IntegerAttr();
    uint64_t expectedWire = index + 1;
    if (!isLayoutIndex(wire) || !path || !isLayoutIndex(signal) ||
        static_cast<uint64_t>(signal.getInt()) >= paths.size() ||
        static_cast<uint64_t>(wire.getInt()) != expectedWire) {
      return selectedCircuit->emitOpError()
             << "cannot export layout map: expected '" << WIRE_BINDINGS_ATTR_NAME << "' entry "
             << index << " to contain wire " << expectedWire << " and an array path";
    }
    std::string rendered;
    llvm::raw_string_ostream pathOutput(rendered);
    if (failed(printSymbolPath(path, rootNames, pathOutput)) ||
        rendered != paths[signal.getInt()]) {
      return selectedCircuit->emitOpError()
             << "cannot export layout map: binding path does not match layout signal";
    }
  }

  // A layout map needs only the physical wire assignment.  In particular, do
  // not parse constraints using a dummy modulus, which could truncate a field
  // element or fail on otherwise valid R1CS text.
  CircuitExportModelBuilder modelBuilder(*selectedCircuit);
  FailureOr<ExportedCircuit> model = modelBuilder.buildWireLayout();
  if (failed(model) || model->numWires != bindings.size() + 1) {
    return selectedCircuit->emitOpError()
           << "cannot export layout map: '" << WIRE_BINDINGS_ATTR_NAME
           << "' does not match the physical R1CS wire layout";
  }

  output << "# LLZK layout map v1\n# signals\n";
  for (auto [id, path] : llvm::enumerate(paths)) {
    output << "signal " << id << '\t' << path << '\n';
  }
  output << "# r1cs\nwire 0\t<one>\n";
  ArrayAttr argumentSignals =
      (*selectedCircuit)->getAttrOfType<ArrayAttr>(LAYOUT_ARGUMENT_SIGNALS_ATTR_NAME);
  for (const ExportedWireInfo &wire : model->wires) {
    IntegerAttr signal = getLayoutSignalId(wire.signal, argumentSignals);
    if (!isLayoutIndex(signal) || static_cast<uint64_t>(signal.getInt()) >= paths.size()) {
      return selectedCircuit->emitOpError() << "cannot export layout map: physical wire "
                                            << wire.wireId << " has no valid layout signal";
    }
    auto binding = dyn_cast<DictionaryAttr>(bindings[wire.wireId - 1]);
    auto bindingWire = binding ? binding.getAs<IntegerAttr>("wire") : IntegerAttr();
    auto bindingSignal = binding ? binding.getAs<IntegerAttr>("signal") : IntegerAttr();
    if (!bindingWire || !bindingSignal || bindingWire.getInt() != wire.wireId ||
        bindingSignal.getInt() != signal.getInt()) {
      return selectedCircuit->emitOpError()
             << "cannot export layout map: '" << WIRE_BINDINGS_ATTR_NAME << "' entry for wire "
             << wire.wireId << " does not match the binary export signal";
    }
    output << "wire " << wire.wireId << "\tsignal " << signal.getInt() << '\n';
  }
  return success();
}
