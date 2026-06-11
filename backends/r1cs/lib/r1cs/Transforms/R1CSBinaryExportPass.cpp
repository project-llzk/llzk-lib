//===-- R1CSBinaryExportPass.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-r1cs-export-binary` pass.
///
//===----------------------------------------------------------------------===//

#include "r1cs/Dialect/IR/Ops.h"
#include "r1cs/Transforms/TransformationPasses.h"

#include "llzk/Util/Compare.h"
#include "llzk/Util/DynamicAPIntHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/FileUtilities.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Endian.h>
#include <llvm/Support/ToolOutputFile.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace r1cs {
#define GEN_PASS_DECL_R1CSBINARYEXPORTPASS
#define GEN_PASS_DEF_R1CSBINARYEXPORTPASS
#include "r1cs/Transforms/TransformationPasses.h.inc"
} // namespace r1cs

using namespace mlir;

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
    auto diag = moduleOp.emitOpError("contains multiple r1cs.circuit ops; specify 'circuit-name'");
    diag << " (available:";
    for (auto circuit : circuits) {
      diag << " @" << circuit.getSymName();
    }
    diag << ')';
    return failure();
  }

  return circuits.front();
}

static FailureOr<llvm::APInt> parsePrime(ModuleOp moduleOp, StringRef primeText) {
  if (primeText.empty()) {
    return moduleOp.emitOpError() << "R1CS binary export requires a non-empty 'prime' option";
  }
  if (!llvm::all_of(primeText, llvm::isDigit)) {
    return moduleOp.emitOpError() << "'prime' must be a base-10 integer";
  }

  // `APInt` requires a bit width up front when parsing from decimal text. Four
  // bits per digit is intentionally loose but always sufficient because
  // `10 < 2^4`.
  unsigned bits = std::max(1u, 4u * static_cast<unsigned>(primeText.size()));
  llvm::APInt tmp(bits, primeText, 10);
  unsigned activeBits = std::max(1u, tmp.getActiveBits());
  llvm::APInt prime = tmp.zextOrTrunc(activeBits);
  if (prime.ule(1)) {
    return moduleOp.emitOpError() << "'prime' must be greater than 1";
  }

  return prime;
}

class BinaryBuffer {
public:
  void writeU32(uint32_t value) { writeInteger(value); }

  void writeU64(uint64_t value) { writeInteger(value); }

  void writeBytes(ArrayRef<char> bytes) { buffer.append(bytes.begin(), bytes.end()); }

  void writeFieldElement(uint32_t fieldSizeBytes, const llvm::DynamicAPInt &value) {
    llvm::APInt fieldValue = llzk::toExactWidthAPInt(value, fieldSizeBytes * CHAR_BIT);
    for (uint32_t byteIndex = 0; byteIndex < fieldSizeBytes; ++byteIndex) {
      buffer.push_back(static_cast<char>(fieldValue.extractBitsAsZExtValue(8, byteIndex * 8)));
    }
  }

  uint64_t size() const { return llzk::checkedCast<uint64_t>(buffer.size()); }

  ArrayRef<char> bytes() const { return buffer; }

private:
  template <typename T> void writeInteger(T value) {
    char bytes[sizeof(T)];
    llvm::support::endian::write<T, llvm::endianness::little>(bytes, value);
    writeBytes(bytes);
  }

  SmallVector<char> buffer;
};

enum class ExportWireClass {
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
  CircuitExportModelBuilder(r1cs::CircuitDefOp circuit, const llvm::APInt &prime)
      : circuit(circuit), primeModulus(llzk::toDynamicAPInt(prime)) {}

  FailureOr<ExportedCircuit> build() {
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
        signalDef.emitOpError() << "label 0 is reserved for the implicit one wire in .r1cs";
        return failure();
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

  void
  addReducedTerm(LinearAccumulator &accumulator, uint32_t wireId, llvm::DynamicAPInt coeff) const {
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
        user->emitOpError() << "cannot export block-defined !r1cs.linear values; expected linear "
                               "expressions built from r1cs.{to_linear,const,add,mul_const,neg}";
        return failure();
      }

      if (auto toLinear = dyn_cast<r1cs::ToLinearOp>(defOp)) {
        LinearAccumulator accumulator;
        auto wireIt = model.wireIdsBySignal.find(toLinear.getInput());
        if (wireIt == model.wireIdsBySignal.end()) {
          failedLinearValues.insert(value);
          toLinear.emitOpError()
              << "references a signal that is not a circuit input or r1cs.def result";
          return failure();
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
        defOp->emitOpError() << "cannot be exported as a .r1cs linear combination; expected one of "
                                "r1cs.to_linear, r1cs.const, r1cs.add, r1cs.mul_const, or r1cs.neg";
        return failure();
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

struct R1CSBinaryExportPass : public r1cs::impl::R1CSBinaryExportPassBase<R1CSBinaryExportPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    if (outputFilename.empty() || outputFilename == "-") {
      moduleOp.emitOpError() << "R1CS binary export requires a non-empty 'output-file'";
      signalPassFailure();
      return;
    }

    FailureOr<r1cs::CircuitDefOp> selectedCircuit = selectCircuit(moduleOp, circuitName);
    if (failed(selectedCircuit)) {
      signalPassFailure();
      return;
    }

    FailureOr<llvm::APInt> parsedPrime = parsePrime(moduleOp, prime);
    if (failed(parsedPrime)) {
      signalPassFailure();
      return;
    }

    CircuitExportModelBuilder modelBuilder(*selectedCircuit, *parsedPrime);
    FailureOr<ExportedCircuit> exportedCircuit = modelBuilder.build();
    if (failed(exportedCircuit)) {
      signalPassFailure();
      return;
    }

    FailureOr<BinaryBuffer> binary =
        serializeExportedCircuit(*selectedCircuit, *parsedPrime, *exportedCircuit);
    if (failed(binary)) {
      signalPassFailure();
      return;
    }

    std::unique_ptr<llvm::ToolOutputFile> outputFile = openOutputFile(outputFilename);
    if (!outputFile) {
      signalPassFailure();
      return;
    }

    outputFile->os().write(binary->bytes().data(), llzk::checkedCast<size_t>(binary->size()));
    outputFile->os().flush();
    if (outputFile->os().has_error()) {
      moduleOp.emitOpError() << "failed to write " << outputFilename;
      signalPassFailure();
      return;
    }
    outputFile->keep();
  }
};

} // namespace

std::unique_ptr<Pass> r1cs::createR1CSBinaryExportPass() {
  return std::make_unique<R1CSBinaryExportPass>();
}
