//===-- Wtns.cpp - snarkjs-compatible witness output ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "Wtns.h"

#include "Errors.h"
#include "WitnessSelection.h"
#include "r1cs/Dialect/IR/Ops.h"
#include "r1cs/Target/R1CSBinary.h"
#include "r1cs/Transforms/TransformationPassPipelines.h"

#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Polymorphic/Transforms/ConstraintEvaluation.h"
#include "llzk/Util/BinaryBuffer.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/ToolOutputFile.h>

#include <climits>
#include <cstdint>
#include <limits>

using namespace mlir;

namespace llzk::witgen {
namespace {

constexpr char WTNS_MAGIC[] = {'w', 't', 'n', 's'};
constexpr uint32_t WTNS_VERSION = 2;
constexpr uint32_t WTNS_SECTION_COUNT = 2;
constexpr uint32_t WTNS_HEADER_SECTION = 1;
constexpr uint32_t WTNS_VALUES_SECTION = 2;
constexpr uint32_t WTNS_FIELD_LIMB_BITS = 64;
constexpr uint32_t WTNS_FIELD_LIMB_BYTES = WTNS_FIELD_LIMB_BITS / CHAR_BIT;

void appendSection(BinaryBuffer &file, uint32_t type, const BinaryBuffer &section) {
  file.writeU32(type);
  file.writeU64(section.size());
  file.writeBytes(section.bytes());
}

llvm::Expected<llvm::DynamicAPInt>
readFelt(const llvm::json::Object &object, StringRef key, const Field &field) {
  const llvm::json::Value *json = object.get(key);
  if (!json) {
    return makeError(llvm::Twine("full witness is missing '") + key + "'");
  }
  std::optional<StringRef> text = json->getAsString();
  if (!text || text->empty() || !llvm::all_of(*text, llvm::isDigit)) {
    return makeError(llvm::Twine("full witness value '") + key + "' is not a non-negative integer");
  }
  llvm::DynamicAPInt value = toDynamicAPInt(*text);
  if (value >= field.prime()) {
    return makeError(llvm::Twine("full witness value '") + key + "' is outside the field");
  }
  return value;
}

llvm::Expected<OwningOpRef<ModuleOp>> lowerModuleToR1CS(ModuleOp moduleOp) {
  OwningOpRef<ModuleOp> lowered = cast<ModuleOp>(moduleOp->clone());
  PassManager pm(lowered->getContext());
  r1cs::buildFullR1CSLoweringPipeline(
      pm, polymorphic::isEvaluatedModule(moduleOp) ? r1cs::R1CSLoweringMode::Direct
                                                   : r1cs::R1CSLoweringMode::Legacy
  );
  if (failed(pm.run(*lowered))) {
    return makeError("failed to lower a module clone while validating .wtns wire ordering");
  }
  return lowered;
}

llvm::Expected<size_t> getR1CSWireCount(ModuleOp r1csModule, StringRef circuitName) {
  auto circuit = r1csModule.lookupSymbol<r1cs::CircuitDefOp>(circuitName);
  if (!circuit) {
    return makeError("R1CS lowering did not produce a circuit for the llzk.main struct");
  }

  Block &entry = circuit.getBody().front();
  size_t wireCount = 1 + entry.getNumArguments();
  wireCount += llvm::range_size(entry.getOps<r1cs::SignalDefOp>());
  return wireCount;
}

llvm::Expected<std::string> getMainStructName(ModuleOp moduleOp) {
  SymbolTableCollection tables;
  auto mainDef = getMainInstanceDef(tables, moduleOp.getOperation());
  if (failed(mainDef) || !mainDef.value()) {
    return makeError("module is missing a concrete llzk.main struct");
  }
  return mainDef->get().getSymName().str();
}

llvm::Expected<SmallVector<llvm::DynamicAPInt>>
collectWitnessValues(ModuleOp moduleOp, const llvm::json::Value &fullWitness, const Field &field) {
  const auto *root = fullWitness.getAsObject();
  const auto *inputs = root ? root->getObject("inputs") : nullptr;
  const auto *signals = root ? root->getObject("signals") : nullptr;
  if (!inputs || !signals) {
    return makeError(".wtns output requires a full-witness llzk-witgen result");
  }

  SymbolTableCollection tables;
  auto mainDef = getMainInstanceDef(tables, moduleOp.getOperation());
  if (failed(mainDef) || !mainDef.value()) {
    return makeError("module is missing a concrete llzk.main struct");
  }
  auto compute = mainDef->get().getComputeFuncOp();
  auto constrain = mainDef->get().getConstrainFuncOp();
  if (!compute) {
    return makeError("main struct is missing @compute");
  }
  if (!constrain || constrain.getNumArguments() != compute.getNumArguments() + 1) {
    return makeError("main @constrain inputs do not match @compute inputs");
  }

  SmallVector<llvm::DynamicAPInt> witness;
  // Wire 0 is the implicit constant-one wire in R1CS and therefore the first
  // value in the corresponding snarkjs witness.
  witness.push_back(field.one());
  auto appendMemberClass = [&mainDef, signals, &field, &witness](bool isPublic) -> llvm::Error {
    for (component::MemberDefOp member : mainDef->get().getMemberDefs()) {
      if (member.hasPublicAttr() != isPublic) {
        continue;
      }
      if (!isa<felt::FeltType>(member.getType())) {
        return makeError(".wtns output currently requires scalar felt main members");
      }
      auto value = readFelt(*signals, member.getSymName(), field);
      if (!value) {
        return value.takeError();
      }
      witness.push_back(*value);
    }
    return llvm::Error::success();
  };
  auto inputBindings = collectInputBindings(compute);
  auto appendInputClass = [&inputBindings, &constrain, inputs, &field,
                           &witness](bool isPublic) -> llvm::Error {
    for (const InputBinding &binding : inputBindings) {
      // R1CS lowering derives input visibility from constrain(), whose first
      // argument is self and whose remaining arguments correspond to compute().
      if (constrain.hasArgPublicAttr(binding.index + 1) != isPublic) {
        continue;
      }
      if (!isa<felt::FeltType>(binding.type)) {
        return makeError(".wtns output currently requires scalar felt main inputs");
      }
      auto value = readFelt(*inputs, binding.name, field);
      if (!value) {
        return value.takeError();
      }
      witness.push_back(*value);
    }
    return llvm::Error::success();
  };

  if (auto error = appendMemberClass(true)) {
    return error;
  }
  if (auto error = appendInputClass(true)) {
    return error;
  }
  if (auto error = appendInputClass(false)) {
    return error;
  }
  if (auto error = appendMemberClass(false)) {
    return error;
  }

  return witness;
}

/// Read nested witness storage in exactly the order recorded by R1CS lowering.
llvm::Expected<SmallVector<llvm::DynamicAPInt>> collectEvaluatedWitness(
    ModuleOp module, ModuleOp lowered, const llvm::json::Value &fullWitness, const Field &field
) {
  auto mainRef = lowered->getAttrOfType<SymbolRefAttr>(r1cs::CIRCUIT_REF_ATTR_NAME);
  auto circuit = mainRef ? lowered.lookupSymbol<r1cs::CircuitDefOp>(mainRef.getRootReference())
                         : r1cs::CircuitDefOp();
  auto bindings =
      circuit ? circuit->getAttrOfType<ArrayAttr>(r1cs::WIRE_BINDINGS_ATTR_NAME) : ArrayAttr();
  if (!bindings) {
    return makeError("evaluated R1CS circuit is missing its wire map");
  }
  SymbolTableCollection tables;
  auto mainDef = getMainInstanceDef(tables, module.getOperation());
  if (failed(mainDef) || !mainDef.value()) {
    return makeError("missing evaluated main");
  }
  auto compute = mainDef->get().getComputeFuncOp();
  auto constrain = mainDef->get().getConstrainFuncOp();
  if (!compute) {
    return makeError("main struct is missing @compute");
  }
  if (!constrain || constrain.getNumArguments() != compute.getNumArguments() + 1) {
    return makeError("main @constrain inputs do not match @compute inputs");
  }
  auto inputs = collectInputBindings(compute);
  auto *root = fullWitness.getAsObject();
  if (!root) {
    return makeError("expected a full witness object");
  }
  SmallVector<llvm::DynamicAPInt> values {field.one()};
  for (auto item : bindings) {
    auto binding = polymorphic::getStorageBinding(item);
    auto dictionary = dyn_cast<DictionaryAttr>(item);
    auto wire = dictionary ? dictionary.getAs<IntegerAttr>("wire") : IntegerAttr();
    if (failed(binding) || !wire || wire.getValue().getActiveBits() > 63 ||
        wire.getInt() != static_cast<int64_t>(values.size())) {
      return makeError("invalid or out-of-order R1CS wire binding");
    }
    auto path = binding->path;
    auto argument = cast<IntegerAttr>(path[0]).getInt();
    const llvm::json::Value *value = root->get(argument == 0 ? "signals" : "inputs");
    if (argument != 0) {
      if (argument < 1 || static_cast<size_t>(argument) > inputs.size()) {
        return makeError("invalid input argument in R1CS wire map");
      }
      auto *object = value ? value->getAsObject() : nullptr;
      value = object ? object->get(inputs[argument - 1].name) : nullptr;
    }
    for (auto segment : path.getValue().drop_front()) {
      if (auto name = dyn_cast<StringAttr>(segment)) {
        auto *object = value ? value->getAsObject() : nullptr;
        value = object ? object->get(name.getValue()) : nullptr;
      } else {
        auto index = cast<IntegerAttr>(segment).getInt();
        auto *array = value ? value->getAsArray() : nullptr;
        value = array && index >= 0 && static_cast<size_t>(index) < array->size() ? &(*array)[index]
                                                                                  : nullptr;
      }
    }
    if (!value) {
      return makeError("witness is missing a storage path required by R1CS");
    }
    auto text = value->getAsString();
    if (!text || text->empty() || !llvm::all_of(*text, llvm::isDigit)) {
      return makeError("R1CS witness leaf must be a non-negative field integer");
    }
    auto felt = toDynamicAPInt(*text);
    if (felt >= field.prime()) {
      return makeError("R1CS witness leaf is outside the field");
    }
    values.push_back(felt);
  }
  return values;
}

llvm::Expected<BinaryBuffer>
serializeWtns(ArrayRef<llvm::DynamicAPInt> witness, const Field &field) {
  if (witness.size() > std::numeric_limits<uint32_t>::max()) {
    return makeError("witness length does not fit in the .wtns header");
  }
  uint32_t fieldSize = ((field.bitWidth() + WTNS_FIELD_LIMB_BITS - 1) / WTNS_FIELD_LIMB_BITS) *
                       WTNS_FIELD_LIMB_BYTES;

  BinaryBuffer header;
  header.writeU32(fieldSize);
  header.writeFieldElement(fieldSize, field.prime());
  header.writeU32(static_cast<uint32_t>(witness.size()));

  BinaryBuffer values;
  for (const llvm::DynamicAPInt &value : witness) {
    values.writeFieldElement(fieldSize, value);
  }

  BinaryBuffer file;
  file.writeBytes(WTNS_MAGIC);
  file.writeU32(WTNS_VERSION);
  file.writeU32(WTNS_SECTION_COUNT);
  appendSection(file, WTNS_HEADER_SECTION, header);
  appendSection(file, WTNS_VALUES_SECTION, values);
  return file;
}

llvm::Error writeBinaryFile(StringRef outputFilename, const BinaryBuffer &file) {
  std::unique_ptr<llvm::ToolOutputFile> output = openOutputFile(outputFilename);
  if (!output) {
    return makeError(llvm::Twine("failed to open .wtns output: ") + outputFilename);
  }
  output->os().write(file.bytes().data(), file.bytes().size());
  output->os().flush();
  if (output->os().has_error()) {
    return makeError(llvm::Twine("failed to write .wtns output: ") + outputFilename);
  }
  output->keep();
  return llvm::Error::success();
}

} // namespace

llvm::Error writeWtns(
    ModuleOp moduleOp, const llvm::json::Value &fullWitness, const Field &field,
    StringRef outputFilename
) {
  auto loweredModule = lowerModuleToR1CS(moduleOp);
  if (!loweredModule) {
    return loweredModule.takeError();
  }
  bool evaluated = polymorphic::isEvaluatedModule(moduleOp);
  auto witness = evaluated ? collectEvaluatedWitness(moduleOp, **loweredModule, fullWitness, field)
                           : collectWitnessValues(moduleOp, fullWitness, field);
  if (!witness) {
    return witness.takeError();
  }
  auto mainName = [&loweredModule, moduleOp, evaluated]() -> llvm::Expected<std::string> {
    if (!evaluated) {
      return getMainStructName(moduleOp);
    }
    auto circuit = (**loweredModule)->getAttrOfType<SymbolRefAttr>(r1cs::CIRCUIT_REF_ATTR_NAME);
    if (!circuit) {
      return makeError("evaluated R1CS module is missing its circuit reference");
    }
    return circuit.getRootReference().str();
  }();
  if (!mainName) {
    return mainName.takeError();
  }
  auto r1csWireCount = getR1CSWireCount(**loweredModule, *mainName);
  if (!r1csWireCount) {
    return r1csWireCount.takeError();
  }
  // This guards witness length (including synthesized wires). The ordering
  // contract itself is documented and tested separately for each wire class.
  if (*r1csWireCount != witness->size()) {
    return makeError(
        llvm::Twine("cannot emit .wtns: R1CS lowering produces ") + llvm::Twine(*r1csWireCount) +
        " wires but llzk-witgen collected " + llvm::Twine(witness->size()) +
        "; synthesized R1CS auxiliary wires are not yet supported"
    );
  }

  auto file = serializeWtns(*witness, field);
  if (!file) {
    return file.takeError();
  }

  return writeBinaryFile(outputFilename, *file);
}

} // namespace llzk::witgen
