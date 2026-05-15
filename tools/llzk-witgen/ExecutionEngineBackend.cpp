//===-- ExecutionEngineBackend.cpp - llzk-witgen JIT backend ----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "ExecutionEngineBackend.h"

#include "Errors.h"
#include "JSON.h"
#include "ValueModel.h"
#include "WitnessSelection.h"
#include "WitgenLowering.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/UBToLLVM/UBToLLVM.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/APInt.h>
#include <llvm/Support/Endian.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdint>

using namespace mlir;

namespace llzk::witgen {

namespace {

/// Distinguish parsed interpreter/runtime values from MLIR SSA values in this file.
using RuntimeValue = llzk::witgen::Value;

/// Hold one raw memref descriptor and its backing storage bytes.
struct BufferPack {
  Type originalType;
  unsigned feltBitWidth = 0;
  size_t elemBytes = 0;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  std::vector<uint8_t> descriptor;
  std::vector<uint8_t> storage;
};

/// Return the byte width needed to store one `iN` field element.
static size_t getElementBytes(unsigned bitWidth) { return (bitWidth + 7U) / 8U; }

/// Return the row-major element count for a shaped felt aggregate.
static size_t getNumElements(ArrayRef<int64_t> shape) {
  size_t count = 1;
  for (int64_t dim : shape) {
    count *= static_cast<size_t>(dim);
  }
  return count;
}

/// Return the static shape of a main-boundary felt or felt-array type.
static llvm::Expected<std::vector<int64_t>> getBoundaryShape(Type type) {
  if (isa<felt::FeltType>(type)) {
    return std::vector<int64_t> {1};
  }
  if (auto arrayType = dyn_cast<array::ArrayType>(type)) {
    if (!isa<felt::FeltType>(arrayType.getElementType())) {
      return makeError("execution-engine backend only supports arrays of felt values at the main boundary");
    }
    return std::vector<int64_t>(arrayType.getShape().begin(), arrayType.getShape().end());
  }
  return makeError("execution-engine backend only supports felt and array<...xfelt> main boundaries");
}

/// Build row-major strides for one shaped buffer.
static std::vector<int64_t> computeStrides(ArrayRef<int64_t> shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

/// Populate the raw memref descriptor for a host buffer.
static void buildDescriptor(BufferPack &buffer) {
  const size_t rank = buffer.shape.size();
  buffer.descriptor.resize(sizeof(void *) * 2 + sizeof(int64_t) * (1 + rank + rank));
  uint8_t *cursor = buffer.descriptor.data();
  void *base = buffer.storage.data();
  std::memcpy(cursor, &base, sizeof(void *));
  cursor += sizeof(void *);
  std::memcpy(cursor, &base, sizeof(void *));
  cursor += sizeof(void *);
  const int64_t offset = 0;
  std::memcpy(cursor, &offset, sizeof(int64_t));
  cursor += sizeof(int64_t);
  for (int64_t size : buffer.shape) {
    std::memcpy(cursor, &size, sizeof(int64_t));
    cursor += sizeof(int64_t);
  }
  for (int64_t stride : buffer.strides) {
    std::memcpy(cursor, &stride, sizeof(int64_t));
    cursor += sizeof(int64_t);
  }
}

/// Create one host buffer matching the C-interface ABI for a memref boundary.
static llvm::Expected<BufferPack> createBufferPack(Type type, const Field &field) {
  auto shape = getBoundaryShape(type);
  if (!shape) {
    return shape.takeError();
  }
  BufferPack buffer;
  buffer.originalType = type;
  buffer.feltBitWidth = field.bitWidth();
  buffer.elemBytes = getElementBytes(buffer.feltBitWidth);
  buffer.shape = std::move(*shape);
  buffer.strides = computeStrides(buffer.shape);
  buffer.storage.resize(getNumElements(buffer.shape) * buffer.elemBytes);
  buildDescriptor(buffer);
  return buffer;
}

/// Store one field element into the buffer at the given flat element index.
static void storeElement(BufferPack &buffer, size_t flatIndex, const llvm::DynamicAPInt &value) {
  llvm::APInt raw = toAPInt(value, buffer.feltBitWidth);
  llvm::StoreIntToMemory(
      raw, buffer.storage.data() + flatIndex * buffer.elemBytes,
      static_cast<unsigned>(buffer.elemBytes)
  );
}

/// Load one field element from the buffer at the given flat element index.
static llvm::DynamicAPInt loadElement(const BufferPack &buffer, size_t flatIndex) {
  llvm::APInt raw(buffer.feltBitWidth, 0);
  llvm::LoadIntFromMemory(
      raw, buffer.storage.data() + flatIndex * buffer.elemBytes,
      static_cast<unsigned>(buffer.elemBytes)
  );
  return toDynamicAPInt(raw);
}

/// Marshal a parsed JSON runtime value into the raw storage for one host buffer.
static llvm::Error fillInputBuffer(BufferPack &buffer, const RuntimeValue &value) {
  if (isa<felt::FeltType>(buffer.originalType)) {
    auto feltValue = asFelt(value);
    if (!feltValue) {
      return feltValue.takeError();
    }
    storeElement(buffer, 0, *feltValue);
    return llvm::Error::success();
  }

  auto arrayValue = asArray(value);
  if (!arrayValue) {
    return arrayValue.takeError();
  }
  if ((*arrayValue)->elements.size() != getNumElements(buffer.shape)) {
    return makeError("input array element count mismatch");
  }
  for (size_t i = 0; i < (*arrayValue)->elements.size(); ++i) {
    auto feltValue = asFelt((*arrayValue)->elements[i]);
    if (!feltValue) {
      return feltValue.takeError();
    }
    storeElement(buffer, i, *feltValue);
  }
  return llvm::Error::success();
}

/// Render one field element buffer entry as the stable JSON decimal string form.
static llvm::json::Value feltElementToJSON(const BufferPack &buffer, size_t flatIndex) {
  std::string rendered;
  llvm::raw_string_ostream os(rendered);
  os << loadElement(buffer, flatIndex);
  return llvm::json::Value(os.str());
}

/// Recursively serialize a felt buffer into nested JSON arrays.
static llvm::json::Value
bufferToJSONArray(const BufferPack &buffer, size_t dimIndex, size_t flatOffset) {
  if (dimIndex + 1 == buffer.shape.size()) {
    llvm::json::Array result;
    for (int64_t i = 0; i < buffer.shape[dimIndex]; ++i) {
      result.push_back(feltElementToJSON(buffer, flatOffset + static_cast<size_t>(i)));
    }
    return llvm::json::Value(std::move(result));
  }

  size_t subArraySize = 1;
  for (size_t i = dimIndex + 1; i < buffer.shape.size(); ++i) {
    subArraySize *= static_cast<size_t>(buffer.shape[i]);
  }

  llvm::json::Array result;
  for (int64_t i = 0; i < buffer.shape[dimIndex]; ++i) {
    result.push_back(
        bufferToJSONArray(
            buffer, dimIndex + 1, flatOffset + static_cast<size_t>(i) * subArraySize
        )
    );
  }
  return llvm::json::Value(std::move(result));
}

/// Serialize one public output buffer into the user-facing JSON value.
static llvm::json::Value bufferToJSON(const BufferPack &buffer) {
  if (isa<felt::FeltType>(buffer.originalType)) {
    return feltElementToJSON(buffer, 0);
  }
  return bufferToJSONArray(buffer, 0, 0);
}

/// Lower the module through the shared LLZK-to-core witgen passes.
static llvm::Expected<OwningOpRef<ModuleOp>>
buildExecutionEngineModule(ModuleOp moduleOp, OutputScope outputScope) {
  OwningOpRef<ModuleOp> cloned = cast<ModuleOp>(moduleOp->clone());
  PassManager pm(cloned->getContext());
  pm.addPass(createLowerComputeToCorePass());
  pm.addPass(createCreateWitgenEntryPass(outputScope == OutputScope::FullWitness));
  if (failed(pm.run(*cloned))) {
    return makeError("failed to lower LLZK compute IR to execution-engine core dialects");
  }
  return cloned;
}

/// Run the standard cleanup and LLVM conversion pipeline on the lowered module.
static llvm::Error finalizeExecutionEngineModule(ModuleOp moduleOp) {
  PassManager pm(moduleOp.getContext());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (failed(pm.run(moduleOp))) {
    return makeError("failed to lower execution-engine module to LLVM dialect");
  }
  return llvm::Error::success();
}

/// Print the module when the corresponding debug flag is enabled.
static void maybeDumpModule(ModuleOp moduleOp, bool enabled, llvm::StringRef title) {
  if (!enabled) {
    return;
  }
  llvm::errs() << title << ":\n";
  moduleOp.print(llvm::errs());
  llvm::errs() << "\n";
}

} // namespace

/// Execute witness generation through MLIR lowering and the LLVM execution engine.
llvm::Expected<llvm::json::Value> runWithExecutionEngine(
    ModuleOp moduleOp, SymbolTableCollection &tables, const Field &field,
    const llvm::json::Value &input, const WitgenOptions &options
) {
  auto mainDef = getMainInstanceDef(tables, moduleOp.getOperation());
  if (failed(mainDef) || !mainDef.value()) {
    return makeError("module is missing a concrete llzk.main struct");
  }
  function::FuncDefOp computeFunc = mainDef->get().getComputeFuncOp();
  if (!computeFunc) {
    return makeError("main struct is missing @compute");
  }

  auto parsedArgs = [&]() -> llvm::Expected<llvm::SmallVector<RuntimeValue>> {
    llvm::SmallVector<RuntimeValue> args;
    auto *jsonObject = input.getAsObject();
    auto *jsonArray = input.getAsArray();
    if (!jsonObject && !jsonArray) {
      return makeError("inputs JSON must be either an object or an array");
    }
    if (jsonObject) {
      for (unsigned i = 0; i < computeFunc.getNumArguments(); ++i) {
        std::optional<StringAttr> argName = computeFunc.getArgNameAttr(i);
        if (!argName) {
          return makeError("JSON object input requires function.arg_name on every main argument");
        }
        const llvm::json::Value *value = jsonObject->get(argName->getValue());
        if (!value) {
          return makeError(llvm::Twine("missing JSON input field: ") + argName->getValue());
        }
        auto parsed =
            parseJSONValue(value, computeFunc.getArgumentTypes()[i], field, computeFunc.getOperation());
        if (!parsed) {
          return parsed.takeError();
        }
        args.push_back(*parsed);
      }
      return args;
    }
    if (jsonArray->size() != computeFunc.getNumArguments()) {
      return makeError("JSON positional input length does not match main compute arity");
    }
    for (unsigned i = 0; i < computeFunc.getNumArguments(); ++i) {
      auto parsed = parseJSONValue(
          &(*jsonArray)[i], computeFunc.getArgumentTypes()[i], field, computeFunc.getOperation()
      );
      if (!parsed) {
        return parsed.takeError();
      }
      args.push_back(*parsed);
    }
    return args;
  }();
  if (!parsedArgs) {
    return parsedArgs.takeError();
  }

  auto inputBindings = collectInputBindings(computeFunc);
  auto outputs =
      collectOutputBindings(mainDef->get(), tables, computeFunc.getOperation(), options.outputScope);
  if (failed(outputs)) {
    return makeError("failed to select witness outputs for execution-engine mode");
  }

  llvm::SmallVector<BufferPack> inputBuffers;
  for (auto [argType, parsed] : llvm::zip(computeFunc.getArgumentTypes(), *parsedArgs)) {
    auto buffer = createBufferPack(argType, field);
    if (!buffer) {
      return buffer.takeError();
    }
    if (auto err = fillInputBuffer(*buffer, parsed)) {
      return std::move(err);
    }
    inputBuffers.push_back(std::move(*buffer));
  }

  llvm::SmallVector<BufferPack> outputBuffers;
  for (const OutputBinding &output : *outputs) {
    auto buffer = createBufferPack(output.type, field);
    if (!buffer) {
      return buffer.takeError();
    }
    outputBuffers.push_back(std::move(*buffer));
  }

  auto loweredModule = buildExecutionEngineModule(moduleOp, options.outputScope);
  if (!loweredModule) {
    return loweredModule.takeError();
  }
  {
    DialectRegistry registry;
    mlir::arith::registerConvertArithToLLVMInterface(registry);
    mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
    mlir::registerConvertFuncToLLVMInterface(registry);
    mlir::index::registerConvertIndexToLLVMInterface(registry);
    mlir::registerConvertMemRefToLLVMInterface(registry);
    mlir::ub::registerConvertUBToLLVMInterface(registry);
    (*loweredModule)->getContext()->appendDialectRegistry(registry);
    (*loweredModule)->getContext()->loadAllAvailableDialects();
  }
  maybeDumpModule(**loweredModule, options.dumpJITCore, "llzk-witgen JIT core");
  if (auto err = finalizeExecutionEngineModule(**loweredModule)) {
    return std::move(err);
  }
  maybeDumpModule(**loweredModule, options.dumpJITLLVM, "llzk-witgen JIT LLVM");

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::registerBuiltinDialectTranslation(*(*loweredModule)->getContext());
  mlir::registerLLVMDialectTranslation(*(*loweredModule)->getContext());

  auto maybeEngine = mlir::ExecutionEngine::create(loweredModule->get());
  if (!maybeEngine) {
    return maybeEngine.takeError();
  }
  (*maybeEngine)->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;
    symbolMap[interner("memrefCopy")] = {
        llvm::orc::ExecutorAddr::fromPtr(&memrefCopy),
        llvm::JITSymbolFlags::Exported};
    return symbolMap;
  });

  llvm::SmallVector<void *> descriptorPtrs;
  descriptorPtrs.reserve(inputBuffers.size() + outputBuffers.size());
  for (BufferPack &buffer : inputBuffers) {
    descriptorPtrs.push_back(buffer.descriptor.data());
  }
  for (BufferPack &buffer : outputBuffers) {
    descriptorPtrs.push_back(buffer.descriptor.data());
  }

  llvm::SmallVector<void *> packedArgs;
  packedArgs.reserve(descriptorPtrs.size());
  for (void *&descriptorPtr : descriptorPtrs) {
    packedArgs.push_back(&descriptorPtr);
  }

  if (auto err = (*maybeEngine)->invokePacked("_mlir_ciface___llzk_witgen_main", packedArgs)) {
    return std::move(err);
  }

  llvm::SmallVector<llvm::json::Value> serializedOutputs;
  serializedOutputs.reserve(outputBuffers.size());
  for (const BufferPack &buffer : outputBuffers) {
    serializedOutputs.push_back(bufferToJSON(buffer));
  }

  if (options.outputScope == OutputScope::Public) {
    return buildSignalsJSONObject(*outputs, serializedOutputs);
  }

  auto inputsJSON = buildInputsJSONObject(inputBindings, *parsedArgs, tables, computeFunc.getOperation());
  if (!inputsJSON) {
    return inputsJSON.takeError();
  }
  llvm::json::Object result;
  result["inputs"] = llvm::json::Value(std::move(*inputsJSON));
  result["signals"] = buildSignalsJSONObject(*outputs, serializedOutputs);
  return llvm::json::Value(std::move(result));
}

} // namespace llzk::witgen
