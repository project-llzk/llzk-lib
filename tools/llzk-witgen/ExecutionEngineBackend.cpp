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
#include "WitgenLowering.h"
#include "WitgenUtils.h"
#include "WitnessSelection.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Util/Compare.h"
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
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/APInt.h>
#include <llvm/Support/Endian.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdint>

using namespace mlir;

namespace llzk::witgen {

namespace {

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

/// Return the static shape of a main-boundary felt or felt-array type.
static llvm::Expected<std::vector<int64_t>> getBoundaryShape(Type type) {
  if (isa<felt::FeltType>(type)) {
    return std::vector<int64_t> {1};
  }
  if (auto arrayType = dyn_cast<array::ArrayType>(type)) {
    if (!isa<felt::FeltType>(arrayType.getElementType())) {
      return makeError(
          "execution-engine backend only supports arrays of felt values at the main boundary"
      );
    }
    if (!arrayType.hasStaticShape()) {
      return makeError(
          "execution-engine backend only supports statically shaped arrays at the main boundary"
      );
    }
    return std::vector<int64_t>(arrayType.getShape().begin(), arrayType.getShape().end());
  }
  return makeError(
      "execution-engine backend only supports felt and array<...xfelt> main boundaries"
  );
}

/// Build row-major strides for one shaped buffer.
static llvm::Expected<std::vector<int64_t>> computeStaticStrides(ArrayRef<int64_t> shape) {
  for (int64_t dim : shape) {
    auto checkedDim = checkedShapeDimToSize(dim, "execution-engine buffer shape");
    if (!checkedDim) {
      return checkedDim.takeError();
    }
    (void)*checkedDim;
  }
  auto strides = mlir::computeStrides(shape);
  return std::vector<int64_t>(strides.begin(), strides.end());
}

/// Populate the raw memref descriptor for a host buffer.
static llvm::Error buildDescriptor(BufferPack &buffer) {
  auto rank = checkedCast<int64_t>(buffer.shape.size());
  if (!rank) {
    return rank.takeError();
  }
  auto descriptorSize = llvm::DynamicAPInt(sizeof(void *)) * 2;
  auto shapeAndStrideCount = llvm::DynamicAPInt(1) + *rank + *rank;
  auto dynamicPart = llvm::DynamicAPInt(sizeof(int64_t)) * shapeAndStrideCount;
  auto totalSize = descriptorSize + dynamicPart;
  auto checkedTotalSize =
      checkedDynamicAPIntToSize(totalSize, "execution-engine memref descriptor");
  if (!checkedTotalSize) {
    return checkedTotalSize.takeError();
  }
  buffer.descriptor.resize(*checkedTotalSize);
  uint8_t *cursor = buffer.descriptor.data();
  uint8_t *base = buffer.storage.data();
  std::memcpy(cursor, static_cast<const void *>(&base), sizeof(void *));
  cursor += sizeof(void *);
  std::memcpy(cursor, static_cast<const void *>(&base), sizeof(void *));
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
  return llvm::Error::success();
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
  auto strides = computeStaticStrides(buffer.shape);
  if (!strides) {
    return strides.takeError();
  }
  buffer.strides = std::move(*strides);
  auto elementCount = getStaticShapeElementCount(buffer.shape, "execution-engine buffer storage");
  if (!elementCount) {
    return elementCount.takeError();
  }
  bool overflow = false;
  size_t storageBytes = llvm::SaturatingMultiply(*elementCount, buffer.elemBytes, &overflow);
  if (overflow) {
    return makeError("execution-engine buffer storage would overflow size_t");
  }
  buffer.storage.resize(storageBytes);
  if (auto error = buildDescriptor(buffer)) {
    return std::move(error);
  }
  return buffer;
}

/// Store one field element into the buffer at the given flat element index.
static llvm::Error
storeElement(BufferPack &buffer, size_t flatIndex, const llvm::DynamicAPInt &value) {
  bool overflow = false;
  size_t byteOffset = llvm::SaturatingMultiply(flatIndex, buffer.elemBytes, &overflow);
  if (overflow) {
    return makeError("execution-engine buffer store offset would overflow size_t");
  }
  auto elemBytesU = checkedCast<unsigned>(buffer.elemBytes);
  if (!elemBytesU) {
    return elemBytesU.takeError();
  }
  llvm::APInt raw = toAPInt(value, buffer.feltBitWidth);
  llvm::StoreIntToMemory(raw, buffer.storage.data() + byteOffset, *elemBytesU);
  return llvm::Error::success();
}

/// Load one field element from the buffer at the given flat element index.
static llvm::Expected<llvm::DynamicAPInt> loadElement(const BufferPack &buffer, size_t flatIndex) {
  bool overflow = false;
  size_t byteOffset = llvm::SaturatingMultiply(flatIndex, buffer.elemBytes, &overflow);
  if (overflow) {
    return makeError("execution-engine buffer load offset would overflow size_t");
  }
  auto elemBytesU = checkedCast<unsigned>(buffer.elemBytes);
  if (!elemBytesU) {
    return elemBytesU.takeError();
  }
  llvm::APInt raw(buffer.feltBitWidth, 0);
  llvm::LoadIntFromMemory(raw, buffer.storage.data() + byteOffset, *elemBytesU);
  return toDynamicAPInt(raw);
}

/// Marshal a parsed JSON runtime value into the raw storage for one host buffer.
static llvm::Error fillInputBuffer(BufferPack &buffer, const WitnessVal &value) {
  if (isa<felt::FeltType>(buffer.originalType)) {
    auto feltValue = asFelt(value);
    if (!feltValue) {
      return feltValue.takeError();
    }
    return storeElement(buffer, 0, *feltValue);
  }

  auto arrayValue = asArray(value);
  if (!arrayValue) {
    return arrayValue.takeError();
  }
  auto elementCount = getStaticShapeElementCount(buffer.shape, "execution-engine input array");
  if (!elementCount) {
    return elementCount.takeError();
  }
  if ((*arrayValue)->elements.size() != *elementCount) {
    return makeError("input array element count mismatch");
  }
  for (size_t i = 0; i < (*arrayValue)->elements.size(); ++i) {
    auto feltValue = asFelt((*arrayValue)->elements[i]);
    if (!feltValue) {
      return feltValue.takeError();
    }
    if (auto err = storeElement(buffer, i, *feltValue)) {
      return err;
    }
  }
  return llvm::Error::success();
}

/// Render one field element buffer entry as the stable JSON decimal string form.
static llvm::Expected<llvm::json::Value>
feltElementToJSON(const BufferPack &buffer, size_t flatIndex) {
  auto element = loadElement(buffer, flatIndex);
  if (!element) {
    return element.takeError();
  }
  std::string rendered;
  llvm::raw_string_ostream(rendered) << *element;
  return llvm::json::Value(rendered);
}

/// Recursively serialize a felt buffer into nested JSON arrays.
static llvm::Expected<llvm::json::Value>
bufferToJSONArray(const BufferPack &buffer, size_t dimIndex, size_t flatOffset) {
  if (dimIndex == SIZE_MAX) {
    return makeError("execution-engine JSON output would overflow size_t");
  }
  auto dimSize = checkedShapeDimToSize(buffer.shape[dimIndex], "execution-engine JSON output");
  if (!dimSize) {
    return dimSize.takeError();
  }
  if (dimIndex + 1 == buffer.shape.size()) {
    llvm::json::Array result;
    for (size_t i = 0; i < *dimSize; ++i) {
      bool overflow = false;
      size_t elementOffset = llvm::SaturatingAdd(i, flatOffset, &overflow);
      if (overflow) {
        return makeError("execution-engine JSON output would overflow size_t");
      }
      auto element = feltElementToJSON(buffer, elementOffset);
      if (!element) {
        return element.takeError();
      }
      result.push_back(*element);
    }
    return llvm::json::Value(std::move(result));
  }

  auto subArraySize = getStaticShapeElementCount(
      llvm::ArrayRef<int64_t>(buffer.shape).drop_front(dimIndex + 1), "execution-engine JSON output"
  );
  if (!subArraySize) {
    return subArraySize.takeError();
  }

  llvm::json::Array result;
  for (size_t i = 0; i < *dimSize; ++i) {
    bool overflow = false;
    size_t nextOffset = llvm::SaturatingMultiplyAdd(i, *subArraySize, flatOffset, &overflow);
    if (overflow) {
      return makeError("execution-engine JSON output would overflow size_t");
    }
    auto subArray = bufferToJSONArray(buffer, dimIndex + 1, nextOffset);
    if (!subArray) {
      return subArray.takeError();
    }
    result.push_back(*subArray);
  }
  return llvm::json::Value(std::move(result));
}

/// Serialize one public output buffer into the user-facing JSON value.
static llvm::Expected<llvm::json::Value> bufferToJSON(const BufferPack &buffer) {
  if (isa<felt::FeltType>(buffer.originalType)) {
    return feltElementToJSON(buffer, 0);
  }
  return bufferToJSONArray(buffer, 0, 0);
}

/// Lower the module through the shared LLZK-to-core witgen passes.
static llvm::Expected<OwningOpRef<ModuleOp>> buildExecutionEngineModule(
    ModuleOp moduleOp, OutputScope outputScope, const WitgenOptions &options
) {
  OwningOpRef<ModuleOp> cloned = cast<ModuleOp>(moduleOp->clone());
  PassManager pm(cloned->getContext());
  pm.addPass(createLowerComputeToCorePass(options));
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
  pm.addPass(mlir::createConvertSCFToCFPass());
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
  llvm::errs() << '\n';
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

  if (options.uninitializedBehavior == UninitializedBehavior::Fail) {
    Interpreter interpreter(
        moduleOp, tables, field, options.uninitializedBehavior, makeDefaultValueRng(options)
    );
    interpreter.setOutputScope(options.outputScope);
    return interpreter.runMainFromJSON(input);
  }

  auto parsedArgs = [&]() -> llvm::Expected<llvm::SmallVector<WitnessVal>> {
    llvm::SmallVector<WitnessVal> args;
    const auto *jsonObject = input.getAsObject();
    const auto *jsonArray = input.getAsArray();
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
        auto parsed = parseJSONValue(
            value, computeFunc.getArgumentTypes()[i], field, computeFunc.getOperation()
        );
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
  auto outputs = collectOutputBindings(
      mainDef->get(), tables, computeFunc.getOperation(), options.outputScope
  );
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

  auto loweredModule = buildExecutionEngineModule(moduleOp, options.outputScope, options);
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
  (*maybeEngine)->registerSymbols([](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;
    symbolMap[interner("memrefCopy")] = {
        llvm::orc::ExecutorAddr::fromPtr(&memrefCopy), llvm::JITSymbolFlags::Exported
    };
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
    packedArgs.push_back(static_cast<void *>(&descriptorPtr));
  }

  if (auto err = (*maybeEngine)->invokePacked("_mlir_ciface___llzk_witgen_main", packedArgs)) {
    return std::move(err);
  }

  llvm::SmallVector<llvm::json::Value> serializedOutputs;
  serializedOutputs.reserve(outputBuffers.size());
  for (const BufferPack &buffer : outputBuffers) {
    auto serialized = bufferToJSON(buffer);
    if (!serialized) {
      return serialized.takeError();
    }
    serializedOutputs.push_back(*serialized);
  }

  if (options.outputScope == OutputScope::Public) {
    return buildSignalsJSONObject(*outputs, serializedOutputs);
  }

  auto inputsJSON =
      buildInputsJSONObject(inputBindings, *parsedArgs, tables, computeFunc.getOperation());
  if (!inputsJSON) {
    return inputsJSON.takeError();
  }
  llvm::json::Object result;
  result["inputs"] = llvm::json::Value(std::move(*inputsJSON));
  result["signals"] = buildSignalsJSONObject(*outputs, serializedOutputs);
  return llvm::json::Value(std::move(result));
}

} // namespace llzk::witgen
