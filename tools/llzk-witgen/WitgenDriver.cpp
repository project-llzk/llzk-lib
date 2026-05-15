//===-- WitgenDriver.cpp - llzk-witgen driver entrypoints -------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "WitgenDriver.h"

#include "ExecutionEngineBackend.h"
#include "Errors.h"
#include "Interpreter.h"
#include "JSON.h"
#include "WitnessSelection.h"
#include "WitgenLowering.h"

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Include/Transforms/InlineIncludesPass.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/PassManager.h>

#include <llvm/ADT/SmallVector.h>

using namespace mlir;

namespace llzk::witgen {

/// Return whether the module needs template/affine flattening before execution.
static bool requiresFlattening(ModuleOp moduleOp) {
  bool needsFlattening = false;
  moduleOp->walk([&](Operation *op) {
    if (isa<function::CallOp>(op)) {
      auto callOp = cast<function::CallOp>(op);
      if (callOp.getTemplateParams() || !callOp.getMapOperands().empty()) {
        needsFlattening = true;
        return WalkResult::interrupt();
      }
    }
    if (op->getName().getStringRef().starts_with("poly.")) {
      needsFlattening = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return needsFlattening;
}

/// Build a driver around one parsed module and field.
Interpreter::Interpreter(ModuleOp moduleOp, SymbolTableCollection &tables, const Field &field)
    : moduleOp(moduleOp), tables(tables), field(field) {}

/// Parse main-function JSON arguments in either object or positional form.
static llvm::Expected<llvm::SmallVector<Value>> parseArgumentsFromJSON(
    function::FuncDefOp computeFunc, const llvm::json::Value &input, const Field &field
) {
  llvm::SmallVector<Value> args;
  auto *jsonObject = input.getAsObject();
  auto *jsonArray = input.getAsArray();
  if (!jsonObject && !jsonArray) {
    return makeError("inputs JSON must be either an object or an array");
  }

  if (jsonObject) {
    for (unsigned i = 0; i < computeFunc.getNumArguments(); ++i) {
      llvm::StringRef argName;
      if (std::optional<StringAttr> attr = computeFunc.getArgNameAttr(i)) {
        argName = attr->getValue();
      } else {
        return makeError("JSON object input requires function.arg_name on every main argument");
      }
      const llvm::json::Value *value = jsonObject->get(argName);
      if (!value) {
        return makeError(llvm::Twine("missing JSON input field: ") + argName);
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
}

/// Execute the concrete `llzk.main` compute function and serialize the selected witness scope.
llvm::Expected<llvm::json::Value> Interpreter::runMainFromJSON(const llvm::json::Value &input) {
  auto mainDef = getMainInstanceDef(tables, moduleOp.getOperation());
  if (failed(mainDef) || !mainDef.value()) {
    return makeError("module is missing a concrete llzk.main struct");
  }

  auto computeFunc = mainDef->get().getComputeFuncOp();
  if (!computeFunc) {
    return makeError("main struct is missing @compute");
  }
  if (computeFunc.getNumResults() != 1) {
    return makeError("main compute must return exactly one value");
  }

  auto args = parseArgumentsFromJSON(computeFunc, input, field);
  if (!args) {
    return args.takeError();
  }

  FunctionInterpreter interpreter(moduleOp, tables, field);
  auto results = interpreter.run(computeFunc, *args);
  if (!results) {
    return results.takeError();
  }
  if (results->size() != 1) {
    return makeError("main compute returned unexpected result count");
  }
  if (outputScope == OutputScope::Public) {
    return serializeJSONValue(
        results->front(), computeFunc.getResultTypes().front(), tables,
        computeFunc.getOperation(), SerializationMode::PublicOutputsOnly
    );
  }

  auto inputBindings = collectInputBindings(computeFunc);
  auto inputsJSON = buildInputsJSONObject(inputBindings, *args, tables, computeFunc.getOperation());
  if (!inputsJSON) {
    return inputsJSON.takeError();
  }

  auto outputBindings =
      collectOutputBindings(mainDef->get(), tables, computeFunc.getOperation(), OutputScope::FullWitness);
  if (failed(outputBindings)) {
    return makeError("failed to select full witness signals");
  }

  llvm::SmallVector<llvm::json::Value> serializedSignals;
  serializedSignals.reserve(outputBindings->size());
  for (const OutputBinding &binding : *outputBindings) {
    auto leafValue = extractValueAtPath(
        results->front(), computeFunc.getResultTypes().front(), binding.path, tables,
        computeFunc.getOperation());
    if (!leafValue) {
      return leafValue.takeError();
    }
    auto serialized = serializeJSONValue(
        *leafValue, binding.type, tables, computeFunc.getOperation(),
        SerializationMode::AllSignals);
    if (!serialized) {
      return serialized.takeError();
    }
    serializedSignals.push_back(*serialized);
  }

  llvm::json::Object result;
  result["inputs"] = llvm::json::Value(std::move(*inputsJSON));
  result["signals"] = buildSignalsJSONObject(*outputBindings, serializedSignals);
  return llvm::json::Value(std::move(result));
}

/// Run include preprocessing and flattening before backend execution.
static llvm::Error preprocessModule(ModuleOp moduleOp, const WitgenOptions &options) {
  // normalizeCallOpProperties(moduleOp);
  PassManager pm(moduleOp.getContext());
  if (options.inlineIncludes) {
    pm.addPass(llzk::include::createInlineIncludesPass());
  }
  if (options.backend == Backend::ExecutionEngine) {
    addWitgenPreparePipeline(pm);
  } else if (requiresFlattening(moduleOp)) {
    pm.addPass(llzk::polymorphic::createFlatteningPass());
  }
  if (failed(pm.run(moduleOp))) {
    return makeError("failed to preprocess LLZK module for llzk-witgen");
  }
  return llvm::Error::success();
}

/// Run include preprocessing, field validation, and backend execution.
llvm::Expected<llvm::json::Value>
runWitgen(ModuleOp moduleOp, const llvm::json::Value &input, const WitgenOptions &options) {
  if (auto err = preprocessModule(moduleOp, options)) {
    return std::move(err);
  }

  FieldSet fields;
  if (failed(collectFields(moduleOp.getOperation(), fields))) {
    return makeError("failed to collect fields for llzk-witgen");
  }
  if (fields.size() != 1) {
    return makeError("llzk-witgen v1 requires exactly one field in the module");
  }

  SymbolTableCollection tables;
  if (options.backend == Backend::ExecutionEngine) {
    return runWithExecutionEngine(moduleOp, tables, *fields.begin(), input, options);
  }
  Interpreter interpreter(moduleOp, tables, *fields.begin());
  interpreter.setOutputScope(options.outputScope);
  return interpreter.runMainFromJSON(input);
}

} // namespace llzk::witgen
