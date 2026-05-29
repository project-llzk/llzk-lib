//===-- WitgenDriver.cpp - llzk-witgen driver entrypoints -------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "WitgenDriver.h"

#include "Errors.h"
#include "ExecutionEngineBackend.h"
#include "Interpreter.h"
#include "JSON.h"
#include "WitgenLowering.h"
#include "WitgenUtils.h"
#include "WitnessSelection.h"

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Include/Transforms/InlineIncludesPass.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/PassManager.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;

namespace llzk::witgen {

/// Return whether the module needs template/affine flattening before execution.
static bool requiresFlattening(ModuleOp moduleOp) {
  return moduleOp
      ->walk([&](Operation *op) {
    if (isa<function::CallOp>(op)) {
      auto callOp = cast<function::CallOp>(op);
      if (callOp.getTemplateParams() || !callOp.getMapOperands().empty()) {
        return WalkResult::interrupt();
      }
    }
    if (op->getDialect()->getNamespace() ==
        polymorphic::PolymorphicDialect::getDialectNamespace()) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  }).wasInterrupted();
}

/// Build a driver around one parsed module and field.
Interpreter::Interpreter(
    ModuleOp mod, SymbolTableCollection &symbolTables, const Field &moduleField,
    UninitializedBehavior behavior, std::mt19937_64 r
)
    : moduleOp(mod), tables(symbolTables), field(moduleField), uninitializedBehavior(behavior),
      rng(r) {}

static void warnPositionalObjectFallback(unsigned namedCount) {
  llvm::errs() << "warning: ";
  if (namedCount == 0) {
    llvm::errs() << "no function.arg_name attributes found on Main.compute arguments; "
                    "binding JSON object inputs by field order\n";
    return;
  }
  llvm::errs() << "partial function.arg_name coverage on Main.compute arguments; "
                  "ignoring JSON object keys and binding inputs by field order\n";
}

/// Parse main-function JSON arguments in either object or positional form.
llvm::Expected<llvm::SmallVector<WitnessVal>> parseMainArgumentsFromJSON(
    function::FuncDefOp computeFunc, const llvm::json::Value &input, const Field &field
) {
  llvm::SmallVector<WitnessVal> args;
  const auto *jsonObject = input.getAsObject();
  const auto *jsonArray = input.getAsArray();
  if (!jsonObject && !jsonArray) {
    return makeError("inputs JSON must be either an object or an array");
  }

  if (jsonObject) {
    unsigned namedCount = 0;
    for (unsigned i = 0; i < computeFunc.getNumArguments(); ++i) {
      if (computeFunc.getArgNameAttr(i)) {
        ++namedCount;
      }
    }

    if (namedCount == computeFunc.getNumArguments()) {
      for (unsigned i = 0; i < computeFunc.getNumArguments(); ++i) {
        llvm::StringRef argName = computeFunc.getArgNameAttr(i)->getValue();
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

    if (jsonObject->size() != computeFunc.getNumArguments()) {
      return makeError("JSON object input length does not match main compute arity");
    }
    warnPositionalObjectFallback(namedCount);

    unsigned i = 0;
    for (const auto &entry : *jsonObject) {
      auto parsed = parseJSONValue(
          &entry.second, computeFunc.getArgumentTypes()[i], field, computeFunc.getOperation()
      );
      if (!parsed) {
        return parsed.takeError();
      }
      args.push_back(*parsed);
      ++i;
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

  auto args = parseMainArgumentsFromJSON(computeFunc, input, field);
  if (!args) {
    return args.takeError();
  }

  FunctionInterpreter interpreter(moduleOp, tables, field, uninitializedBehavior, rng);
  auto results = interpreter.run(computeFunc, *args);
  if (!results) {
    return results.takeError();
  }
  if (results->size() != 1) {
    return makeError("main compute returned unexpected result count");
  }
  if (outputScope == OutputScope::Public) {
    return serializeJSONValue(
        results->front(), computeFunc.getResultTypes().front(), tables, computeFunc.getOperation(),
        SerializationMode::PublicOutputsOnly
    );
  }

  auto inputBindings = collectInputBindings(computeFunc);
  auto inputsJSON = buildInputsJSONObject(inputBindings, *args, tables, computeFunc.getOperation());
  if (!inputsJSON) {
    return inputsJSON.takeError();
  }

  auto outputBindings = collectOutputBindings(
      mainDef->get(), tables, computeFunc.getOperation(), OutputScope::FullWitness
  );
  if (failed(outputBindings)) {
    return makeError("failed to select full witness signals");
  }

  llvm::SmallVector<llvm::json::Value> serializedSignals;
  serializedSignals.reserve(outputBindings->size());
  for (const OutputBinding &binding : *outputBindings) {
    auto leafValue = extractValueAtPath(
        results->front(), computeFunc.getResultTypes().front(), binding.path, tables,
        computeFunc.getOperation()
    );
    if (!leafValue) {
      return leafValue.takeError();
    }
    auto serialized = serializeJSONValue(
        *leafValue, binding.type, tables, computeFunc.getOperation(), SerializationMode::AllSignals
    );
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
    addWitgenPreparePipeline(pm, options);
  } else if (requiresFlattening(moduleOp)) {
    pm.addPass(llzk::polymorphic::createFlatteningPass());
  }
  pm.printAsTextualPipeline(llvm::errs());
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
  Interpreter interpreter(
      moduleOp, tables, *fields.begin(), options.uninitializedBehavior, makeDefaultValueRng(options)
  );
  interpreter.setOutputScope(options.outputScope);
  return interpreter.runMainFromJSON(input);
}

} // namespace llzk::witgen
