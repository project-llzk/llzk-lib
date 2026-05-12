//===-- WitgenDriver.cpp - llzk-witgen driver entrypoints ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "WitgenDriver.h"

#include "Errors.h"
#include "Interpreter.h"
#include "JSON.h"

#include "llzk/Dialect/Include/Transforms/InlineIncludesPass.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/Pass/PassManager.h>

#include <llvm/ADT/SmallVector.h>

using namespace mlir;

namespace llzk::witgen {

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

/// Execute the concrete `llzk.main` compute function and serialize its public outputs.
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
  return serializeJSONValue(
      results->front(), computeFunc.getResultTypes().front(), tables, computeFunc.getOperation()
  );
}

/// Run include preprocessing, field validation, and main compute execution.
llvm::Expected<llvm::json::Value>
runWitgen(ModuleOp moduleOp, const llvm::json::Value &input, bool inlineIncludes) {
  if (inlineIncludes) {
    PassManager pm(moduleOp.getContext());
    pm.addPass(llzk::include::createInlineIncludesPass());
    if (failed(pm.run(moduleOp))) {
      return makeError("failed to inline includes");
    }
  }

  FieldSet fields;
  if (failed(collectFields(moduleOp.getOperation(), fields))) {
    return makeError("failed to collect fields for llzk-witgen");
  }
  if (fields.size() != 1) {
    return makeError("llzk-witgen v1 requires exactly one field in the module");
  }

  SymbolTableCollection tables;
  Interpreter interpreter(moduleOp, tables, *fields.begin());
  return interpreter.runMainFromJSON(input);
}

} // namespace llzk::witgen
