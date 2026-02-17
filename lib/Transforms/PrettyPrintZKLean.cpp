//===-- PrettyPrintZKLean.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.h"
#include "llzk/Dialect/ZKExpr/IR/ZKExprOps.h"
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanOps.h"
#include "llzk/Transforms/ZKLeanPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

// Pretty-printing overview:
// - emitZKLeanModule collects structs via collectZKLeanStructDefs, prints them
//   with emitLeanStructsFromZKLean, and formats bodies with formatLeanStatement.
using namespace mlir;

namespace llzk::zklean {
#define GEN_PASS_DECL_PRETTYPRINTZKLEANPASS
#define GEN_PASS_DEF_PRETTYPRINTZKLEANPASS
#include "llzk/Transforms/ZKLeanPasses.h.inc"

namespace namespace_detail {

// Identify ops that belong to any ZK* dialect family.
// Used to filter non-ZK ops during pretty-printing.
static bool isZkDialect(Operation *op) {
  StringRef dialectNs = op->getName().getDialectNamespace();
  return dialectNs.starts_with("ZK") || dialectNs.starts_with("zk");
}

// Identify ZKBuilder ops to decide monadic bind syntax.
// Returns true for both ZKBuilder and zkbuilder dialect spellings.
static bool isBuilderOp(Operation *op) {
  StringRef dialectNs = op->getName().getDialectNamespace();
  return dialectNs == "ZKBuilder" || dialectNs == "zkbuilder";
}

// Build a Lean-friendly name for `llzk::function::FuncDefOp`.
// Flattens nested symbol references into a single identifier using "__".
[[maybe_unused]] static std::string
buildLeanFunctionName(llzk::function::FuncDefOp func) {
  std::string name;
  auto fq = func.getFullyQualifiedName(false);
  if (!fq)
    return func.getSymName().str();

  name = fq.getRootReference().str();
  for (SymbolRefAttr nested : fq.getNestedReferences()) {
    name.append("__");
    name.append(nested.getLeafReference().str());
  }
  return name;
}

// Extract a struct name from ZKLean struct types.
// Returns `std::nullopt` when the type is not a ZKLean struct.
static std::optional<std::string> getStructTypeName(Type type) {
  if (auto structType = dyn_cast<mlir::zkleanlean::StructType>(type))
    return structType.getNameRef().getLeafReference().str();
  return std::nullopt;
}

// Format an MLIR `Type` into a Lean-facing type string.
// Maps ZKExpr/ZKBuilder/felt types to Lean aliases and preserves unknown types.
static std::string formatLeanType(Type type) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  type.print(os);
  os.flush();

  StringRef typeStr(buffer);
  if (typeStr.starts_with("!ZKExpr"))
    return "ZKExpr f";
  if (typeStr.starts_with("!ZKBuilder"))
    return "ZKBuilder f PUnit";
  // Figure out how to handle ZKLean namespace, if necessary
  if (typeStr.starts_with("!ZKLean"))
    return "ZKLean f";
  if (typeStr.starts_with("!felt"))
    // ZKLean functions operate on ZKExpr values, not raw felts.
    return "ZKExpr f";

  return buffer;
}

// Lookup or assign a printable name for an SSA `Value`.
// Uses local-scope printing for unknown values and memoizes the result.
static std::string
lookupValueName(Value value, llvm::DenseMap<Value, std::string> &valueNames) {
  if (auto it = valueNames.find(value); it != valueNames.end())
    return it->second;

  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  OpPrintingFlags flags;
  flags.useLocalScope();
  value.printAsOperand(os, flags);
  os.flush();
  if (!buffer.empty() && buffer.front() == '%')
    buffer.erase(buffer.begin());
  valueNames[value] = buffer;
  return buffer;
}

// Wrap result names for Lean tuple syntax.
// Returns "_" for empty results and "(a, b)" for multiple names.
static std::string wrapResultNames(ArrayRef<std::string> names) {
  if (names.empty())
    return "_";
  if (names.size() == 1)
    return names.front();

  std::string joined = "(";
  for (auto it = names.begin(), e = names.end(); it != e; ++it) {
    if (it != names.begin())
      joined.append(", ");
    joined.append(*it);
  }
  joined.push_back(')');
  return joined;
}

// Strip dialect prefixes to match Lean surface syntax.
// This keeps output closer to the Lean API surface than raw MLIR names.
static std::string formatOperationName(Operation &op) {
  StringRef name = op.getName().getStringRef();
  if (name.consume_front("ZKExpr."))
    return name.str();
  if (name.consume_front("ZKBuilder."))
    return name.str();
  if (name.consume_front("zkexpr."))
    return name.str();
  if (name.consume_front("zkbuilder."))
    return name.str();
  return name.str();
}

// Flatten a symbol reference into a Lean-friendly call name.
// Nested references are joined with "__" for stability.
static std::string formatLeanCallee(SymbolRefAttr callee) {
  std::string name = callee.getRootReference().str();
  for (SymbolRefAttr nested : callee.getNestedReferences()) {
    name.append("__");
    name.append(nested.getLeafReference().str());
  }
  return name;
}

// Format an op as a Lean-style function call.
// Uses `formatOperationName` and `lookupValueName` for operands.
static std::string formatOperationCall(Operation &op,
                                       llvm::DenseMap<Value, std::string> &names) {
  std::string call = formatOperationName(op);
  for (Value operand : op.getOperands()) {
    call.push_back(' ');
    call.append(lookupValueName(operand, names));
  }
  return call;
}

// Render a felt constant as a decimal literal.
// Uses a base-10 string conversion for readability.
static std::string formatFeltConstant(llzk::felt::FeltConstantOp constOp) {
  auto value = constOp.getValueAttr().getValue();
  llvm::SmallString<32> buffer;
  value.toString(buffer, 10, false, false, false, false);
  return std::string(buffer);
}

// Assign vN names to SSA results for readable Lean output.
// Updates `valueNames` and advances `nextValueId` for later allocations.
static llvm::SmallVector<std::string, 4>
assignResultNames(Operation &op,
                  llvm::DenseMap<Value, std::string> &valueNames,
                  unsigned &nextValueId) {
  llvm::SmallVector<std::string, 4> resultNames;
  resultNames.reserve(op.getNumResults());
  for (Value result : op.getResults()) {
    std::string name = ("v" + llvm::Twine(nextValueId++)).str();
    valueNames[result] = name;
    resultNames.push_back(std::move(name));
  }
  return resultNames;
}

// Extract witness metadata used for struct field projection.
// Supports both `StringAttr` and `FlatSymbolRefAttr` forms.
static std::optional<std::string> getWitnessFieldName(Operation &op) {
  if (auto fieldAttr = op.getAttrOfType<StringAttr>("llzk.field"))
    return fieldAttr.getValue().str();
  if (auto fieldAttr = op.getAttrOfType<FlatSymbolRefAttr>("llzk.field"))
    return fieldAttr.getValue().str();
  return std::nullopt;
}

// Extract struct metadata used to select the right struct parameter.
// Accepts `SymbolRefAttr` or `StringAttr` under `llzk.struct`.
static std::optional<std::string> getWitnessStructName(Operation &op) {
  if (auto structAttr = op.getAttrOfType<SymbolRefAttr>("llzk.struct"))
    return structAttr.getLeafReference().str();
  if (auto structAttr = op.getAttrOfType<StringAttr>("llzk.struct"))
    return structAttr.getValue().str();
  return std::nullopt;
}

// Infer struct name from a common Struct__constrain naming convention.
// Only returns a name that appears in `structNames`.
static std::optional<std::string>
inferStructNameFromFuncName(StringRef funcName,
                            const llvm::DenseSet<StringRef> &structNames) {
  auto splitPos = funcName.find("__");
  if (splitPos == StringRef::npos)
    return std::nullopt;
  StringRef candidate = funcName.take_front(splitPos);
  if (structNames.contains(candidate))
    return candidate.str();
  return std::nullopt;
}

// Discover a struct name via witness metadata or the function name.
// Prefers explicit `llzk.struct` metadata from witnesses when available.
static std::optional<std::string>
findStructName(Block &entry, StringRef funcName,
               const llvm::DenseSet<StringRef> &structNames) {
  for (Operation &op : entry) {
    if (isa<mlir::zkexpr::WitnessOp>(op)) {
      if (auto name = getWitnessStructName(op))
        return name;
    }
  }
  return inferStructNameFromFuncName(funcName, structNames);
}

// Format ZKLean/ZKBuilder ops into Lean statements with special cases.
// Handles literals, witnesses, accessors, calls, arithmetic, and constrain ops
// before falling back to a generic op printing path.
static std::optional<std::string>
formatLeanStatement(Operation &op,
                    llvm::DenseMap<Value, std::string> &valueNames,
                    unsigned &nextValueId,
                    const llvm::StringMap<std::string> &structArgNames,
                    StringRef defaultStructVarName) {
  if (auto literal = dyn_cast<mlir::zkexpr::LiteralOp>(op)) {
    Value litValue = literal.getLiteral();
    if (auto arg = dyn_cast<BlockArgument>(litValue)) {
      std::string argName = lookupValueName(arg, valueNames);
      for (Value result : op.getResults())
        valueNames[result] = argName;
      nextValueId += op.getNumResults();
      return std::nullopt;
    }
    if (auto constOp = litValue.getDefiningOp<llzk::felt::FeltConstantOp>()) {
      auto resultNames = assignResultNames(op, valueNames, nextValueId);
      std::string line = "  let ";
      line.append(wrapResultNames(resultNames));
      line.append(" := ");
      line.append(formatFeltConstant(constOp));
      return line;
    }
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" := ");
    line.append(lookupValueName(litValue, valueNames));
    return line;
  }

  if (auto witness = dyn_cast<mlir::zkexpr::WitnessOp>(op)) {
    auto fieldName = getWitnessFieldName(op);
    std::string structVarName = defaultStructVarName.str();
    if (auto structName = getWitnessStructName(op)) {
      if (auto it = structArgNames.find(*structName);
          it != structArgNames.end()) {
        structVarName = it->second;
      }
    }
    if (fieldName && !structVarName.empty()) {
      auto resultNames = assignResultNames(op, valueNames, nextValueId);
      std::string line = "  let ";
      line.append(wrapResultNames(resultNames));
      line.append(" := ");
      line.append(structVarName);
      line.push_back('.');
      line.append(*fieldName);
      return line;
    }
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" <- Witnessable.witness");
    return line;
  }

  if (auto accessor = dyn_cast<mlir::zkleanlean::AccessorOp>(op)) {
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" := ");
    line.append(lookupValueName(accessor.getComponent(), valueNames));
    line.push_back('.');
    line.append(accessor.getMemberNameAttr().getValue());
    return line;
  }

  if (auto call = dyn_cast<mlir::zkleanlean::CallOp>(op)) {
    std::string callee = formatLeanCallee(call.getCallee());
    std::string callLine = callee;
    for (Value operand : op.getOperands()) {
      callLine.push_back(' ');
      callLine.append(lookupValueName(operand, valueNames));
    }
    if (op.getNumResults() == 0) {
      return std::string("  ") + callLine;
    }
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" <- ");
    line.append(callLine);
    return line;
  }

  if (auto neg = dyn_cast<mlir::zkexpr::NegOp>(op)) {
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" := -");
    line.append(lookupValueName(neg.getValue(), valueNames));
    return line;
  }

  if (auto add = dyn_cast<mlir::zkexpr::AddOp>(op)) {
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" := ");
    line.append(lookupValueName(add.getLhs(), valueNames));
    line.append(" + ");
    line.append(lookupValueName(add.getRhs(), valueNames));
    return line;
  }

  if (auto sub = dyn_cast<mlir::zkexpr::SubOp>(op)) {
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" := ");
    line.append(lookupValueName(sub.getLhs(), valueNames));
    line.append(" - ");
    line.append(lookupValueName(sub.getRhs(), valueNames));
    return line;
  }

  if (auto mul = dyn_cast<mlir::zkexpr::MulOp>(op)) {
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" := ");
    line.append(lookupValueName(mul.getLhs(), valueNames));
    line.append(" * ");
    line.append(lookupValueName(mul.getRhs(), valueNames));
    return line;
  }

  if (auto eq = dyn_cast<mlir::zkbuilder::ConstrainEqOp>(op)) {
    std::string call = "constrainEq ";
    call.append(lookupValueName(eq.getLhs(), valueNames));
    call.push_back(' ');
    call.append(lookupValueName(eq.getRhs(), valueNames));
    if (op.getResults().empty() || op.getResult(0).use_empty()) {
      nextValueId += op.getNumResults();
      return std::string("  ") + call;
    }
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" <- ");
    line.append(call);
    return line;
  }

  auto resultNames = assignResultNames(op, valueNames, nextValueId);
  std::string line = "  let ";
  line.append(wrapResultNames(resultNames));
  line.push_back(' ');
  line.append(isBuilderOp(&op) ? "<-" : ":=");
  line.push_back(' ');
  line.append(formatOperationCall(op, valueNames));
  return line;
}

// Emit Lean for ZKLean/ZKBuilder functions, with struct arg inference.
// Assigns argument names, infers struct parameters, and formats statements.
template <typename FuncOpTy>
static bool emitLeanFunc(FuncOpTy func, raw_ostream &os,
                         const llvm::DenseSet<StringRef> &structNames) {
  if (func.getBody().empty())
    return false;

  llvm::DenseMap<Value, std::string> valueNames;
  llvm::StringMap<std::string> structArgNames;
  Block &entry = func.front();
  std::string defaultStructVarName;
  unsigned structIndex = 0;
  unsigned nonStructIndex = 0;
  for (auto arg : entry.getArguments()) {
    if (auto structTypeName = getStructTypeName(arg.getType())) {
      std::string name = structIndex == 0
                             ? "strct"
                             : (llvm::Twine("strct") + llvm::Twine(structIndex))
                                   .str();
      valueNames[arg] = name;
      if (structArgNames.find(*structTypeName) == structArgNames.end())
        structArgNames[*structTypeName] = name;
      if (structIndex == 0)
        defaultStructVarName = name;
      ++structIndex;
      continue;
    }
    valueNames[arg] = ("arg" + llvm::Twine(nonStructIndex++)).str();
  }

  bool hasStructArgs = structIndex != 0;
  auto structName =
      hasStructArgs ? std::optional<std::string>()
                    : findStructName(entry, func.getSymName(), structNames);
  if (!hasStructArgs && structName)
    defaultStructVarName = "strct";

  unsigned nextValueId = 0;
  llvm::SmallVector<std::string, 16> statements;

  for (Operation &op : entry) {
    if (isa<func::ReturnOp>(op))
      continue;
    if (!isZkDialect(&op))
      continue;
    if (auto stmt = formatLeanStatement(op, valueNames, nextValueId,
                                        structArgNames, defaultStructVarName))
      statements.push_back(*stmt);
  }

  if (statements.empty())
    return false;

  os << "def " << func.getSymName() << " [Field f]";
  // Add a synthetic struct parameter if only witness metadata references it.
  if (!hasStructArgs && structName)
    os << " (" << defaultStructVarName << " : " << *structName << " f)";
  for (auto arg : entry.getArguments()) {
    auto argName = lookupValueName(arg, valueNames);
    if (auto structTypeName = getStructTypeName(arg.getType())) {
      os << " (" << argName << " : " << *structTypeName << " f)";
      continue;
    }
    os << " (" << argName << " : " << formatLeanType(arg.getType()) << ")";
  }
  os << " : ZKBuilder f PUnit := do\n";
  for (auto &stmt : statements)
    os << stmt << '\n';

  return true;
}

// Emit Lean structure declarations from ZKLean structs.
// Reads ZKLean struct bodies and formats fields in order.
static bool emitLeanStructsFromZKLean(
    ArrayRef<mlir::zkleanlean::StructDefOp> structs,
    raw_ostream &os) {
  bool printed = false;
  for (auto def : structs) {
    // All ZKLean structs are implicitly parameterized over the field type `f`.
    os << "structure " << def.getSymName() << " (f : Type) where\n";
    for (auto member :
         def.getBodyRegion().getOps<mlir::zkleanlean::MemberDefOp>()) {
      os << "  " << member.getSymName() << " : "
         << formatLeanType(member.getType()) << '\n';
    }
    os << '\n';
    printed = true;
  }
  return printed;
}

// Collect unique ZKLean struct defs and record their names.
// Preserves first-seen order for deterministic output.
static llvm::SmallVector<mlir::zkleanlean::StructDefOp, 8>
collectZKLeanStructDefs(ModuleOp module,
                        llvm::DenseSet<StringRef> &structNames) {
  llvm::SmallVector<mlir::zkleanlean::StructDefOp, 8> structDefs;
  module.walk([&](mlir::zkleanlean::StructDefOp def) {
    StringRef name = def.getSymName();
    if (structNames.insert(name).second)
      structDefs.push_back(def);
    return WalkResult::advance();
  });
  return structDefs;
}

// Emit a ZKLean function definition if it produces any statements.
// Adds a separating newline and updates `printedSomething` on success.
template <typename FuncOpTy>
static void emitZKLeanFuncIfAny(FuncOpTy func, raw_ostream &os,
                                const llvm::DenseSet<StringRef> &structNames,
                                bool &printedSomething) {
  if (!emitLeanFunc(func, os, structNames))
    return;
  printedSomething = true;
  os << '\n';
}

// Emit Lean output from modules that already contain ZKLean/ZKBuilder ops.
// Prints structs first, then functions using `emitLeanFunc`.
static bool emitZKLeanModule(ModuleOp module, raw_ostream &os) {
  bool printedSomething = false;
  llvm::DenseSet<StringRef> structNames;
  auto structDefs = collectZKLeanStructDefs(module, structNames);

  if (emitLeanStructsFromZKLean(structDefs, os))
    printedSomething = true;

  module.walk([&](func::FuncOp func) {
    emitZKLeanFuncIfAny(func, os, structNames, printedSomething);
    return WalkResult::advance();
  });

  return printedSomething;
}

} // namespace namespace_detail
namespace {

// Pass that pretty-prints ZK dialect IR into Lean-like syntax.
// Manages output file handling for pretty-printed Lean code.
struct PrettyPrintZKLeanPass
    : public impl::PrettyPrintZKLeanPassBase<PrettyPrintZKLeanPass> {
  // Run the pass and emit Lean output to the selected stream.
  // Uses ZKLean formatting when ZK ops are present.
  void runOnOperation() override {
    raw_ostream *stream = &llvm::outs();
    std::unique_ptr<llvm::ToolOutputFile> outputFile;
    if (!outputFilename.empty() && outputFilename != "-") {
      outputFile = openOutputFile(outputFilename);
      if (!outputFile) {
        signalPassFailure();
        return;
      }
      stream = &outputFile->os();
    }

    ModuleOp module = getOperation();
    *stream << "import zkLean\n";
    *stream << "open ZKBuilder\n\n";

    bool printedSomething = namespace_detail::emitZKLeanModule(module, *stream);

    if (!printedSomething)
      *stream << "-- No zk dialect operations found.\n";

    if (outputFile)
      outputFile->keep();
  }
};

} // namespace

// Pass factory for `PrettyPrintZKLeanPass`.
// Used by pass registration and external callers.
std::unique_ptr<Pass> createPrettyPrintZKLeanPass() {
  return std::make_unique<PrettyPrintZKLeanPass>();
}

} // namespace llzk::zklean
