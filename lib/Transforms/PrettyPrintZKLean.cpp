#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.h"
#include "llzk/Dialect/ZKExpr/IR/ZKExprOps.h"
#include "llzk/Dialect/ZKLeanStruct/IR/ZKLeanStructOps.h"
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

using namespace mlir;

namespace llzk::zklean {
namespace namespace_detail {

static bool isZkDialect(Operation *op) {
  StringRef dialectNs = op->getName().getDialectNamespace();
  return dialectNs.starts_with("ZK") || dialectNs.starts_with("zk");
}

static bool isBuilderOp(Operation *op) {
  StringRef dialectNs = op->getName().getDialectNamespace();
  return dialectNs == "ZKBuilder" || dialectNs == "zkbuilder";
}

// Build a Lean-friendly function name from LLZK nested symbol paths.
static std::string buildLeanFunctionName(llzk::function::FuncDefOp func) {
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

// Extract a struct name from either ZKLean or LLZK struct types.
static std::optional<std::string> getStructTypeName(Type type) {
  if (auto structType = dyn_cast<mlir::zkleanstruct::StructType>(type))
    return structType.getNameRef().getLeafReference().str();
  if (auto structType = dyn_cast<llzk::component::StructType>(type))
    return structType.getNameRef().getLeafReference().str();
  return std::nullopt;
}

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
static std::string formatFeltConstant(llzk::felt::FeltConstantOp constOp) {
  auto value = constOp.getValueAttr().getValue();
  llvm::SmallString<32> buffer;
  value.toString(buffer, 10, false, false, false, false);
  return std::string(buffer);
}

// Assign vN names to SSA results for readable Lean output.
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
static std::optional<std::string> getWitnessFieldName(Operation &op) {
  if (auto fieldAttr = op.getAttrOfType<StringAttr>("llzk.field"))
    return fieldAttr.getValue().str();
  if (auto fieldAttr = op.getAttrOfType<FlatSymbolRefAttr>("llzk.field"))
    return fieldAttr.getValue().str();
  return std::nullopt;
}

// Extract struct metadata used to select the right struct parameter.
static std::optional<std::string> getWitnessStructName(Operation &op) {
  if (auto structAttr = op.getAttrOfType<SymbolRefAttr>("llzk.struct"))
    return structAttr.getLeafReference().str();
  if (auto structAttr = op.getAttrOfType<StringAttr>("llzk.struct"))
    return structAttr.getValue().str();
  return std::nullopt;
}

// Infer struct name from a common Struct__constrain naming convention.
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

  if (auto read = dyn_cast<mlir::zkleanstruct::ReadOp>(op)) {
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" := ");
    line.append(lookupValueName(read.getComponent(), valueNames));
    line.push_back('.');
    line.append(read.getFieldNameAttr().getValue());
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

// Format LLZK ops into Lean-like statements for the fallback path.
static std::optional<std::string>
formatLLZKStatement(Operation &op,
                    llvm::DenseMap<Value, std::string> &valueNames,
                    unsigned &nextValueId) {
  if (auto constOp = dyn_cast<llzk::felt::FeltConstantOp>(op)) {
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" := ");
    line.append(formatFeltConstant(constOp));
    return line;
  }

  if (auto read = dyn_cast<llzk::component::FieldReadOp>(op)) {
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" := ");
    line.append(lookupValueName(read.getComponent(), valueNames));
    line.push_back('.');
    line.append(read.getFieldName().str());
    return line;
  }

  if (auto neg = dyn_cast<llzk::felt::NegFeltOp>(op)) {
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" := -");
    line.append(lookupValueName(neg.getOperand(), valueNames));
    return line;
  }

  if (auto add = dyn_cast<llzk::felt::AddFeltOp>(op)) {
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" := ");
    line.append(lookupValueName(add.getLhs(), valueNames));
    line.append(" + ");
    line.append(lookupValueName(add.getRhs(), valueNames));
    return line;
  }

  if (auto sub = dyn_cast<llzk::felt::SubFeltOp>(op)) {
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" := ");
    line.append(lookupValueName(sub.getLhs(), valueNames));
    line.append(" - ");
    line.append(lookupValueName(sub.getRhs(), valueNames));
    return line;
  }

  if (auto mul = dyn_cast<llzk::felt::MulFeltOp>(op)) {
    auto resultNames = assignResultNames(op, valueNames, nextValueId);
    std::string line = "  let ";
    line.append(wrapResultNames(resultNames));
    line.append(" := ");
    line.append(lookupValueName(mul.getLhs(), valueNames));
    line.append(" * ");
    line.append(lookupValueName(mul.getRhs(), valueNames));
    return line;
  }

  if (auto eq = dyn_cast<llzk::constrain::EmitEqualityOp>(op)) {
    std::string line = "  constrainEq ";
    line.append(lookupValueName(eq.getLhs(), valueNames));
    line.push_back(' ');
    line.append(lookupValueName(eq.getRhs(), valueNames));
    return line;
  }

  if (op.getNumResults() == 0)
    return std::nullopt;

  auto resultNames = assignResultNames(op, valueNames, nextValueId);
  std::string line = "  let ";
  line.append(wrapResultNames(resultNames));
  line.append(" := ");
  line.append(formatOperationCall(op, valueNames));
  return line;
}

// Emit Lean for ZKLean/ZKBuilder functions, with struct arg inference.
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

// Emit Lean output directly from LLZK constrain functions.
static bool emitLeanFuncFromLLZK(llzk::function::FuncDefOp func,
                                 raw_ostream &os) {
  if (func.getBody().empty())
    return false;
  if (!func->hasAttr("function.allow_constraint"))
    return false;

  llvm::DenseMap<Value, std::string> valueNames;
  Block &entry = func.front();
  unsigned nonStructIndex = 0;
  unsigned structIndex = 0;
  for (auto arg : entry.getArguments()) {
    if (auto structName = getStructTypeName(arg.getType())) {
      std::string name = structIndex == 0
                             ? "strct"
                             : (llvm::Twine("strct") + llvm::Twine(structIndex))
                                   .str();
      valueNames[arg] = name;
      ++structIndex;
    } else {
      valueNames[arg] = ("arg" + llvm::Twine(nonStructIndex++)).str();
    }
  }

  unsigned nextValueId = 0;
  llvm::SmallVector<std::string, 16> statements;

  for (Operation &op : entry) {
    if (isa<func::ReturnOp>(op))
      continue;
    if (auto stmt = formatLLZKStatement(op, valueNames, nextValueId))
      statements.push_back(*stmt);
  }

  if (statements.empty())
    return false;

  os << "def " << buildLeanFunctionName(func) << " [Field f]";
  for (auto arg : entry.getArguments()) {
    auto argName = lookupValueName(arg, valueNames);
    if (auto structName = getStructTypeName(arg.getType())) {
      os << " (" << argName << " : " << *structName << " f)";
    } else {
      os << " (" << argName << " : " << formatLeanType(arg.getType()) << ")";
    }
  }
  os << " : ZKBuilder f PUnit := do\n";
  for (auto &stmt : statements)
    os << stmt << '\n';

  return true;
}

// Emit Lean structure declarations from LLZK structs.
static bool emitLeanStructsFromLLZK(
    ArrayRef<llzk::component::StructDefOp> structs,
    raw_ostream &os) {
  bool printed = false;
  for (auto def : structs) {
    os << "structure " << def.getSymName() << " where\n";
    for (auto field :
         def.getBody()->getOps<llzk::component::FieldDefOp>()) {
      os << "  " << field.getSymName() << " : "
         << formatLeanType(field.getType()) << '\n';
    }
    os << '\n';
    printed = true;
  }
  return printed;
}

// Emit Lean structure declarations from ZKLean structs.
static bool emitLeanStructsFromZKLean(
    ArrayRef<mlir::zkleanstruct::StructDefOp> structs,
    raw_ostream &os) {
  bool printed = false;
  for (auto def : structs) {
    os << "structure " << def.getSymName() << " where\n";
    for (auto field :
         def.getBodyRegion().getOps<mlir::zkleanstruct::FieldDefOp>()) {
      os << "  " << field.getSymName() << " : "
         << formatLeanType(field.getType()) << '\n';
    }
    os << '\n';
    printed = true;
  }
  return printed;
}

} // namespace namespace_detail
namespace {

struct PrettyPrintZKLeanPass
    : public PassWrapper<PrettyPrintZKLeanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrettyPrintZKLeanPass)

  PrettyPrintZKLeanPass() = default;
  PrettyPrintZKLeanPass(const PrettyPrintZKLeanPass &other)
      : PassWrapper(other) {
    outputFilename = other.outputFilename;
  }

  StringRef getArgument() const final { return "zklean-pretty-print"; }
  StringRef getDescription() const final {
    return "Pretty-print zk dialect IR as Lean code";
  }

  Option<std::string> outputFilename{
      *this, "output-file",
      llvm::cl::desc(
          "File to write the pretty-printed zk dialect operations to. "
          "Use '-' to write to stdout."),
      llvm::cl::init("-")};

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
    *stream << "import ZKLean\nopen ZKBuilder\n\n";

    // Choose ZKLean vs LLZK printing based on whether any ZK ops are present.
    bool hasZkOps = false;
    module.walk([&](Operation *op) {
      if (namespace_detail::isZkDialect(op)) {
        hasZkOps = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    bool printedSomething = false;

    if (hasZkOps) {
      llvm::SmallVector<mlir::zkleanstruct::StructDefOp, 8> structDefs;
      llvm::DenseSet<StringRef> structNames;
      module.walk([&](mlir::zkleanstruct::StructDefOp def) {
        StringRef name = def.getSymName();
        if (structNames.insert(name).second)
          structDefs.push_back(def);
        return WalkResult::advance();
      });

      if (namespace_detail::emitLeanStructsFromZKLean(structDefs, *stream))
        printedSomething = true;

      auto tryEmit = [&](auto funcOp) {
        if (!namespace_detail::emitLeanFunc(funcOp, *stream, structNames))
          return;
        printedSomething = true;
        *stream << '\n';
      };

      module.walk([&](func::FuncOp func) {
        tryEmit(func);
        return WalkResult::advance();
      });
      module.walk([&](llzk::function::FuncDefOp func) {
        tryEmit(func);
        return WalkResult::advance();
      });
    } else {
      llvm::SmallVector<llzk::component::StructDefOp, 8> structDefs;
      module.walk([&](llzk::component::StructDefOp def) {
        structDefs.push_back(def);
        return WalkResult::advance();
      });

      if (namespace_detail::emitLeanStructsFromLLZK(structDefs, *stream))
        printedSomething = true;

      module.walk([&](llzk::function::FuncDefOp func) {
        if (!namespace_detail::emitLeanFuncFromLLZK(func, *stream))
          return WalkResult::advance();
        printedSomething = true;
        *stream << '\n';
        return WalkResult::advance();
      });
    }

    if (!printedSomething)
      *stream << "-- No zk dialect operations found.\n";

    if (outputFile)
      outputFile->keep();
  }
};

} // namespace

std::unique_ptr<Pass> createPrettyPrintZKLeanPass() {
  return std::make_unique<PrettyPrintZKLeanPass>();
}

} // namespace llzk::zklean
