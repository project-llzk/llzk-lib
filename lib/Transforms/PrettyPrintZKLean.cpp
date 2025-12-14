#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Transforms/ZKLeanPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

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
    return "f";

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

static std::string formatOperationCall(Operation &op,
                                       llvm::DenseMap<Value, std::string> &names) {
  std::string call = op.getName().getStringRef().str();
  for (Value operand : op.getOperands()) {
    call.push_back(' ');
    call.append(lookupValueName(operand, names));
  }
  return call;
}

static std::string
formatLeanStatement(Operation &op,
                    llvm::DenseMap<Value, std::string> &valueNames,
                    unsigned &nextValueId) {
  llvm::SmallVector<std::string, 4> resultNames;
  for (Value result : op.getResults()) {
    std::string name = ("v" + llvm::Twine(nextValueId++)).str();
    valueNames[result] = name;
    resultNames.push_back(std::move(name));
  }

  // indentation in front of 'let'
  std::string line = "  let ";
  // append reformatted variable names
  line.append(wrapResultNames(resultNames));
  line.push_back(' ');
  // replace with '<-' if ZKBuilder op and := otherwise
  line.append(isBuilderOp(&op) ? "<-" : ":=");
  line.push_back(' ');
  line.append(formatOperationCall(op, valueNames));
  return line;
}

template <typename FuncOpTy>
static bool emitLeanFunc(FuncOpTy func, raw_ostream &os) {
  if (func.getBody().empty())
    return false;

  llvm::DenseMap<Value, std::string> valueNames;
  Block &entry = func.front();
  for (auto [idx, arg] : llvm::enumerate(entry.getArguments())) {
    std::string name = ("arg" + llvm::Twine(idx)).str();
    valueNames[arg] = name;
  }

  unsigned nextValueId = 0;
  llvm::SmallVector<std::string, 16> statements;

  for (Operation &op : entry) {
    if (isa<func::ReturnOp>(op))
      continue;
    if (!isZkDialect(&op))
      continue;
    statements.push_back(formatLeanStatement(op, valueNames, nextValueId));
  }

  if (statements.empty())
    return false;

  os << "def " << func.getSymName() << " [Field f]";
  for (auto [idx, arg] : llvm::enumerate(entry.getArguments())) {
    os << " (arg" << idx << " : " << formatLeanType(arg.getType()) << ")";
  }
  os << " : ZKBuilder f PUnit := do\n";
  for (auto &stmt : statements)
    os << stmt << '\n';

  return true;
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

  StringRef getArgument() const final { return "zk-lean-pretty-print"; }
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
    bool printedSomething = false;
    auto tryEmit = [&](auto funcOp) {
      if (!namespace_detail::emitLeanFunc(funcOp, *stream))
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
