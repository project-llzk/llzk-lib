//===-- llzk-smt-check.cpp - SMT-LIB staged checker ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tools/config.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/raw_ostream.h>

#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

enum class SatResult { Sat, Unsat, Unknown };

struct StageExpectation {
  std::string rootName;
  std::string stageName;
  SatResult expected;
  bool hasExpected = false;
};

struct ScriptMetadata {
  SmallVector<StageExpectation> stages;
  size_t checkSatCount = 0;
};

struct SolverInvocationResult {
  int exitCode = 0;
  bool executionFailed = false;
  std::string errorMessage;
  std::string stdoutText;
  std::string stderrText;
};

struct TempFileCleanup {
  SmallVector<SmallString<128>> paths;

  ~TempFileCleanup() {
    for (const SmallString<128> &path : paths) {
      (void)sys::fs::remove(path);
    }
  }
};

static cl::opt<std::string> InputFilename(cl::Positional, cl::Required);
static cl::opt<std::string>
    SolverBinary("solver-binary", cl::desc("SMT solver executable"), cl::init("z3"));
static cl::opt<bool> Quiet("quiet", cl::desc("Suppress per-stage summaries"));
static cl::opt<bool>
    DumpRawOutput("dump-raw-output", cl::desc("Print raw solver stdout after the stage summaries"));

StringRef stringify(SatResult result) {
  switch (result) {
  case SatResult::Sat:
    return "sat";
  case SatResult::Unsat:
    return "unsat";
  case SatResult::Unknown:
    return "unknown";
  }
  llvm_unreachable("unknown sat result");
}

std::optional<SatResult> parseSatResult(StringRef text) {
  if (text == "sat") {
    return SatResult::Sat;
  }
  if (text == "unsat") {
    return SatResult::Unsat;
  }
  if (text == "unknown") {
    return SatResult::Unknown;
  }
  return std::nullopt;
}

Expected<std::string> readInput(StringRef inputFilename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> buffer = MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!buffer) {
    return createStringError(buffer.getError(), "failed to read input '%s'", inputFilename.data());
  }
  return std::string(buffer.get()->getBuffer());
}

Expected<ScriptMetadata> scanScript(StringRef script) {
  ScriptMetadata metadata;
  std::optional<std::string> currentRoot;
  std::optional<std::string> pendingInfoStage;
  std::optional<SatResult> pendingInfoExpected;

  auto parseQuotedString = [&](StringRef value, StringRef fullLine) -> Expected<std::string> {
    StringRef trimmed = value.trim();
    if (trimmed.size() < 2 || !trimmed.starts_with("\"") || !trimmed.ends_with("\"")) {
      return createStringError(
          inconvertibleErrorCode(), "invalid set-info annotation: '%s'", fullLine.str().c_str()
      );
    }
    return trimmed.drop_front().drop_back().str();
  };

  SmallVector<StringRef> lines;
  script.split(lines, '\n');
  for (StringRef line : lines) {
    StringRef trimmed = line.ltrim();
    if (trimmed.starts_with("(set-info")) {
      StringRef body = trimmed.drop_front(StringRef("(set-info").size()).trim();
      if (!body.ends_with(")")) {
        return createStringError(
            inconvertibleErrorCode(), "invalid set-info annotation: '%s'", trimmed.str().c_str()
        );
      }
      body = body.drop_back().trim();
      size_t splitPos = body.find_first_of(" \t");
      if (splitPos == StringRef::npos) {
        return createStringError(
            inconvertibleErrorCode(), "invalid set-info annotation: '%s'", trimmed.str().c_str()
        );
      }
      StringRef key = body.take_front(splitPos).trim();
      StringRef value = body.drop_front(splitPos).trim();
      if (key == ":status") {
        pendingInfoExpected = parseSatResult(value);
        if (!pendingInfoExpected) {
          return createStringError(
              inconvertibleErrorCode(), "invalid set-info annotation: '%s'", trimmed.str().c_str()
          );
        }
      } else if (key == ":llzk-stage") {
        auto parsed = parseQuotedString(value, trimmed);
        if (!parsed) {
          return parsed.takeError();
        }
        pendingInfoStage = std::move(*parsed);
      } else if (key == ":llzk-root") {
        auto parsed = parseQuotedString(value, trimmed);
        if (!parsed) {
          return parsed.takeError();
        }
        currentRoot = std::move(*parsed);
      }
      continue;
    }
    if (trimmed == "(check-sat)") {
      std::optional<std::string> stageName = pendingInfoStage;
      std::optional<SatResult> expected = pendingInfoExpected;

      ++metadata.checkSatCount;

      if (stageName && !expected) {
        return createStringError(
            inconvertibleErrorCode(), "missing expected result before check-sat for stage '%s'",
            stageName->c_str()
        );
      }

      if (stageName || expected) {
        metadata.stages.push_back(
            StageExpectation {
                currentRoot.value_or(""), stageName.value_or(""),
                expected.value_or(SatResult::Unknown), expected.has_value()
            }
        );
      }

      pendingInfoStage.reset();
      pendingInfoExpected.reset();
      continue;
    }
  }

  if (!metadata.stages.empty() && metadata.stages.size() != metadata.checkSatCount) {
    return createStringError(
        inconvertibleErrorCode(), "check metadata count (%zu) does not match check-sat count (%zu)",
        metadata.stages.size(), metadata.checkSatCount
    );
  }

  return metadata;
}

std::string stripMetadataForSolver(StringRef script) {
  std::string sanitized;
  raw_string_ostream os(sanitized);

  SmallVector<StringRef> lines;
  script.split(lines, '\n');
  for (StringRef line : lines) {
    StringRef trimmed = line.ltrim();
    bool shouldStrip = false;
    if (trimmed.starts_with("(set-info")) {
      StringRef body = trimmed.drop_front(StringRef("(set-info").size()).trim();
      if (body.ends_with(")")) {
        body = body.drop_back().trim();
        size_t splitPos = body.find_first_of(" \t");
        if (splitPos != StringRef::npos) {
          StringRef key = body.take_front(splitPos).trim();
          shouldStrip = key == ":status" || key == ":llzk-stage" || key == ":llzk-root";
        }
      }
    }

    if (!shouldStrip) {
      os << line << '\n';
    }
  }

  return sanitized;
}

std::string formatStageLabel(const StageExpectation &stage, size_t index) {
  if (!stage.stageName.empty() && !stage.rootName.empty()) {
    return (Twine(stage.rootName) + "/" + stage.stageName).str();
  }
  if (!stage.stageName.empty()) {
    return stage.stageName;
  }
  return ("check[" + std::to_string(index) + "]");
}

Expected<std::string> readWholeFile(StringRef path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> buffer = MemoryBuffer::getFile(path);
  if (!buffer) {
    return createStringError(buffer.getError(), "failed to read '%s'", path.data());
  }
  return std::string(buffer.get()->getBuffer());
}

Expected<std::string> resolveSolverPath(StringRef solverBinary) {
  if (sys::path::has_parent_path(solverBinary)) {
    return solverBinary.str();
  }
  ErrorOr<std::string> found = sys::findProgramByName(solverBinary);
  if (!found) {
    return createStringError(
        found.getError(), "failed to find solver binary '%s'", solverBinary.data()
    );
  }
  return *found;
}

Expected<SmallString<128>> createTempFile(StringRef prefix, StringRef suffix, StringRef contents) {
  SmallString<128> path;
  int fd = -1;
  std::error_code ec = sys::fs::createTemporaryFile(prefix, suffix, fd, path);
  if (ec) {
    return createStringError(ec, "failed to create temporary file");
  }

  raw_fd_ostream os(fd, true);
  if (!contents.empty()) {
    os << contents;
  }
  os.flush();
  if (os.has_error()) {
    return createStringError(inconvertibleErrorCode(), "failed to write temporary file");
  }

  return path;
}

Expected<SolverInvocationResult> runSolver(StringRef solverPath, StringRef script) {
  SolverInvocationResult result;
  TempFileCleanup cleanup;

  auto stdoutFile = createTempFile("llzk-smt-check-stdout", "txt", "");
  if (!stdoutFile) {
    return stdoutFile.takeError();
  }
  auto stderrFile = createTempFile("llzk-smt-check-stderr", "txt", "");
  if (!stderrFile) {
    return stderrFile.takeError();
  }
  cleanup.paths.push_back(*stdoutFile);
  cleanup.paths.push_back(*stderrFile);

  std::array<std::optional<StringRef>, 3> redirects = {
      std::nullopt, StringRef(stdoutFile->data(), stdoutFile->size()),
      StringRef(stderrFile->data(), stderrFile->size())
  };

  SmallVector<StringRef> args;
  args.push_back(solverPath);
  auto tempInput = createTempFile("llzk-smt-check-input", "smt2", script);
  if (!tempInput) {
    return tempInput.takeError();
  }
  cleanup.paths.push_back(*tempInput);
  args.push_back("-smt2");
  args.push_back(StringRef(tempInput->data(), tempInput->size()));

  std::string errorMessage;
  bool executionFailed = false;
  result.exitCode = sys::ExecuteAndWait(
      solverPath, args, std::nullopt, redirects, 0, 0, &errorMessage, &executionFailed
  );
  result.executionFailed = executionFailed;
  result.errorMessage = std::move(errorMessage);

  auto stdoutText = readWholeFile(*stdoutFile);
  if (!stdoutText) {
    return stdoutText.takeError();
  }
  result.stdoutText = std::move(*stdoutText);

  auto stderrText = readWholeFile(*stderrFile);
  if (!stderrText) {
    return stderrText.takeError();
  }
  result.stderrText = std::move(*stderrText);

  return result;
}

Expected<std::pair<SmallVector<SatResult>, SmallVector<std::string>>>
parseSolverStdout(StringRef text) {
  SmallVector<SatResult> results;
  SmallVector<std::string> extraLines;
  SmallVector<StringRef> lines;
  text.split(lines, '\n');
  for (StringRef line : lines) {
    StringRef trimmed = line.trim();
    if (trimmed.empty()) {
      continue;
    }
    if (std::optional<SatResult> result = parseSatResult(trimmed)) {
      results.push_back(*result);
      continue;
    }

    std::string lowered = trimmed.lower();
    if (StringRef(lowered).starts_with("z3 ")) {
      continue;
    }
    extraLines.push_back(trimmed.str());
  }
  return std::make_pair(std::move(results), std::move(extraLines));
}

void printSolverFailure(const SolverInvocationResult &invocation) {
  if (!invocation.errorMessage.empty()) {
    errs() << "llzk-smt-check: " << invocation.errorMessage << '\n';
  }
  if (!invocation.stderrText.empty()) {
    errs() << invocation.stderrText;
    if (!invocation.stderrText.ends_with('\n')) {
      errs() << '\n';
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(StringRef());
  setBugReportMsg(
      "PLEASE submit a bug report to " BUG_REPORT_URL
      " and include the crash backtrace, relevant SMT-LIB inputs, and associated run script(s).\n"
  );

  cl::ParseCommandLineOptions(
      argc, argv,
      "llzk-smt-check: run an SMT solver on staged SMT-LIB and validate per-stage results.\n"
  );

  auto input = readInput(InputFilename);
  if (!input) {
    errs() << toString(input.takeError()) << '\n';
    return EXIT_FAILURE;
  }

  auto metadata = scanScript(*input);
  if (!metadata) {
    errs() << "llzk-smt-check: " << toString(metadata.takeError()) << '\n';
    return EXIT_FAILURE;
  }

  auto solverPath = resolveSolverPath(SolverBinary);
  if (!solverPath) {
    errs() << "llzk-smt-check: " << toString(solverPath.takeError()) << '\n';
    return EXIT_FAILURE;
  }

  std::string solverScript = stripMetadataForSolver(*input);
  auto invocation = runSolver(*solverPath, solverScript);
  if (!invocation) {
    errs() << "llzk-smt-check: " << toString(invocation.takeError()) << '\n';
    return EXIT_FAILURE;
  }
  if (invocation->executionFailed || invocation->exitCode != 0) {
    errs() << "llzk-smt-check: solver exited with code " << invocation->exitCode << '\n';
    printSolverFailure(*invocation);
    return EXIT_FAILURE;
  }

  auto parsedStdout = parseSolverStdout(invocation->stdoutText);
  if (!parsedStdout) {
    errs() << "llzk-smt-check: " << toString(parsedStdout.takeError()) << '\n';
    return EXIT_FAILURE;
  }

  SmallVector<SatResult> solverResults = std::move(parsedStdout->first);
  SmallVector<std::string> extraLines = std::move(parsedStdout->second);
  if (!DumpRawOutput && !extraLines.empty()) {
    errs() << "llzk-smt-check: unexpected solver stdout:\n";
    for (const std::string &line : extraLines) {
      errs() << line << '\n';
    }
    return EXIT_FAILURE;
  }
  if (solverResults.size() != metadata->checkSatCount) {
    errs() << "llzk-smt-check: solver returned " << solverResults.size() << " result(s) for "
           << metadata->checkSatCount << " check-sat command(s)\n";
    return EXIT_FAILURE;
  }

  SmallVector<std::string> mismatches;
  SmallVector<std::string> summaries;
  for (size_t i = 0; i < solverResults.size(); ++i) {
    std::string label;
    if (metadata->stages.empty() || metadata->stages[i].stageName.empty()) {
      label = "check[" + std::to_string(i) + "]";
    } else {
      label = formatStageLabel(metadata->stages[i], i);
    }
    if (!Quiet) {
      std::string summary = (Twine(label) + ": " + stringify(solverResults[i])).str();
      if (!metadata->stages.empty() && metadata->stages[i].hasExpected) {
        summary =
            (Twine(summary) + " (expected " + stringify(metadata->stages[i].expected) + ")").str();
      }
      summaries.push_back(std::move(summary));
    }
    if (!metadata->stages.empty() && metadata->stages[i].hasExpected &&
        solverResults[i] != metadata->stages[i].expected) {
      mismatches.push_back((Twine(label) + ": got " + stringify(solverResults[i]) + ", expected " +
                            stringify(metadata->stages[i].expected))
                               .str());
    }
  }

  if (DumpRawOutput && !invocation->stdoutText.empty()) {
    outs() << "--- raw solver stdout ---\n" << invocation->stdoutText;
    if (!invocation->stdoutText.ends_with('\n')) {
      outs() << '\n';
    }
  }

  if (!mismatches.empty()) {
    for (const std::string &summary : summaries) {
      errs() << summary << '\n';
    }
    errs() << "llzk-smt-check: stage result mismatch:\n";
    for (const std::string &mismatch : mismatches) {
      errs() << mismatch << '\n';
    }
    return EXIT_FAILURE;
  }

  for (const std::string &summary : summaries) {
    outs() << summary << '\n';
  }

  return EXIT_SUCCESS;
}
