//===-- AnalysisPasses.td - LLZK Analysis Passes -----------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_ANALYSIS_TD
#define LLZK_ANALYSIS_TD

include "llzk/Pass/PassBase.td"
include "mlir/IR/EnumAttr.td"

def OutputStreamDescription {
  string r = "Specifies the stream to which the pass prints.";
}

/// Enumeration of the output stream options in LLVM.
def OutputStream : I32EnumAttr<"OutputStream", OutputStreamDescription.r,
                               [I32EnumAttrCase<"Outs", 1, "outs">,
                                I32EnumAttrCase<"Errs", 2, "errs">,
                                I32EnumAttrCase<"Dbgs", 3, "dbgs">,
]> {
  let cppNamespace = "::llzk";
  let genSpecializedAttr = 0;
}

/// Reusable pass Option for allowing a pass user to specify the output stream.
def OutputStreamOption
    : Option<"outputStream", "stream", "::llzk::OutputStream",
             /* default */ "::llzk::OutputStream::Errs",
             OutputStreamDescription.r, [{::llvm::cl::values(
                clEnumValN(::llzk::OutputStream::Outs, stringifyOutputStream(::llzk::OutputStream::Outs),
                           "Print pass output to 'llvm::outs()'"),
                clEnumValN(::llzk::OutputStream::Errs, stringifyOutputStream(::llzk::OutputStream::Errs),
                           "Print pass output to 'llvm::errs()'"),
                clEnumValN(::llzk::OutputStream::Dbgs, stringifyOutputStream(::llzk::OutputStream::Dbgs),
                           "Print pass output to 'llvm::dbgs()'")
          )}]>;

/// Reusable pass Option allowing pass user to specify whether or not to dump
/// the DOT graph to file.
def SaveDotGraphOption
    : Option<"saveDotGraph", "saveDot", "bool",
             /* default */ "false", "Whether to dump the graph to DOT format.">;

//===----------------------------------------------------------------------===//
// Analysis Pass definitions
//===----------------------------------------------------------------------===//

def CallGraphPrinterPass : LLZKPass<"llzk-print-call-graph"> {
  let summary = "Print the LLZK module's call graph.";
  let constructor = "llzk::createCallGraphPrinterPass(llvm::errs())";
}

def CallGraphSCCsPrinterPass : LLZKPass<"llzk-print-call-graph-sccs"> {
  let summary = "Print the SCCs from the LLZK module's call graph.";
  let constructor = "llzk::createCallGraphSCCsPrinterPass(llvm::errs())";
}

def ConstraintDependencyGraphPrinterPass
    : LLZKPass<"llzk-print-constraint-dependency-graphs"> {
  let summary = "Print constraint dependency graph for all LLZK structs.";
  let constructor =
      "llzk::createConstraintDependencyGraphPrinterPass(llvm::errs())";
  let options = [Option<"runIntraprocedural", "intraprocedural", "bool",
                        /* default */ "false",
                        "Whether to run the analysis intra-procedurally only "
                        "(default is false).">];
}

def IntervalAnalysisPrinterPass : LLZKPass<"llzk-print-interval-analysis"> {
  let summary = "Print interval analysis results for all LLZK structs.";
  let constructor = "llzk::createIntervalAnalysisPrinterPass(llvm::errs())";
  let options =
      [Option<"fieldName", "field", "std::string", /* default */ "\"bn128\"",
              "The field to use for interval analysis. Supported fields: "
              "bn128/bn254, babybear, goldilocks, mersenne31">,
       Option<"propagateInputConstraints", "propagate-input-constraints",
              "bool", /* default */ "false",
              "Whether to propagate constraints on inputs from @constrain to "
              "@compute functions. "
              "This allows for tighter intervals to possibly be found for "
              "computed values, "
              "assuming that the witness generator would include constraints "
              "as assertions during the computation.">,
       Option<"printSolverConstraints", "print-solver-constraints", "bool",
              /* default */ "false",
              "Whether to output SMT solver constraints along with intervals.">,
       Option<"printComputeIntervals", "print-compute-intervals", "bool",
              /* default */ "false",
              "Whether to print compute function intervals (default only "
              "prints constrain function intervals).">,
  ];
}

def SymbolDefTreePrinterPass : LLZKPass<"llzk-print-symbol-def-tree"> {
  let summary = "Print symbol definition tree.";
  let constructor = "llzk::createSymbolDefTreePrinterPass()";
  let options = [OutputStreamOption, SaveDotGraphOption];
}

def SymbolUseGraphPrinterPass : LLZKPass<"llzk-print-symbol-use-graph"> {
  let summary = "Print symbol use graph.";
  let constructor = "llzk::createSymbolUseGraphPrinterPass()";
  let options = [OutputStreamOption, SaveDotGraphOption];
}

def PredecessorPrinterPass : LLZKPass<"llzk-print-predecessors"> {
  let summary = "Print the predecessors of all operations.";
  let constructor = "llzk::createPredecessorPrinterPass()";
  let options = [OutputStreamOption,
                 Option<"preRunRequiredAnalyses", "prerun", "bool",
                        /* default */ "false",
                        "Whether to pre-run the required dataflow analyses "
                        "(e.g., liveness analysis).">];
}

#endif // LLZK_ANALYSIS_TD
