//===-- DialectRegistration.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines dialect and pass registration functions for core LLZK dialects.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/DialectRegistration.h"

#include "smt/Conversions/ConversionPasses.h"

#include "llzk/Analysis/AnalysisPasses.h"
#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Array/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Bool/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Global/IR/Dialect.h"
#include "llzk/Dialect/Global/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/Include/Transforms/InlineIncludesPass.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/POD/IR/Dialect.h"
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Polymorphic/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/RAM/IR/Dialect.h"
#include "llzk/Dialect/SMTInfo/IR/SMTInfoDialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"
#include "llzk/Dialect/Struct/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Verif/IR/Dialect.h"
#include "llzk/Transforms/LLZKTransformationPassPipelines.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Transforms/SpecializedMemoryPasses.h"
#include "llzk/Validators/LLZKValidationPasses.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SMT/IR/SMTDialect.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

namespace llzk {

void registerDialects(mlir::DialectRegistry &registry) {
  registry.insert<
      // clang-format off
      llzk::LLZKDialect,
      llzk::array::ArrayDialect,
      llzk::boolean::BoolDialect,
      llzk::cast::CastDialect,
      llzk::component::StructDialect,
      llzk::constrain::ConstrainDialect,
      llzk::felt::FeltDialect,
      llzk::function::FunctionDialect,
      llzk::global::GlobalDialect,
      llzk::include::IncludeDialect,
      llzk::pod::PODDialect,
      llzk::polymorphic::PolymorphicDialect,
      llzk::ram::RAMDialect,
      llzk::smt_info::SMTInfoDialect,
      llzk::string::StringDialect,
      llzk::verif::VerifDialect,
      mlir::arith::ArithDialect,
      mlir::scf::SCFDialect,
      mlir::smt::SMTDialect
      // clang-format on
      >();

  verif::registerExtensions(registry);
}

/// Replace `mlir::registerTransformsPasses()` to register a custom `remove-dead-values` pass
/// because MLIR version 23.1.0 has a bug where the pass tracks `poison` values that it created
/// during its current invocation only and may end up leaving behind dead `poison` values.
namespace mlir_patch {

static inline void registerTransformsPasses() {
  mlir::registerBubbleDownMemorySpaceCastsPass();
  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerCompositeFixedPointPass();
  mlir::registerControlFlowSinkPass();
  mlir::registerGenerateRuntimeVerificationPass();
  mlir::registerInlinerPass();
  mlir::registerLocationSnapshotPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerLoopInvariantSubsetHoistingPass();
  mlir::registerMem2RegPass();
  mlir::registerPrintIRPass();
  mlir::registerPrintOpStatsPass();
  mlir::registerPass(llzk::createRemoveDeadValuesWorkaroundPass);
  mlir::registerSCCPPass();
  mlir::registerSROAPass();
  mlir::registerStripDebugInfoPass();
  mlir::registerSymbolDCEPass();
  mlir::registerSymbolPrivatizePass();
  mlir::registerTopologicalSortPass();
  mlir::registerTrivialDeadCodeEliminationPass();
  mlir::registerViewOpGraphPass();
}

} // namespace mlir_patch

void registerPasses(mlir::DialectRegistry &registry) {
  llzk::registerValidationPasses();
  llzk::registerAnalysisPasses();
  llzk::registerTransformationPasses();
  llzk::array::registerTransformationPasses();
  llzk::component::registerTransformationPasses();
  llzk::boolean::registerTransformationPasses();
  llzk::global::registerTransformationPasses();
  llzk::include::registerTransformationPasses();
  llzk::polymorphic::registerTransformationPasses();
  llzk::pod::registerTransformationPasses();
  llzk::smt::registerConversionPasses();
  llzk::registerTransformationPassPipelines();

  // Inlining extensions for `llzk-inline-structs` (or MLIR inlining in general)
  registerInliningExtensions(registry);

  // Inlining extensions for `llzk-inline-free-functions` and smt lowering
  mlir::func::registerInlinerExtension(registry);

  // MLIR builtin pass registration
  mlir_patch::registerTransformsPasses();
}

} // namespace llzk
