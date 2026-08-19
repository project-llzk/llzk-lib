//===-- Modes.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Temporary.h"
#include "Types.h"
#include "UsedFreeFunctions.h"

#include <mlir/IR/BuiltinOps.h>

namespace mlir {
class TypeConverter;
class RewritePatternSet;
class ConversionTarget;
} // namespace mlir

namespace pcl::lowering {

/// Base class for lowering modes.
class BaseMode {
  mlir::ModuleOp module;
  UsedFreeFunctions usedFreeFunctions;
  Temporaries temps;

protected:
  mlir::MLIRContext &getContext() { return *module.getContext(); }

  mlir::ModuleOp getOperation() { return module; }

  UsedFreeFunctions &getUsedFreeFunctions() { return usedFreeFunctions; }

  Temporaries &getTemps() { return temps; }

  /// Returns true if the operation is legal wrt step 1.
  ///
  /// An operation is legal in Step 1 if its located outside the
  /// `constrain` function of a struct or a used free function.
  bool isStep1LegalOp(mlir::Operation *op);

  void
  populateStep1ConversionPatterns(const mlir::TypeConverter &tc, mlir::RewritePatternSet &patterns);

  void populateStep1ConversionTarget(mlir::ConversionTarget &target);

  virtual void populateStep2ConversionPatterns(
      const mlir::TypeConverter &tc, mlir::RewritePatternSet &patterns
  ) = 0;

  void populateStep2ConversionTarget(mlir::ConversionTarget &target);

  /// Populates the set with the patterns used in step 3 of the conversion.
  void populateStep3ConversionPatterns(
      mlir::RewritePatternSet &patterns, DupVarsReplacements &replacements
  );

  /// Populates the conversion target with the legality expected of step 3 of the conversion.
  void
  populateStep3ConversionTarget(mlir::ConversionTarget &target, DupVarsReplacements &replacements);

  /// Step 1 converts the body of each struct to PCL operations.
  ///
  /// This conversion is performed before moving the body to a function because
  /// that way the IR can access information about the members of the struct.
  virtual mlir::LogicalResult runStep1() = 0;

  /// Step 2 converts the struct to a function, moving the contents of the @constrain function
  /// into the body of the new function.
  mlir::LogicalResult runStep2();

  /// Step 3 cleans up the IR removing unnecessary ops that may be left over by the previous steps.
  ///
  /// The cleanup operations are:
  ///
  /// - The conversion process may generate multiple copies of the same variable. This is fine since
  /// `VarOp` implements `Pure`. However, for cleaniness we remove these duplicates now, replacing
  /// all extra instances with the value that dominates everyone else.
  ///
  /// - Remove empty non-root module ops.
  mlir::LogicalResult runStep3();

  /// Collects all the vars that need to be replaced.
  DupVarsReplacements collectDupVarsReplacements();

public:
  BaseMode(mlir::ModuleOp op);

  virtual ~BaseMode() = default;

  mlir::LogicalResult lower() {
    return llvm::failure(failed(runStep1()) || failed(runStep2()) || failed(runStep3()));
  }
};

/// Full lowering mode.
struct FullLoweringMode : public BaseMode {
  using BaseMode::BaseMode;

  /// Step 1 converts the body of each struct to PCL operations.
  ///
  /// This conversion is performed before moving the body to a function because
  /// that way the IR can access information about the members of the struct.
  mlir::LogicalResult runStep1() final;

  void populateStep2ConversionPatterns(
      const mlir::TypeConverter &tc, mlir::RewritePatternSet &patterns
  ) override;
};

/// Stubbed lowering mode.
struct StubbedLoweringMode : public BaseMode {
  using BaseMode::BaseMode;

private:
  llvm::DenseSet<mlir::Operation *> legalizableOps;
  llvm::DenseSet<llzk::function::FuncDefOp> stubs;

  /// Run step 1 in analysis mode. Running the conversion this way will note
  /// all the ops that will be converted if the conversion is actually applied.
  mlir::LogicalResult analyze();

  /// Checks what operations inside the used free functions are not in the legalizableOps set.
  /// If any, the free function is removed from the set and marked as a stub.
  /// Functions marked as stubs are lowered as such in step 2.
  void checkAnalysisResult();

  /// Step 1 converts the body of each struct to PCL operations.
  ///
  /// This conversion is performed before moving the body to a function because
  /// that way the IR can access information about the members of the struct.
  mlir::LogicalResult runStep1() final;

  void populateStep2ConversionPatterns(
      const mlir::TypeConverter &tc, mlir::RewritePatternSet &patterns
  ) override;
};

} // namespace pcl::lowering
