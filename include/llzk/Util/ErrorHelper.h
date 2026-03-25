//===-- ErrorHelper.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/ErrorHandling.h>

#include <variant>

namespace llzk {

/// Wrapper around InFlightDiagnostic that can either be a regular InFlightDiagnostic or a
/// special version that asserts false after reporting the diagnostic.
/// See `wrapNullableInFlightDiagnostic()` below for details.
class InFlightDiagnosticWrapper {
private:
  /// Implementation that gives an assertion failure after reporting.
  /// See `wrapNullableInFlightDiagnostic()` below for details.
  class DefaultAndFailInFlightDiagnostic : public mlir::InFlightDiagnostic {
  public:
    DefaultAndFailInFlightDiagnostic(mlir::InFlightDiagnostic &&base)
        : mlir::InFlightDiagnostic(std::move(base)) {}

    DefaultAndFailInFlightDiagnostic(DefaultAndFailInFlightDiagnostic &&other) noexcept
        : mlir::InFlightDiagnostic(std::move(other)) {}

    ~DefaultAndFailInFlightDiagnostic() {
      if (getUnderlyingDiagnostic()) {
        report();
      }
    }

    void report() {
      InFlightDiagnostic::report();
      assert(false);
    }
  };

  std::variant<mlir::InFlightDiagnostic, DefaultAndFailInFlightDiagnostic> inner;

  explicit InFlightDiagnosticWrapper(DefaultAndFailInFlightDiagnostic &&diag)
      : inner(std::move(diag)) {}

public:
  /// Constructor for regular InFlightDiagnostic.
  explicit InFlightDiagnosticWrapper(mlir::InFlightDiagnostic &&diag) : inner(std::move(diag)) {}

  /// Constructor for DefaultAndFailInFlightDiagnostic from MLIRContext.
  /// NOTE: This is not a common use case since it will always result in an assertion failure
  /// immediately after reporting the error; likely only useful in custom type builders.
  explicit InFlightDiagnosticWrapper(mlir::MLIRContext *ctx)
      : InFlightDiagnosticWrapper(
            DefaultAndFailInFlightDiagnostic(mlir::detail::getDefaultDiagnosticEmitFn(ctx)())
        ) {}

  /// Constructor for DefaultAndFailInFlightDiagnostic from Location.
  /// NOTE: This is not a common use case since it will always result in an assertion failure
  /// immediately after reporting the error; likely only useful in custom type builders.
  explicit InFlightDiagnosticWrapper(const mlir::Location &loc)
      : InFlightDiagnosticWrapper(loc.getContext()) {}

  /// Stream operator for new diagnostic arguments.
  template <typename Arg> InFlightDiagnosticWrapper &operator<<(Arg &&arg) & {
    return append(std::forward<Arg>(arg));
  }
  /// Stream operator for new diagnostic arguments.
  template <typename Arg> InFlightDiagnosticWrapper &&operator<<(Arg &&arg) && {
    return std::move(append(std::forward<Arg>(arg)));
  }

  /// Append arguments to the diagnostic.
  template <typename... Args> InFlightDiagnosticWrapper &append(Args &&...args) & {
    std::visit([&](auto &diag) { diag.append(std::forward<Args>(args)...); }, inner);
    return *this;
  }
  /// Append arguments to the diagnostic.
  template <typename... Args> InFlightDiagnosticWrapper &&append(Args &&...args) && {
    return std::move(append(std::forward<Args>(args)...));
  }

  /// Attaches a note to this diagnostic.
  mlir::Diagnostic &attachNote(std::optional<mlir::Location> noteLoc = std::nullopt) {
    return std::visit([&](auto &diag) -> mlir::Diagnostic & {
      return diag.attachNote(noteLoc);
    }, inner);
  }

  /// Returns the underlying diagnostic or nullptr if this diagnostic isn't active.
  mlir::Diagnostic *getUnderlyingDiagnostic() {
    return std::visit([](auto &diag) -> mlir::Diagnostic * {
      return diag.getUnderlyingDiagnostic();
    }, inner);
  }

  /// Reports the diagnostic to the engine.
  void report() {
    std::visit([](auto &diag) { diag.report(); }, inner);
  }

  /// Abandons this diagnostic so that it will no longer be reported.
  void abandon() {
    std::visit([](auto &diag) { diag.abandon(); }, inner);
  }

  /// Allow an inflight diagnostic to be converted to 'failure', otherwise 'success' if this is an
  /// empty diagnostic.
  operator mlir::LogicalResult() const {
    return std::visit([](const auto &diag) -> mlir::LogicalResult { return diag; }, inner);
  }

  /// Allow an inflight diagnostic to be converted to 'failure', otherwise 'success' if this is an
  /// empty diagnostic.
  operator mlir::ParseResult() const { return mlir::ParseResult(mlir::LogicalResult(*this)); }

  /// Allow an inflight diagnostic to be converted to FailureOr<T>. Always results in 'failure'
  /// because this cast cannot possibly return an object of 'T'.
  template <typename T> operator mlir::FailureOr<T>() const { return mlir::failure(); }

  // Match move/copy semantics of InFlightDiagnostic
  InFlightDiagnosticWrapper(InFlightDiagnosticWrapper &&) = default;
  InFlightDiagnosticWrapper(const InFlightDiagnosticWrapper &) = delete;
  InFlightDiagnosticWrapper &operator=(InFlightDiagnosticWrapper &&) = delete;
  InFlightDiagnosticWrapper &operator=(const InFlightDiagnosticWrapper &) = delete;
};

/// Callback to produce an error diagnostic.
using EmitErrorFn = llvm::function_ref<InFlightDiagnosticWrapper()>;

/// This type is required in cases like the functions below to take ownership of the lambda so it is
/// not destroyed upon return from the function. It can be implicitly converted to EmitErrorFn.
using OwningEmitErrorFn = std::function<InFlightDiagnosticWrapper()>;

inline OwningEmitErrorFn getEmitOpErrFn(mlir::Operation *op) {
  return [op]() { return InFlightDiagnosticWrapper(op->emitOpError()); };
}

template <typename OpImplClass> inline OwningEmitErrorFn getEmitOpErrFn(OpImplClass *opImpl) {
  return getEmitOpErrFn(opImpl->getOperation());
}

inline void ensure(bool condition, const llvm::Twine &errMsg) {
  if (!condition) {
    llvm::report_fatal_error(errMsg);
  }
}

/// If the given `emitError` is non-null, return it. Otherwise, mirror how the verification failure
/// is handled by `*Type::get()` via `StorageUserBase` (i.e., use DefaultDiagnosticEmitFn and assert
/// after reporting the error).
///
/// NOTE: Passing `emitError == null` is not a common use case since it will always result in an
/// assertion failure immediately after reporting the error; likely only useful in custom type
/// builders.
///
/// SEE:
/// https://github.com/llvm/llvm-project/blob/0897373f1a329a7a02f8ce3c501a05d2f9c89390/mlir/include/mlir/IR/StorageUniquerSupport.h#L179-L180
inline OwningEmitErrorFn wrapNullableInFlightDiagnostic(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::MLIRContext *ctx
) {
  if (emitError) {
    return [emitError]() -> auto { return InFlightDiagnosticWrapper(emitError()); };
  } else {
    return [ctx]() -> auto { return InFlightDiagnosticWrapper(ctx); };
  }
}

inline OwningEmitErrorFn
wrapNonNullableInFlightDiagnostic(llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  assert(emitError && "emitError must be non-null");
  return [emitError]() -> auto { return InFlightDiagnosticWrapper(emitError()); };
}

} // namespace llzk
