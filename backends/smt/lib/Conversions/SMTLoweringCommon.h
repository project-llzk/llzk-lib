//===-- SMTLoweringCommon.h ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Common private implementation shared by the integer SMT lowerings.
///
/// This layer only contains mechanical lowering infrastructure that is
/// independent of the modular encoding strategy: field selection, type
/// conversion, legality checks, and generic conversion patterns for struct,
/// function, and control-flow scaffolding. The naive and optimized passes then
/// supply their own policy for how felt arithmetic constraints are encoded.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Analysis/Intervals.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Field.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SMT/IR/SMTOps.h>
#include <mlir/Dialect/SMT/IR/SMTTypes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringMap.h>

#include <memory>
#include <optional>
#include <utility>

namespace llzk::smt::detail {

// Keep references to the upstream SMT dialect unambiguous inside LLZK's own
// `llzk::smt` pass namespace.
namespace smt = mlir::smt;

/// Theory-neutral primitive emitter interface used by non-native encoders.
class NonNativeTheoryEmitter {
public:
  virtual ~NonNativeTheoryEmitter() = default;

  virtual std::pair<mlir::Value, mlir::Value> getRangeBoundAssertions(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
      const UnreducedInterval &range
  ) const = 0;

  virtual void emitRangeConstraint(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
      const UnreducedInterval &range
  ) const = 0;

  virtual mlir::Value
  emitFreshSymbol(mlir::OpBuilder &builder, mlir::Location loc, mlir::StringRef name) const = 0;

  virtual mlir::Value emitConstant(
      mlir::OpBuilder &builder, mlir::Location loc, const llvm::DynamicAPInt &value
  ) const = 0;

  virtual mlir::Value
  emitSub(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const = 0;

  virtual mlir::Value
  emitAdd(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const = 0;

  virtual mlir::Value
  emitMul(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const = 0;

  virtual mlir::Value
  emitDiv(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const = 0;

  virtual mlir::Value emitSignedDiv(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs
  ) const = 0;

  virtual mlir::Value emitSignedRem(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs
  ) const = 0;

  virtual mlir::Value
  emitModPrime(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value) const = 0;

  virtual mlir::Value
  emitPrimeMultiple(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value factor) const = 0;

  virtual mlir::Value emitOrderedComparison(
      mlir::OpBuilder &builder, mlir::Location loc, boolean::FeltCmpPredicate predicate,
      mlir::Value lhs, mlir::Value rhs
  ) const = 0;
};

/// Emit primitive integer-theory terms for the optimized non-native encoding.
///
/// This layer only builds integer-sorted values and
/// arithmetic fragments. Higher-level non-native encoding structure lives above
/// this emitter.
class SMTIntTheoryEmitter : public NonNativeTheoryEmitter {
private:
  mlir::MLIRContext *ctx;
  llvm::APSInt prime;
  // `freshSymbolCounts` is a map to improve readability. We could just have a counter.
  mutable llvm::StringMap<unsigned> freshSymbolCounts;

public:
  SMTIntTheoryEmitter(mlir::MLIRContext *context, const llvm::APSInt &smtPrime)
      : ctx(context), prime(smtPrime) {}

  std::pair<mlir::Value, mlir::Value> getRangeBoundAssertions(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
      const UnreducedInterval &range
  ) const override;

  void emitRangeConstraint(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
      const UnreducedInterval &range
  ) const override;

  mlir::Value emitFreshSymbol(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::StringRef name
  ) const override;

  /// Declare a fresh SMT array with the requested type.
  mlir::Value emitFreshArray(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::StringRef name, mlir::smt::ArrayType type
  ) const;

  mlir::Value emitConstant(
      mlir::OpBuilder &builder, mlir::Location loc, const llvm::DynamicAPInt &value
  ) const override;

  mlir::Value emitSub(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs
  ) const override;

  mlir::Value emitAdd(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs
  ) const override;

  mlir::Value emitMul(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs
  ) const override;

  mlir::Value emitDiv(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs
  ) const override;

  mlir::Value emitSignedDiv(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs
  ) const override;

  mlir::Value emitSignedRem(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs
  ) const override;

  mlir::Value
  emitModPrime(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value) const override;
  mlir::Value emitPrimeMultiple(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value factor
  ) const override;

  mlir::Value emitOrderedComparison(
      mlir::OpBuilder &builder, mlir::Location loc, boolean::FeltCmpPredicate predicate,
      mlir::Value lhs, mlir::Value rhs
  ) const override;

  /// Lower `arr[i, j, k]` to nested SMT array selections.
  mlir::Value emitArraySelect(
      mlir::Location loc, mlir::Value array, mlir::ValueRange indices, mlir::OpBuilder &builder
  );

  /// Functionally update a nested SMT array at the given rank-N index.
  ///
  /// The leaf is updated first, then each containing array is rebuilt with an
  /// inside-out sequence of `smt.array.store` operations.
  mlir::Value emitArrayStore(
      mlir::Location loc, mlir::Value array, mlir::ValueRange indices, mlir::Value value,
      mlir::OpBuilder &builder
  );

  // forall x, inbounds(x, arr) => phi(x)
  mlir::Value emitQuantifiedAssertion(
      mlir::Location loc, mlir::ArrayRef<size_t> extents,
      llvm::function_ref<mlir::Value(mlir::ValueRange)> body, mlir::OpBuilder &builder
  );

private:
  /// |value| = if value < 0 then -value else value
  mlir::Value emitAbsValue(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value) const;

  /// absQuotient = |lhs| / |rhs|
  /// quotient = if sign(lhs) != sign(rhs) then -absQuotient else absQuotient
  mlir::Value emitTruncatingSignedDivision(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs
  ) const;

  std::string getFreshName(mlir::StringRef baseName) const;

  smt::IntConstantOp createPrimeConstant(mlir::OpBuilder &builder, mlir::Location loc) const;

  smt::IntConstantOp createIntConstant(
      mlir::OpBuilder &builder, mlir::Location loc, const llvm::DynamicAPInt &value
  ) const;
};

bool isFeltOrArrayOfFelt(mlir::Type type);
llvm::FailureOr<llvm::SmallVector<size_t>> getExtents(array::ArrayType type);

enum class ArrayWriteMode : std::uint8_t {
  Overwrite, /* Multiple writes to the same index clobber the previous value */
  WriteOnce  /* Assume each array index can only be written to once */
};

using SignalSymbols = llvm::DenseMap<llvm::StringRef, std::pair<mlir::Value, mlir::Value>>;
using ArrayWritePolicy = std::function<ArrayWriteMode(mlir::Value)>;

/// The canonical identity and initial value of one overwrite-enabled array.
///
/// The key is an SSA value for local allocations today. Keeping it behind the
/// resolver boundary allows a future policy to use a canonical member/storage
/// identity instead of treating each member-read SSA result as distinct state.
struct ArrayStateRoot {
  mlir::Value key;
  mlir::Value initialValue;
  mlir::Location diagnosticLoc;
  mlir::Block *stateBlock;
};

/// The policy-selected semantics and optional mutable state root for an array.
struct ResolvedArraySemantics {
  ArrayWriteMode mode;
  std::optional<ArrayStateRoot> overwriteRoot;
};

/// Resolves array provenance into write semantics and a logical state root.
class ArraySemanticsResolver {
  ArrayWritePolicy policy;

public:
  explicit ArraySemanticsResolver(ArrayWritePolicy writePolicy) : policy(std::move(writePolicy)) {}

  mlir::FailureOr<ResolvedArraySemantics> resolve(mlir::Value array) const;
};

/// Prevalidated straight-line lowering state shared by array access patterns.
///
/// The state groups accesses by their resolver-provided logical identity and
/// records them by walking blocks in source order. It deliberately does not use
/// SSA use-list order or conversion-pattern visitation order. Each overwrite
/// chain is lowered as a single rewrite transaction, while write-once accesses
/// continue through the existing assertion-based path.
class ArrayLoweringState {
  struct Record {
    ArrayStateRoot root;
    llvm::SmallVector<mlir::Operation *> accesses;
    bool processed = false;
  };

  llvm::SmallVector<Record> records;
  llvm::DenseMap<mlir::Value, unsigned> recordByKey;
  llvm::DenseMap<mlir::Operation *, unsigned> overwriteAccesses;

  mlir::LogicalResult initialize(mlir::Operation *scope, const ArraySemanticsResolver &resolver);

public:
  static mlir::FailureOr<std::unique_ptr<ArrayLoweringState>>
  create(mlir::Operation *scope, const ArraySemanticsResolver &resolver);

  bool isOverwriteAccess(mlir::Operation *access) const;

  mlir::LogicalResult lowerOverwriteChain(
      mlir::Operation *trigger, mlir::ConversionPatternRewriter &rewriter,
      SMTIntTheoryEmitter &emitter
  );
};

mlir::FailureOr<FieldRef> resolveSelectedField(mlir::ModuleOp mod, llvm::StringRef fieldName);

class LLZKToSMTTypeConverter : public mlir::TypeConverter {
public:
  explicit LLZKToSMTTypeConverter(mlir::MLIRContext *ctx);
};

bool containsFeltOrStruct(mlir::Type type);

mlir::Operation *convertStructProductToFunc(mlir::Operation *op, mlir::MLIRContext *context);

void configureSMTNoCFBodyConversionTarget(mlir::ConversionTarget &target);

/// Add optimized-only legality requirements for local array lowering.
void configureSMTOverwriteArrayConversionTarget(mlir::ConversionTarget &target);

mlir::Operation *applySMTNoCFBodyConversion(
    mlir::Operation *op, mlir::ConversionTarget &target, mlir::RewritePatternSet &&patterns
);

template <class From, class To> class BasicConverter : public mlir::OpConversionPattern<From> {
  using mlir::OpConversionPattern<From>::OpConversionPattern;

public:
  mlir::LogicalResult matchAndRewrite(
      From fromOp, typename From::Adaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override {
    rewriter.template replaceOpWithNewOp<To>(fromOp, adaptor.getOperands());
    return mlir::success();
  }
};

class FunctionDefConverter : public mlir::OpConversionPattern<function::FuncDefOp> {
  using mlir::OpConversionPattern<function::FuncDefOp>::OpConversionPattern;

public:
  mlir::LogicalResult matchAndRewrite(
      function::FuncDefOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

class MemberReadConverter : public mlir::OpConversionPattern<component::MemberReadOp> {
public:
  MemberReadConverter(
      mlir::TypeConverter &converter, mlir::MLIRContext *context, const SignalSymbols &signalMap
  );

  mlir::LogicalResult matchAndRewrite(
      component::MemberReadOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;

private:
  SignalSymbols symbols;
};

class StructDefConverter : public mlir::OpConversionPattern<component::StructDefOp> {
  using mlir::OpConversionPattern<component::StructDefOp>::OpConversionPattern;

public:
  mlir::LogicalResult matchAndRewrite(
      component::StructDefOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

class ReturnConverter : public mlir::OpConversionPattern<function::ReturnOp> {
  using mlir::OpConversionPattern<function::ReturnOp>::OpConversionPattern;

public:
  mlir::LogicalResult matchAndRewrite(
      function::ReturnOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

class SCFIfConverter : public mlir::OpConversionPattern<mlir::scf::IfOp> {
  using mlir::OpConversionPattern<mlir::scf::IfOp>::OpConversionPattern;

public:
  mlir::LogicalResult matchAndRewrite(
      mlir::scf::IfOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

class YieldConverter : public mlir::OpConversionPattern<mlir::scf::YieldOp> {
  using mlir::OpConversionPattern<mlir::scf::YieldOp>::OpConversionPattern;

public:
  mlir::LogicalResult matchAndRewrite(
      mlir::scf::YieldOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

class FeltConstConverter : public mlir::OpConversionPattern<felt::FeltConstantOp> {
  using mlir::OpConversionPattern<felt::FeltConstantOp>::OpConversionPattern;

public:
  mlir::LogicalResult matchAndRewrite(
      felt::FeltConstantOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

/// Lower felt negation to upstream SMT subtraction from an explicit zero.
class FeltNegConverter : public mlir::OpConversionPattern<felt::NegFeltOp> {
  using mlir::OpConversionPattern<felt::NegFeltOp>::OpConversionPattern;

public:
  mlir::LogicalResult matchAndRewrite(
      felt::NegFeltOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override {
    auto zero = mlir::smt::IntConstantOp::create(
        rewriter, op.getLoc(),
        mlir::IntegerAttr::get(getContext(), llvm::APSInt(llvm::APInt(1, 0), false))
    );
    rewriter.replaceOpWithNewOp<mlir::smt::IntSubOp>(op, zero.getResult(), adaptor.getOperand());
    return mlir::success();
  }
};

class IndexConstConverter : public mlir::OpConversionPattern<mlir::arith::ConstantIndexOp> {
  using mlir::OpConversionPattern<mlir::arith::ConstantIndexOp>::OpConversionPattern;

  SMTIntTheoryEmitter *emitter;

public:
  IndexConstConverter(
      mlir::TypeConverter &converter, mlir::MLIRContext *context, SMTIntTheoryEmitter *emitter
  );
  mlir::LogicalResult matchAndRewrite(
      mlir::arith::ConstantIndexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

/// Convert a fresh LLZK allocation to a uniquely named functional SMT array.
class CreateArrayConverter : public mlir::OpConversionPattern<array::CreateArrayOp> {
  using mlir::OpConversionPattern<array::CreateArrayOp>::OpConversionPattern;

  SMTIntTheoryEmitter *emitter;

public:
  CreateArrayConverter(
      mlir::TypeConverter &converter, mlir::MLIRContext *context, SMTIntTheoryEmitter *emitter
  );
  mlir::LogicalResult matchAndRewrite(
      array::CreateArrayOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

/// Fold a statically known LLZK array dimension to an SMT integer constant.
class ArrayLengthConverter : public mlir::OpConversionPattern<array::ArrayLengthOp> {
  using mlir::OpConversionPattern<array::ArrayLengthOp>::OpConversionPattern;

  SMTIntTheoryEmitter *emitter;

public:
  ArrayLengthConverter(
      mlir::TypeConverter &converter, mlir::MLIRContext *context, SMTIntTheoryEmitter *emitter
  );
  mlir::LogicalResult matchAndRewrite(
      array::ArrayLengthOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

class WriteArrayConverter : public mlir::OpConversionPattern<array::WriteArrayOp> {
  using mlir::OpConversionPattern<array::WriteArrayOp>::OpConversionPattern;

  ArrayLoweringState *state;
  SMTIntTheoryEmitter *emitter;

public:
  WriteArrayConverter(
      mlir::TypeConverter &converter, mlir::MLIRContext *context, ArrayLoweringState *state,
      SMTIntTheoryEmitter *emitter
  );

  mlir::LogicalResult matchAndRewrite(
      array::WriteArrayOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

class ReadArrayConverter : public mlir::OpConversionPattern<array::ReadArrayOp> {
  using mlir::OpConversionPattern<array::ReadArrayOp>::OpConversionPattern;

  ArrayLoweringState *state;
  SMTIntTheoryEmitter *emitter;

public:
  ReadArrayConverter(
      mlir::TypeConverter &converter, mlir::MLIRContext *context, ArrayLoweringState *state,
      SMTIntTheoryEmitter *emitter
  );
  mlir::LogicalResult matchAndRewrite(
      array::ReadArrayOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;
};

} // namespace llzk::smt::detail
