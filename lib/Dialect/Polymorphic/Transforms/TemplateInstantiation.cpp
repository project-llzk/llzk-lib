//===-- TemplateInstantiation.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Private parameter substitution helpers shared by polymorphic transformations.
/// Callers own cloning, symbol insertion, specialization identity and scheduling.
///
//===----------------------------------------------------------------------===//

#include "TemplateInstantiation.h"

#include "SharedImpl.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>

#include <llvm/ADT/TypeSwitch.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "llzk-flatten"

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::constrain;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::polymorphic;
using namespace llzk::polymorphic::detail;

namespace {

template <typename Impl, typename Op, typename... HandledAttrs>
class SymbolUserHelper : public OpConversionPattern<Op> {
private:
  const DenseMap<Attribute, Attribute> &paramNameToValue;

  SymbolUserHelper(
      TypeConverter &converter, MLIRContext *ctx, unsigned patternBenefit,
      const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue
  )
      : OpConversionPattern<Op>(converter, ctx, patternBenefit),
        paramNameToValue(paramNameToInstantiatedValue) {}

public:
  using OpAdaptor = typename mlir::OpConversionPattern<Op>::OpAdaptor;

  virtual Attribute getNameAttr(Op) const = 0;

  virtual LogicalResult handleDefaultRewrite(
      Attribute, Op op, OpAdaptor, ConversionPatternRewriter &, Attribute a
  ) const {
    return op->emitOpError().append("expected value with type ", op.getType(), " but found ", a);
  }

  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "[SymbolUserHelper] op: " << op << '\n');
    auto res = this->paramNameToValue.find(getNameAttr(op));
    if (res == this->paramNameToValue.end()) {
      LLVM_DEBUG(llvm::dbgs() << "[SymbolUserHelper] no instantiation for " << op << '\n');
      return failure();
    }
    llvm::TypeSwitch<Attribute, LogicalResult> TS(res->second);
    llvm::TypeSwitch<Attribute, LogicalResult> *ptr = &TS;

    ((ptr = &(ptr->template Case<HandledAttrs>([&](HandledAttrs a) {
      return static_cast<const Impl *>(this)->handleRewrite(res->first, op, adaptor, rewriter, a);
    }))),
     ...);

    return TS.Default([&](Attribute a) {
      return handleDefaultRewrite(res->first, op, adaptor, rewriter, a);
    });
  }
  friend Impl;
};

class ClonedBodyConstReadOpPattern
    : public SymbolUserHelper<
          ClonedBodyConstReadOpPattern, ConstReadOp, IntegerAttr, FeltConstAttr> {
  SmallVector<Diagnostic> &diagnostics;

  using super =
      SymbolUserHelper<ClonedBodyConstReadOpPattern, ConstReadOp, IntegerAttr, FeltConstAttr>;

public:
  ClonedBodyConstReadOpPattern(
      TypeConverter &converter, MLIRContext *ctx,
      const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue,
      SmallVector<Diagnostic> &instantiationDiagnostics
  )
      // benefit>0 so this applies instead of GeneralTypeReplacePattern<ConstReadOp>
      : super(converter, ctx, /*patternBenefit=*/1, paramNameToInstantiatedValue),
        diagnostics(instantiationDiagnostics) {}

  Attribute getNameAttr(ConstReadOp op) const override { return op.getConstNameAttr(); }

  LogicalResult handleRewrite(
      Attribute sym, ConstReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter, IntegerAttr a
  ) const {
    const APInt &attrValue = a.getValue();
    Type origResTy = op.getType();
    Type newResTy = getTypeConverter()->convertType(origResTy);
    if (!newResTy) {
      return op->emitOpError().append("could not convert result type ", origResTy);
    }

    if (FeltType ty = llvm::dyn_cast<FeltType>(newResTy)) {
      replaceOpWithNewOp<FeltConstantOp>(
          rewriter, op, FeltConstAttr::get(getContext(), attrValue, ty)
      );
      return success();
    }

    if (llvm::isa<IndexType>(newResTy)) {
      replaceOpWithNewOp<arith::ConstantIndexOp>(rewriter, op, fromAPInt(attrValue));
      return success();
    }

    if (newResTy.isSignlessInteger(1)) {
      // Treat 0 as false and any other value as true (but give a warning if it's not 1)
      if (attrValue.isZero()) {
        replaceOpWithNewOp<arith::ConstantIntOp>(rewriter, op, newResTy, false);
        return success();
      }
      if (!attrValue.isOne()) {
        Location opLoc = op.getLoc();
        Diagnostic diag(opLoc, DiagnosticSeverity::Warning);
        diag << "Interpreting non-zero value " << stringWithoutType(a) << " as true";
        if (getContext()->shouldPrintOpOnDiagnostic()) {
          diag.attachNote(opLoc) << "see current operation: " << *op;
        }
        diag.attachNote(UnknownLoc::get(getContext()))
            << "when instantiating '" << StructDefOp::getOperationName() << "' parameter \"" << sym
            << "\" for this call";
        diagnostics.push_back(std::move(diag));
      }
      replaceOpWithNewOp<arith::ConstantIntOp>(rewriter, op, newResTy, true);
      return success();
    }
    return op->emitOpError().append("unexpected result type ", newResTy);
  }

  LogicalResult handleRewrite(
      Attribute, ConstReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter, FeltConstAttr a
  ) const {
    replaceOpWithNewOp<FeltConstantOp>(rewriter, op, a);
    return success();
  }
};

static inline bool tableOffsetIsntSymbol(MemberReadOp op) {
  return !llvm::isa_and_present<SymbolRefAttr>(op.getTableOffset().value_or(nullptr));
}

/// Materialize symbolic member table offsets only from integer template bindings. Member tables are
/// index-addressed, so other concrete attribute kinds emit diagnostics instead of being coerced.
class ClonedMemberReadOpPattern
    : public SymbolUserHelper<ClonedMemberReadOpPattern, MemberReadOp, IntegerAttr> {
  using super = SymbolUserHelper<ClonedMemberReadOpPattern, MemberReadOp, IntegerAttr>;

public:
  ClonedMemberReadOpPattern(
      TypeConverter &converter, MLIRContext *ctx,
      const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue
  )
      // benefit>0 so this applies instead of GeneralTypeReplacePattern<MemberReadOp>
      : super(converter, ctx, /*patternBenefit=*/1, paramNameToInstantiatedValue) {}

  Attribute getNameAttr(MemberReadOp op) const override {
    return op.getTableOffset().value_or(nullptr);
  }

  LogicalResult handleRewrite(
      Attribute, MemberReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter, IntegerAttr a
  ) const {
    rewriter.modifyOpInPlace(op, [&]() {
      op.setTableOffsetAttr(rewriter.getIndexAttr(fromAPInt(a.getValue())));
    });

    return success();
  }

  LogicalResult handleDefaultRewrite(
      Attribute, MemberReadOp op, OpAdaptor, ConversionPatternRewriter &, Attribute a
  ) const override {
    return op->emitOpError().append(
        "table offset requires an integer template value, but found ", a
    );
  }

  LogicalResult matchAndRewrite(
      MemberReadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    LLVM_DEBUG(llvm::dbgs() << "[ClonedMemberReadOpPattern]   MemberReadOp: " << op << '\n';);
    if (tableOffsetIsntSymbol(op)) {
      return failure();
    }

    return super::matchAndRewrite(op, adaptor, rewriter);
  }
};

class MappedTypeConverter : public TypeConverter {
  StructType origTy;
  StructType newTy;
  const DenseMap<Attribute, Attribute> &paramNameToValue;

  inline Attribute convertIfPossible(Attribute a) const {
    auto res = this->paramNameToValue.find(a);
    return (res != this->paramNameToValue.end()) ? res->second : a;
  }

public:
  MappedTypeConverter(
      StructType originalType, StructType newType,
      /// Instantiated values for the parameter names in `originalType`
      const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue
  )
      : TypeConverter(), origTy(originalType), newTy(newType),
        paramNameToValue(paramNameToInstantiatedValue) {

    addConversion([](Type inputTy) { return inputTy; });

    addConversion([this](StructType inputTy) {
      LLVM_DEBUG(llvm::dbgs() << "[MappedTypeConverter] convert " << inputTy << '\n');

      // Check for replacement of the full type
      if (inputTy == this->origTy) {
        return this->newTy;
      }
      // Check for replacement of parameter symbol names with concrete values
      if (ArrayAttr inputTyParams = inputTy.getParams()) {
        SmallVector<Attribute> updated;
        for (Attribute a : inputTyParams) {
          if (TypeAttr ta = dyn_cast<TypeAttr>(a)) {
            updated.push_back(TypeAttr::get(this->convertType(ta.getValue())));
          } else {
            updated.push_back(convertIfPossible(a));
          }
        }
        return getStructTypeWithParams(inputTy.getNameRef(), inputTy.getContext(), updated);
      }
      // Otherwise, return the type unchanged
      return inputTy;
    });

    addConversion([this](ArrayType inputTy) {
      // Check for replacement of parameter symbol names with concrete values
      ArrayRef<Attribute> dimSizes = inputTy.getDimensionSizes();
      if (!dimSizes.empty()) {
        SmallVector<Attribute> updated;
        for (Attribute a : dimSizes) {
          updated.push_back(convertIfPossible(a));
        }
        return ArrayType::get(this->convertType(inputTy.getElementType()), updated);
      }
      // Otherwise, return the type unchanged
      return inputTy;
    });

    addConversion([this](TypeVarType inputTy) -> Type {
      // Check for replacement of parameter symbol name with a concrete type
      if (TypeAttr tyAttr = llvm::dyn_cast<TypeAttr>(convertIfPossible(inputTy.getNameRef()))) {
        Type convertedType = tyAttr.getValue();
        // Use the new type unless it contains a TypeVarType because a TypeVarType from a
        // different struct references a parameter name from that other struct, not from the
        // current struct so the reference would be invalid.
        if (isConcreteType(convertedType)) {
          return convertedType;
        }
      }
      return inputTy;
    });
  }
};

/// Rewrite cloned scalar array reads to ranged extract ops when a wildcard element type
/// resolves to a higher-rank array.
class ClonedBodyArrayReadOpPattern final : public OpConversionPattern<ReadArrayOp> {
public:
  using OpConversionPattern<ReadArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ReadArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    Type newResultTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!llvm::isa<ArrayType>(newResultTy)) {
      return failure();
    }
    replaceOpWithNewOp<ExtractArrayOp>(
        rewriter, op, newResultTy, adaptor.getArrRef(), adaptor.getIndices()
    );
    return success();
  }
};

/// Rewrite cloned scalar array writes to ranged inserts when a wildcard element type
/// resolves to a higher-rank array.
class ClonedBodyArrayWriteOpPattern final : public OpConversionPattern<WriteArrayOp> {
public:
  using OpConversionPattern<WriteArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      WriteArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (!llvm::isa<ArrayType>(adaptor.getRvalue().getType())) {
      return failure();
    }
    replaceOpWithNewOp<InsertArrayOp>(
        rewriter, op, adaptor.getArrRef(), adaptor.getIndices(), adaptor.getRvalue()
    );
    return success();
  }
};

static SymbolRefAttr
convertCalleeSymRefs(SymbolRefAttr callee, const DenseMap<Attribute, Attribute> &paramNameToValue) {
  auto it = paramNameToValue.find(FlatSymbolRefAttr::get(callee.getRootReference()));
  if (it == paramNameToValue.end()) {
    return callee;
  }

  auto tyAttr = llvm::dyn_cast<TypeAttr>(it->second);
  if (!tyAttr) {
    return callee;
  }

  auto structTy = llvm::dyn_cast<StructType>(tyAttr.getValue());
  if (!structTy) {
    return callee;
  }

  SmallVector<FlatSymbolRefAttr> newPieces = getPieces(structTy.getNameRef());
  llvm::append_range(newPieces, callee.getNestedReferences());
  return asSymbolRefAttr(newPieces);
}

/// Attempt to evaluate the concrete result of a single `TemplateExprOp` expression given
/// the currently-known concrete param values in `paramNameToConcrete`. Returns the result
/// attribute if all referenced params are concrete and all operations in the body can be
/// constant-folded; otherwise returns `std::nullopt`.
static std::optional<Attribute>
evaluateExpr(TemplateExprOp exprOp, const DenseMap<Attribute, Attribute> &paramNameToConcrete) {
  // Map from SSA value in the expr body to its concrete Attribute.
  DenseMap<Value, Attribute> valueMap;
  for (Operation &bodyOp : exprOp.getInitializerRegion().front()) {
    if (auto yieldOp = llvm::dyn_cast<YieldOp>(bodyOp)) {
      auto it = valueMap.find(yieldOp.getVal());
      return it != valueMap.end() ? std::make_optional(it->second) : std::nullopt;
    }

    if (auto constReadOp = llvm::dyn_cast<ConstReadOp>(bodyOp)) {
      auto it = paramNameToConcrete.find(constReadOp.getConstNameAttr());
      if (it == paramNameToConcrete.end()) {
        return std::nullopt; // a referenced param is not concrete
      }
      // If the attribute type is `FeltType` but it's stored as an IntegerAttr, promote to
      // a `FeltConstAttr`.
      Attribute val = it->second;
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(val)) {
        if (auto feltTy = llvm::dyn_cast<FeltType>(constReadOp.getResult().getType())) {
          val = FeltConstAttr::get(bodyOp.getContext(), intAttr.getValue(), feltTy);
        }
      }
      valueMap[constReadOp.getResult()] = val;
      continue;
    }

    // Gather constant attributes for all operands.
    SmallVector<Attribute> operandAttrs;
    operandAttrs.reserve(bodyOp.getNumOperands());
    for (Value operand : bodyOp.getOperands()) {
      auto it = valueMap.find(operand);
      if (it == valueMap.end()) {
        return std::nullopt; // operand not known as a constant
      }
      operandAttrs.push_back(it->second);
    }

    // Try constant folding.
    SmallVector<OpFoldResult> foldResults;
    if (succeeded(bodyOp.fold(operandAttrs, foldResults)) &&
        foldResults.size() == bodyOp.getNumResults()) {
      for (auto [result, fr] : llvm::zip_equal(bodyOp.getResults(), foldResults)) {
        if (Attribute a = llvm::dyn_cast<Attribute>(fr)) {
          valueMap[result] = a;
        } else {
          return std::nullopt;
        }
      }
    }
  }
  return std::nullopt; // no YieldOp found (shouldn't happen in a valid expr)
}

} // namespace

namespace llzk::polymorphic::detail {

Attribute TemplateTypeConverter::convertIfPossible(Attribute attr) const {
  auto it = paramNameToValue.find(attr);
  return it == paramNameToValue.end() ? attr : it->second;
}

TemplateTypeConverter::TemplateTypeConverter(DenseMap<Attribute, Attribute> bindings)
    : paramNameToValue(std::move(bindings)) {
  addConversion([](Type t) { return t; });

  addConversion([this](TypeVarType inputTy) -> Type {
    if (TypeAttr tyAttr = llvm::dyn_cast<TypeAttr>(convertIfPossible(inputTy.getNameRef()))) {
      Type convertedType = tyAttr.getValue();
      if (isConcreteType(convertedType)) {
        return convertedType;
      }
    }
    return inputTy;
  });

  addConversion([this](ArrayType inputTy) {
    SmallVector<Attribute> updated;
    bool changed = false;
    for (Attribute a : inputTy.getDimensionSizes()) {
      Attribute converted = convertIfPossible(a);
      updated.push_back(converted);
      if (converted != a) {
        changed = true;
      }
    }
    Type newElemTy = this->convertType(inputTy.getElementType());
    if (!changed && newElemTy == inputTy.getElementType()) {
      return inputTy;
    }
    return flattenArrayElementType(inputTy.cloneWith(inputTy.getElementType(), updated), newElemTy);
  });

  addConversion([this](StructType inputTy) -> StructType {
    if (ArrayAttr params = inputTy.getParams()) {
      SmallVector<Attribute> updated;
      bool changed = false;
      for (Attribute a : params) {
        if (TypeAttr ta = dyn_cast<TypeAttr>(a)) {
          Type newTy = this->convertType(ta.getValue());
          if (newTy != ta.getValue()) {
            updated.push_back(TypeAttr::get(newTy));
            changed = true;
            continue;
          }
        } else {
          Attribute converted = convertIfPossible(a);
          if (converted != a) {
            updated.push_back(converted);
            changed = true;
            continue;
          }
        }
        updated.push_back(a);
      }
      if (changed) {
        return getStructTypeWithParams(inputTy.getNameRef(), inputTy.getContext(), updated);
      }
    }
    return inputTy;
  });
}

Attribute TemplateTypeConverter::convertAttr(Attribute attr) const {
  if (TypeAttr tyAttr = llvm::dyn_cast<TypeAttr>(attr)) {
    Type convertedTy = convertType(tyAttr.getValue());
    if (convertedTy != tyAttr.getValue()) {
      return TypeAttr::get(convertedTy);
    }
  }
  return convertIfPossible(attr);
}

void convertCalleesInPlace(Operation *op, const DenseMap<Attribute, Attribute> &paramNameToValue) {
  op->walk([&paramNameToValue](CallOp callOp) {
    callOp.setCalleeAttr(convertCalleeSymRefs(callOp.getCalleeAttr(), paramNameToValue));
  });
}

/// Evaluate all `TemplateExprOp`s in `templateOp` that can be computed from the currently-known
/// concrete param values in `paramNameToConcrete`, and add their results to the map.
/// Exprs whose operands are not all concrete are silently skipped (partial instantiation).
void evaluateTemplateExprs(
    TemplateOp templateOp, DenseMap<Attribute, Attribute> &paramNameToConcrete
) {
  LLVM_DEBUG(
      llvm::dbgs() << "[evaluateTemplateExprs] before: " << debug::toStringList(paramNameToConcrete)
                   << '\n'
  );
  for (TemplateExprOp exprOp : templateOp.getConstOps<TemplateExprOp>()) {
    std::optional<Attribute> result = evaluateExpr(exprOp, paramNameToConcrete);
    if (result.has_value()) {
      auto exprNameAttr = FlatSymbolRefAttr::get(exprOp.getSymNameAttr());
      paramNameToConcrete.try_emplace(exprNameAttr, *result);
      LLVM_DEBUG(
          llvm::dbgs() << "[evaluateTemplateExprs] expr @" << exprOp.getSymName()
                       << " evaluated to " << *result << '\n'
      );
    }
  }
  LLVM_DEBUG(
      llvm::dbgs() << "[evaluateTemplateExprs] after: " << debug::toStringList(paramNameToConcrete)
                   << '\n'
  );
}

/// Return the callee-side unification-derived value for a template parameter, if any.
std::optional<Attribute>
inferUnifiedParam(const UnificationMap &unifyResult, SymbolRefAttr paramName) {
  auto it = unifyResult.find({paramName, Side::RHS});
  return (it == unifyResult.end()) ? std::nullopt : std::make_optional(it->second);
}

LogicalResult substituteStructBody(
    StructDefOp newStruct, StructType typeAtDef,
    const DenseMap<Attribute, Attribute> &paramNameToConcrete, SmallVector<Diagnostic> &diagnostics
) {
  MLIRContext *ctx = newStruct.getContext();
  MappedTypeConverter tyConv(typeAtDef, newStruct.getType(), paramNameToConcrete);
  ConversionTarget target =
      newConverterDefinedTarget<EmitEqualityOp>(tyConv, ctx, tableOffsetIsntSymbol);
  target.addDynamicallyLegalOp<ConstReadOp>([&paramNameToConcrete](ConstReadOp op) {
    // Legal if it's not in the map of concrete attribute instantiations
    return !paramNameToConcrete.contains(op.getConstNameAttr());
  });

  RewritePatternSet patterns = newGeneralRewritePatternSet<EmitEqualityOp>(tyConv, ctx, target);
  patterns.add<ClonedBodyConstReadOpPattern>(tyConv, ctx, paramNameToConcrete, diagnostics);
  patterns.add<ClonedMemberReadOpPattern>(tyConv, ctx, paramNameToConcrete);
  return applyFullConversion(newStruct, target, std::move(patterns));
}

LogicalResult substituteFunctionBody(
    FuncDefOp newFunc, const DenseMap<Attribute, Attribute> &paramNameToConcrete,
    SmallVector<Diagnostic> &delayedDiagnostics
) {
  MLIRContext *ctx = newFunc.getContext();
  TemplateTypeConverter tyConv(paramNameToConcrete);
  ConversionTarget target = newConverterDefinedTarget<>(tyConv, ctx, tableOffsetIsntSymbol);
  target.addDynamicallyLegalOp<ConstReadOp>([&tyConv](ConstReadOp p) {
    // Legal if it's not in the map of concrete attribute instantiations
    return !tyConv.containsParam(p.getConstNameAttr());
  });
  RewritePatternSet bodyPatterns = newGeneralRewritePatternSet(tyConv, ctx, target);
  bodyPatterns.add<ClonedBodyConstReadOpPattern>(
      tyConv, ctx, tyConv.getParamMap(), delayedDiagnostics
  );
  bodyPatterns.add<ClonedBodyArrayReadOpPattern, ClonedBodyArrayWriteOpPattern>(tyConv, ctx);
  bodyPatterns.add<ClonedMemberReadOpPattern>(tyConv, ctx, paramNameToConcrete);
  return applyFullConversion(newFunc, target, std::move(bodyPatterns));
}

} // namespace llzk::polymorphic::detail
