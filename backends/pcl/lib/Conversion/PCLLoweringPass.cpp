//===-- PCLLoweringPass.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-to-pcl` pass.
///
//===----------------------------------------------------------------------===//

#include "pcl/Conversion/ConversionPasses.h"

#include "llzk/Config/Config.h"
#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Enums.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Dialect.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Dialect.h"
#include "llzk/Dialect/RAM/IR/Dialect.h"
#include "llzk/Dialect/SMT/IR/SMTDialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Verif/IR/Dialect.h"
#include "llzk/Transforms/LLZKLoweringUtils.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/Field.h"

#include <pcl/Dialect/IR/Dialect.h>
#include <pcl/Dialect/IR/Ops.h>
#include <pcl/Dialect/IR/Types.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/EquivalenceClasses.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>

#include <cstdint>
#include <optional>

// Include the generated base pass class definitions.
namespace pcl {
#define GEN_PASS_DECL_PCLLOWERINGPASS
#define GEN_PASS_DEF_PCLLOWERINGPASS
#include "pcl/Conversion/ConversionPasses.h.inc"
} // namespace pcl

using namespace mlir;
using namespace llzk;
using namespace llzk::cast;
using namespace llzk::boolean;
using namespace llzk::constrain;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::component;

namespace {

/// Returns a flat representation of the fully qualified name of the struct.
template <typename Op> static std::string flatStructName(Op op) {
  auto fqn = op.getFullyQualifiedName();
  std::string name;
  llvm::raw_string_ostream o(name);
  StringRef sep = "::";
  o << fqn.getRootReference().getValue();
  if (!fqn.getNestedReferences().empty()) {
    o << sep;
    llvm::interleave(fqn.getNestedReferences(), o, [&o](auto ref) { o << ref.getValue(); }, sep);
  }

  return name;
}

template <typename Op> class ConstantOpValue {};

template <> class ConstantOpValue<FeltConstantOp> {
protected:
  llvm::APInt getValue(FeltConstantOp op) const { return op.getValue().getValue(); }
};

template <> class ConstantOpValue<arith::ConstantOp> {
protected:
  llvm::APInt getValue(arith::ConstantOp op) const {
    // Extend width by 1 bit to avoid sign issues.
    auto value = mlir::cast<IntegerAttr>(op.getValue()).getValue();
    return value.zext(value.getBitWidth() + 1);
  }
};

/// Generic conversion pattern for lowering constants.
template <typename SrcOp>
class ConvertConstantOp : public OpConversionPattern<SrcOp>, ConstantOpValue<SrcOp> {
  using OpAdaptor = typename OpConversionPattern<SrcOp>::OpAdaptor;
  using ConstantOpValue<SrcOp>::getValue;

public:
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult match(SrcOp) const override { return success(); }

  void rewrite(SrcOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto value = pcl::FeltAttr::get(rewriter.getContext(), getValue(op));
    rewriter.replaceOpWithNewOp<pcl::ConstOp>(op, value);
  }
};

/// Generic conversion pattern for binary ops that have a 1:1 correspondence with a pcl op.
template <typename SrcOp, typename DstOp>
class ConvertBinaryOp : public OpConversionPattern<SrcOp> {
  using OpAdaptor = typename OpConversionPattern<SrcOp>::OpAdaptor;

  using OpConversionPattern<SrcOp>::getTypeConverter;
  using OpConversionPattern<SrcOp>::getContext;

public:
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult match(SrcOp) const override { return success(); }

  void rewrite(SrcOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<DstOp>(op, adaptor.getLhs(), adaptor.getRhs());
  }
};

/// Generic conversion pattern for unary ops that have a 1:1 correspondence with a pcl op.
template <typename SrcOp, typename DstOp> class ConvertUnaryOp : public OpConversionPattern<SrcOp> {
  using OpAdaptor = typename OpConversionPattern<SrcOp>::OpAdaptor;

public:
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult match(SrcOp) const override { return success(); }

  void rewrite(SrcOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<DstOp>(op, adaptor.getOperand());
  }
};

/// Converts a `bool.xor` into a negated `pcl.iff`.
struct ConvertBoolXorOp : public OpConversionPattern<XorBoolOp> {
  using OpConversionPattern<XorBoolOp>::OpConversionPattern;

  LogicalResult match(XorBoolOp) const override { return success(); }

  void
  rewrite(XorBoolOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto iffOp = rewriter.create<pcl::IffOp>(op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOpWithNewOp<pcl::NotOp>(op, iffOp);
  }
};

/// Removes `cast.tofelt` ops since all numerical types in pcl are the same.
struct RemoveIntToFeltOp : public OpConversionPattern<IntToFeltOp> {
  using OpConversionPattern<IntToFeltOp>::OpConversionPattern;

  LogicalResult match(IntToFeltOp) const override { return success(); }

  void
  rewrite(IntToFeltOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands());
  }
};

/// Converts `bool.cmp` ops into their pcl counterparts.
struct ConvertCmpOp : public OpConversionPattern<CmpOp> {
  using OpConversionPattern<CmpOp>::OpConversionPattern;

  LogicalResult match(CmpOp) const override { return success(); }

  void rewrite(CmpOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto pred = op.getPredicate();

    switch (pred) {
    case FeltCmpPredicate::EQ:
      rewriter.replaceOpWithNewOp<pcl::CmpEqOp>(op, adaptor.getLhs(), adaptor.getRhs());
      break;
    case FeltCmpPredicate::NE: {
      auto eqOp = rewriter.create<pcl::CmpEqOp>(op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOpWithNewOp<pcl::NotOp>(op, eqOp);
      break;
    }
    case FeltCmpPredicate::LT:
      rewriter.replaceOpWithNewOp<pcl::CmpLtOp>(op, adaptor.getLhs(), adaptor.getRhs());
      break;
    case FeltCmpPredicate::LE:
      rewriter.replaceOpWithNewOp<pcl::CmpLeOp>(op, adaptor.getLhs(), adaptor.getRhs());
      break;
    case FeltCmpPredicate::GT:
      rewriter.replaceOpWithNewOp<pcl::CmpGtOp>(op, adaptor.getLhs(), adaptor.getRhs());
      break;
    case FeltCmpPredicate::GE:
      rewriter.replaceOpWithNewOp<pcl::CmpGeOp>(op, adaptor.getLhs(), adaptor.getRhs());
      break;
    }
  }
};

/// Converts `constrain.eq` ops into an optimized `pcl.assert`.
///
/// XXX: The optimization should probably be defined as a canonicalization pattern of
/// that op instead to exploit it better.
class ConvertEmitEqualityOp : public OpConversionPattern<EmitEqualityOp> {
  bool isBool(Value v) const { return llvm::isa<pcl::BoolType>(v.getType()); }

  std::optional<llvm::APInt> getConstAPInt(Value v) const {
    if (auto c = llvm::dyn_cast_if_present<pcl::ConstOp>(v.getDefiningOp())) {
      return c.getValueAPInt();
    }
    return std::nullopt;
  }

  bool isConstOne(Value v) const {
    if (auto ap = getConstAPInt(v)) {
      return ap->isOne();
    }
    return false;
  }

  bool isConstZero(Value v) const {
    if (auto ap = getConstAPInt(v)) {
      return ap->isZero();
    }
    return false;
  }

public:
  using OpConversionPattern<EmitEqualityOp>::OpConversionPattern;

  LogicalResult match(EmitEqualityOp) const override { return success(); }

  void rewrite(
      EmitEqualityOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (isBool(adaptor.getLhs()) && isConstOne(adaptor.getRhs())) {
      rewriter.replaceOpWithNewOp<pcl::AssertOp>(op, adaptor.getLhs());
    } else if (isBool(adaptor.getRhs()) && isConstOne(adaptor.getLhs())) {
      rewriter.replaceOpWithNewOp<pcl::AssertOp>(op, adaptor.getRhs());
    } else if (isBool(adaptor.getLhs()) && isConstZero(adaptor.getRhs())) {
      auto notOp = rewriter.create<pcl::NotOp>(op.getLoc(), adaptor.getLhs());
      rewriter.replaceOpWithNewOp<pcl::AssertOp>(op, notOp);
    } else if (isBool(adaptor.getRhs()) && isConstZero(adaptor.getLhs())) {
      auto notOp = rewriter.create<pcl::NotOp>(op.getLoc(), adaptor.getRhs());
      rewriter.replaceOpWithNewOp<pcl::AssertOp>(op, notOp);
    } else {
      auto cmpEqOp = rewriter.create<pcl::CmpEqOp>(op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOpWithNewOp<pcl::AssertOp>(op, cmpEqOp);
    }
  }
};

/// Converts `bool.assert` ops into pcl asserts.
struct ConvertAssertOp : public OpConversionPattern<AssertOp> {
  using OpConversionPattern<AssertOp>::OpConversionPattern;

  LogicalResult match(AssertOp) const override { return success(); }

  void rewrite(AssertOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<pcl::AssertOp>(op, adaptor.getCondition());
  }
};

/// Converts `struct.readm` ops that read members of felt type from the struct into `pcl.var` ops.
struct ConvertSelfMemberReadOpOfFelt : public OpConversionPattern<MemberReadOp> {
  using OpConversionPattern<MemberReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MemberReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto parent = op->getParentOfType<FuncDefOp>();
    if (!parent || op.getComponent() != parent.getArgument(0)) {
      return failure();
    }
    SymbolTableCollection tables;
    auto defOp = op.getMemberDefOp(tables);
    if (failed(defOp)) {
      return failure();
    }
    if (!mlir::isa<FeltType>(defOp->get().getType())) {
      return failure();
    }

    auto pclVar = rewriter.create<pcl::VarOp>(
        defOp->get().getLoc(), defOp->get().getName(), defOp->get().hasPublicAttr()
    );
    rewriter.replaceOp(op, pclVar);

    return success();
  }
};

/// Removes `struct.readm` ops that read subcmp members from the struct.
struct ConvertSelfMemberReadOpOfSubcmp : public OpConversionPattern<MemberReadOp> {
  using OpConversionPattern<MemberReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MemberReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto parent = op->getParentOfType<FuncDefOp>();
    if (!parent || op.getComponent() != parent.getArgument(0)) {
      return failure();
    }
    SymbolTableCollection tables;
    auto defOp = op.getMemberDefOp(tables);
    if (failed(defOp)) {
      return failure();
    }
    if (!mlir::isa<StructType>(defOp->get().getType())) {
      return failure();
    }

    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertSubcmpMemberReadOp : public OpConversionPattern<MemberReadOp> {
  using OpConversionPattern<MemberReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MemberReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto subcmp = mlir::dyn_cast_if_present<MemberReadOp>(op.getComponent().getDefiningOp());
    if (!subcmp) {
      return failure();
    }
    auto parent = subcmp->getParentOfType<FuncDefOp>();
    if (!parent || subcmp.getComponent() != parent.getArgument(0)) {
      return failure();
    }
    SymbolTableCollection tables;
    auto defOp = op.getMemberDefOp(tables);
    if (failed(defOp)) {
      return failure();
    }

    auto name = (Twine(subcmp.getMemberName()) + "." + defOp->get().getName()).str();
    auto pclVar = rewriter.create<pcl::VarOp>(
        defOp->get().getLoc(), rewriter.getStringAttr(name), /*public=*/false
    );

    rewriter.replaceOp(op, pclVar);

    return success();
  }
};

/// Maps the list of struct members that are considered outputs for the pcl module.
///
/// A member is an output if it has the `llzk.pub` attribute and is of type `!felt.type`.
template <typename T, typename Fn> SmallVector<T> mapOutputMembers(StructDefOp op, Fn callback) {
  SmallVector<T> out;
  auto members = op.getMemberDefs();
  out.reserve(members.size());
  for (auto memberDef : members) {
    if (mlir::isa<FeltType>(memberDef.getType()) && memberDef.hasPublicAttr()) {
      out.push_back(callback(memberDef));
    }
  }
  return out;
}

/// Converts `function.return` ops inside `@constrain` functions into func return ops.
struct ConvertReturnOp : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto structDefOp = op->getParentOfType<StructDefOp>();
    if (!structDefOp) {
      return failure();
    }
    auto values = mapOutputMembers<Value>(structDefOp, [&rewriter](MemberDefOp memberDef) {
      return rewriter.create<pcl::VarOp>(memberDef.getLoc(), memberDef.getName(), /*public=*/true);
    });

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, values);
    return success();
  }
};

/// Converts `function.return` ops inside free functions into func return ops.
struct ConvertFreeFuncReturnOp : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto structDefOp = op->getParentOfType<StructDefOp>();
    if (structDefOp) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

using NonDetOpNames = llvm::DenseMap<NonDetOp, StringAttr>;

/// Converts `llzk.nondet` ops into fresh pcl variables.
class ConvertNonDetOp : public OpConversionPattern<NonDetOp> {
  NonDetOpNames &names;

public:
  ConvertNonDetOp(MLIRContext *context, NonDetOpNames &opNames, PatternBenefit patBenefit = 1)
      : OpConversionPattern(context, patBenefit), names(opNames) {}
  ConvertNonDetOp(
      const TypeConverter &tc, MLIRContext *context, NonDetOpNames &opNames,
      PatternBenefit patBenefit = 1
  )
      : OpConversionPattern(tc, context, patBenefit), names(opNames) {}

  LogicalResult match(NonDetOp op) const override { return success(names.find(op) != names.end()); }

  void rewrite(NonDetOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<pcl::VarOp>(op, names[op], /*public=*/false);
  }
};

/// Base class for patterns that copy the body of a function.
template <typename Op, typename Impl> struct BodyCopier {
private:
  BodyCopier() = default;

public:
  LogicalResult copyBody(Op op, ConversionPatternRewriter &rewriter) const {
    FailureOr<FuncDefOp> srcFuncOp = static_cast<const Impl *>(this)->getFuncOp(op);
    if (failed(srcFuncOp)) {
      return failure();
    }

    unsigned baseOffset = static_cast<const Impl *>(this)->baseOffset();

    SmallVector<Type> inputs(
        srcFuncOp->getNumArguments() - baseOffset, pcl::FeltType::get(rewriter.getContext())
    );
    SmallVector<Type> outputs = static_cast<const Impl *>(this)->outputTypes(op);

    auto funcOp = func::FuncOp::create(
        op.getLoc(), flatStructName(op), rewriter.getFunctionType(inputs, outputs)
    );
    funcOp.addEntryBlock();
    IRMapping mapping;
    for (auto arg : srcFuncOp->getArguments().take_front(baseOffset)) {
      mapping.map(arg, Value());
    }
    for (auto [srcArg, dstArg] :
         llvm::zip_equal(srcFuncOp->getArguments().drop_front(baseOffset), funcOp.getArguments())) {
      auto argType = srcArg.getType();
      if (!llvm::isa<FeltType>(argType)) {
        return srcFuncOp->emitError() << "function's args are expected to be felts. Found "
                                      << argType << "for arg #: " << srcArg.getArgNumber();
      }
      mapping.map(srcArg, dstArg);
    }

    if (!srcFuncOp->getBody().hasOneBlock()) {
      return srcFuncOp->emitError(
          "llzk-to-pcl conversion assumes the constrain function body has 1 block"
      );
    }
    rewriter.cloneRegionBefore(
        srcFuncOp->getRegion(), funcOp.getRegion(), funcOp.getRegion().end(), mapping
    );
    rewriter.mergeBlocks(&funcOp.getRegion().back(), &funcOp.getRegion().front());

    rewriter.eraseOp(op);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(
          &static_cast<const Impl *>(this)->getRoot()->getRegion(0).front()
      );
      rewriter.insert(funcOp);
    }

    return success();
  }
  friend Impl;
};

/// Converts `struct.def` ops into pcl modules (represented with `func.def` ops).
class ConvertStructDefOp : public OpConversionPattern<StructDefOp>,
                           public BodyCopier<StructDefOp, ConvertStructDefOp> {
  ModuleOp root;

public:
  ConvertStructDefOp(MLIRContext *context, ModuleOp rootMod, PatternBenefit patBenefit = 1)
      : OpConversionPattern(context, patBenefit), root(rootMod) {}
  ConvertStructDefOp(
      const TypeConverter &tc, MLIRContext *context, ModuleOp rootMod, PatternBenefit patBenefit = 1
  )
      : OpConversionPattern(tc, context, patBenefit), root(rootMod) {}

  FailureOr<FuncDefOp> getFuncOp(StructDefOp op) const {
    auto constrainFuncOp = op.getConstrainFuncOp();
    if (!constrainFuncOp) {
      return op.emitOpError() << "must have a @" << FUNC_NAME_CONSTRAIN
                              << " function for converting to pcl";
    }
    return constrainFuncOp;
  }

  unsigned int baseOffset() const { return 1; }

  llvm::SmallVector<Type> outputTypes(StructDefOp op) const {
    return mapOutputMembers<Type>(op, [ctx = op.getContext()](MemberDefOp) {
      return pcl::FeltType::get(ctx);
    });
  }

  ModuleOp getRoot() const { return root; }

  LogicalResult
  matchAndRewrite(StructDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    return copyBody(op, rewriter);
  }
};

class ConvertFreeFunction : public OpConversionPattern<FuncDefOp>,
                            public BodyCopier<FuncDefOp, ConvertFreeFunction> {
  llvm::DenseSet<FuncDefOp> &funcs;
  ModuleOp root;

public:
  ConvertFreeFunction(
      MLIRContext *context, llvm::DenseSet<FuncDefOp> &usedFreeFunctions, ModuleOp rootOp,
      PatternBenefit patBenefit = 1
  )
      : OpConversionPattern(context, patBenefit), funcs(usedFreeFunctions), root(rootOp) {}
  ConvertFreeFunction(
      const TypeConverter &tc, MLIRContext *context, llvm::DenseSet<FuncDefOp> &usedFreeFunctions,
      ModuleOp rootOp, PatternBenefit patBenefit = 1
  )
      : OpConversionPattern(tc, context, patBenefit), funcs(usedFreeFunctions), root(rootOp) {}

  FailureOr<FuncDefOp> getFuncOp(FuncDefOp op) const { return op; }

  unsigned int baseOffset() const { return 0; }

  llvm::SmallVector<Type> outputTypes(FuncDefOp op) const {
    return llvm::SmallVector<Type>(
        op.getFunctionType().getNumResults(), pcl::FeltType::get(op.getContext())
    );
  }

  ModuleOp getRoot() const { return root; }

  LogicalResult
  matchAndRewrite(FuncDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    // Don't convert functions outside the set.
    if (!funcs.contains(op)) {
      return failure();
    }
    return copyBody(op, rewriter);
  }
};

/// Removes `builtin.module` operations.
struct RemoveModuleOp : public OpConversionPattern<ModuleOp> {
  using OpConversionPattern<ModuleOp>::OpConversionPattern;

  LogicalResult match(ModuleOp) const override { return success(); }

  void rewrite(ModuleOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
  }
};

using DupVarsReplacements = llvm::DenseMap<pcl::VarOp, Value>;

class RemoveDuplicateVarOp : public OpConversionPattern<pcl::VarOp> {
  DupVarsReplacements &replacements;

public:
  RemoveDuplicateVarOp(
      MLIRContext *context, DupVarsReplacements &opReplacements, PatternBenefit patBenefit = 1
  )
      : OpConversionPattern(context, patBenefit), replacements(opReplacements) {}
  RemoveDuplicateVarOp(
      const TypeConverter &tc, MLIRContext *context, DupVarsReplacements &opReplacements,
      PatternBenefit patBenefit = 1
  )
      : OpConversionPattern(tc, context, patBenefit), replacements(opReplacements) {}

  LogicalResult match(pcl::VarOp op) const override {
    return success(replacements.find(op) != replacements.end());
  }

  void rewrite(pcl::VarOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto value = replacements[op];
    rewriter.replaceOp(op, value);
  }
};

struct ConvertConstrainCall : public OpConversionPattern<CallOp> {
  using OpConversionPattern<CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CallOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    SymbolTableCollection tables;
    auto callee = op.getCalleeTarget(tables);
    // We only care about constrain functions.
    if (failed(callee) || !callee->get().isStructConstrain()) {
      return failure();
    }

    auto subcmp = mlir::dyn_cast_if_present<TypedValue<StructType>>(op.getArgOperands().front());
    if (!subcmp) {
      return op->emitOpError() << "expected argument #0 to be a struct type";
    }
    auto subcmpOp = mlir::dyn_cast_if_present<MemberReadOp>(subcmp.getDefiningOp());
    if (!subcmpOp) {
      return failure();
    }
    Twine subcmpName(subcmpOp.getMemberName());
    auto defOp = subcmp.getType().getDefinition(tables, op);
    if (failed(defOp)) {
      return failure();
    }

    auto members = defOp->get().getMemberDefs();
    auto publicMembers = llvm::filter_to_vector(members, [](MemberDefOp memberDefOp) {
      return memberDefOp.hasPublicAttr();
    });
    SmallVector<Type> resultTypes(publicMembers.size(), pcl::FeltType::get(getContext()));
    auto calleeName = flatStructName(defOp->get());
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), calleeName, TypeRange(resultTypes), adaptor.getArgOperands().drop_front()
    );
    for (auto [member, result] : llvm::zip_equal(publicMembers, call.getResults())) {
      auto name = (subcmpName + "." + member.getSymName()).str();
      auto var =
          rewriter.create<pcl::VarOp>(op.getLoc(), rewriter.getStringAttr(name), /*public=*/false);
      auto eqCmp = rewriter.create<pcl::CmpEqOp>(op.getLoc(), var, result);
      rewriter.create<pcl::AssertOp>(op.getLoc(), eqCmp);
    }
    rewriter.eraseOp(op);

    return success();
  }
};

/// Converts calls to free functions. Only `function.call` ops inside functions that need to be
/// converted are marked illegal, so the pattern only needs to check if the callee is not a contrain
/// call.
struct ConvertFreeFunctionCall : public OpConversionPattern<CallOp> {
  using OpConversionPattern<CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CallOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    SymbolTableCollection tables;
    auto callee = op.getCalleeTarget(tables);
    if (failed(callee) || callee->get().isStructConstrain()) {
      return failure();
    }

    SmallVector<Type> resultTypes(op.getNumResults(), pcl::FeltType::get(getContext()));
    auto calleeName = flatStructName(callee->get());
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, calleeName, TypeRange(resultTypes), adaptor.getArgOperands()
    );
    return success();
  }
};

class RemoveFreeFunction : public OpConversionPattern<FuncDefOp> {
  llvm::DenseSet<FuncDefOp> &funcs;

public:
  RemoveFreeFunction(
      MLIRContext *context, llvm::DenseSet<FuncDefOp> &usedFreeFunctions,
      PatternBenefit patBenefit = 1
  )
      : OpConversionPattern(context, patBenefit), funcs(usedFreeFunctions) {}
  RemoveFreeFunction(
      const TypeConverter &tc, MLIRContext *context, llvm::DenseSet<FuncDefOp> &usedFreeFunctions,
      PatternBenefit patBenefit = 1
  )
      : OpConversionPattern(tc, context, patBenefit), funcs(usedFreeFunctions) {}
  LogicalResult
  matchAndRewrite(FuncDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    // Remove any op that is NOT in the set.
    if (funcs.contains(op)) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Populates the set with the patterns used in step 1 of the conversion.
static void populateStep1ConversionPatterns(
    const TypeConverter &tc, RewritePatternSet &patterns, MLIRContext *ctx, NonDetOpNames &names
) {
  patterns.add<
      // clang-format off
      ConvertConstantOp<FeltConstantOp>,
      ConvertConstantOp<arith::ConstantOp>,
      ConvertBinaryOp<AddFeltOp, pcl::AddOp>,
      ConvertBinaryOp<SubFeltOp, pcl::SubOp>,
      ConvertBinaryOp<MulFeltOp, pcl::MulOp>,
      ConvertUnaryOp<NegFeltOp, pcl::NegOp>,
      ConvertBinaryOp<AndBoolOp, pcl::AndOp>,
      ConvertBinaryOp<OrBoolOp, pcl::OrOp>,
      ConvertUnaryOp<NotBoolOp, pcl::NotOp>,
      ConvertBoolXorOp,
      RemoveIntToFeltOp,
      ConvertCmpOp,
      ConvertEmitEqualityOp,
      // This pattern is currently disabled because asserts may represent predicates that are not actually part of the constraint system.
      // ConvertAssertOp,
      ConvertSelfMemberReadOpOfFelt,
      ConvertSelfMemberReadOpOfSubcmp,
      ConvertSubcmpMemberReadOp,
      ConvertReturnOp,
      ConvertFreeFuncReturnOp,
      ConvertConstrainCall,
      ConvertFreeFunctionCall
      // clang-format on
      >(tc, ctx);
  patterns.add<ConvertNonDetOp>(tc, ctx, names);
}

/// Populates the set with the patterns used in step 2 of the conversion.
static void populateStep2ConversionPatterns(
    const TypeConverter &tc, RewritePatternSet &patterns, MLIRContext *ctx, ModuleOp root,
    llvm::DenseSet<FuncDefOp> &usedFreeFunctions
) {
  patterns.add<ConvertStructDefOp>(tc, ctx, root);
  patterns.add<RemoveFreeFunction>(tc, ctx, usedFreeFunctions);
  patterns.add<ConvertFreeFunction>(tc, ctx, usedFreeFunctions, root);
}

/// Populates the set with the patterns used in step 3 of the conversion.
static void populateStep3ConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *context, DupVarsReplacements &replacements
) {
  patterns.add<RemoveDuplicateVarOp>(context, replacements);
  patterns.add<RemoveModuleOp>(context);
}

/// Returns true if the operation is legal wrt step 1.
///
/// An operation is legal in Step 1 if its located outside the
/// `constrain` function of a struct.
static bool isStep1LegalOp(Operation *op, llvm::DenseSet<FuncDefOp> &usedFreeFunctions) {
  auto structDefOp = op->getParentOfType<StructDefOp>();
  if (!structDefOp) {
    auto funcDefOp = op->getParentOfType<FuncDefOp>();
    if (!funcDefOp) {
      // Legal because is not within a struct nor a free function definition.
      return true;
    }

    // If the op is inside a free function, check if the function is in the set.
    // If the function is in the set, then the op is illegal.
    return !usedFreeFunctions.contains(funcDefOp);
  }
  auto funcDefOp = op->getParentOfType<FuncDefOp>();
  if (!funcDefOp) {
    // Legal because is not within a function definition.
    return true;
  }
  // Legal if the containing function definition is not the struct's constrain function.
  return structDefOp.getConstrainFuncOp() != funcDefOp;
}

/// Populates the conversion target with the legallity expected of step 1 of the conversion.
static void populateStep1ConversionTarget(
    ConversionTarget &target, NonDetOpNames &names, llvm::DenseSet<FuncDefOp> &usedFreeFunctions
) {
  target.addLegalDialect<pcl::PCLDialect, func::FuncDialect>();
  target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
  target.addDynamicallyLegalDialect<
      BoolDialect, FeltDialect, CastDialect, arith::ArithDialect, ConstrainDialect,
      array::ArrayDialect, global::GlobalDialect, include::IncludeDialect, pod::PODDialect,
      polymorphic::PolymorphicDialect, ram::RAMDialect, smt::SMTDialect, string::StringDialect,
      verif::VerifDialect, LLZKDialect, StructDialect, FunctionDialect, scf::SCFDialect>(
      [&usedFreeFunctions](Operation *op) { return isStep1LegalOp(op, usedFreeFunctions); }
  );
  target.addLegalOp<FuncDefOp>();

  target.addDynamicallyLegalOp<NonDetOp>([&names, &usedFreeFunctions](NonDetOp op) {
    return isStep1LegalOp(op, usedFreeFunctions) && names.find(op) == names.end();
  });
}

/// Populates the conversion target with the legallity expected of step 2 of the conversion.
static void populateStep2ConversionTarget(ConversionTarget &target) {
  target.addLegalDialect<pcl::PCLDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::CallOp, func::ReturnOp>();
  target.addIllegalOp<StructDefOp, FuncDefOp>();
}

/// Populates the conversion target with the legallity expected of step 3 of the conversion.
static void populateStep3ConversionTarget(
    ConversionTarget &target, DupVarsReplacements &replacements, ModuleOp root
) {
  target.addLegalDialect<pcl::PCLDialect, func::FuncDialect>();
  target.addDynamicallyLegalOp<pcl::VarOp>([&replacements](pcl::VarOp op) {
    return replacements.find(op) == replacements.end();
  });
  target.addDynamicallyLegalOp<ModuleOp>([root](ModuleOp op) { return op == root; });
}

/// Type converter from LLZK types to PCL.
struct PCLTypeConverter : public TypeConverter {
  PCLTypeConverter() {
    // Default conversion.
    addConversion([](Type t) { return t; });

    addConversion([](IntegerType t) -> Type { return pcl::BoolType::get(t.getContext()); });

    addConversion([](FeltType t) { return pcl::FeltType::get(t.getContext()); });

    addSourceMaterialization(
        [](OpBuilder &builder, Type t, ValueRange values, Location location) -> Value {
      if (values.size() != 1) {
        return nullptr;
      }
      return builder.create<UnrealizedConversionCastOp>(location, t, values[0]).getResult(0);
    }
    );

    addTargetMaterialization(
        [](OpBuilder &builder, Type t, ValueRange values, Location location) -> Value {
      if (values.size() != 1) {
        return nullptr;
      }

      return builder.create<UnrealizedConversionCastOp>(location, t, values[0]).getResult(0);
    }
    );

    // Handles the conversion from booleans to felts.
    //
    // This conversion may be necessary in situations where a boolean result is used as operand of
    // an operation that expects a felt. For example, the following input IR:
    //
    // ```
    //  %felt_const_1 = felt.const 1 : !F
    //  %felt_const_65536 = felt.const 65536 : !F
    //  %0 = bool.cmp lt(%in, %felt_const_65536) : !F, !F
    //  %1 = cast.tofelt %0 : i1, !F
    //  constrain.eq %1, %felt_const_1 : !F, !F
    // ```
    //
    // Can be represented in PCL as:
    //
    // ```
    // (assert (= (< %in 65536) 1))
    // ```
    //
    // The result of `(< %in 65536)` needs to be converted from a `pcl.bool` to a `pcl.felt` in
    // order for the IR to typecheck.
    addTargetMaterialization(
        [](OpBuilder &builder, pcl::FeltType, ValueRange values, Location location) -> Value {
      if (values.size() != 1 || !mlir::isa<pcl::BoolType>(values[0].getType())) {
        return nullptr;
      }
      return builder.create<pcl::AsFeltOp>(location, values[0]);
    }
    );

    // Handles the conversion from felts to booleans.
    //
    // This conversion is the counterpart of the conversion above and is used in situations where a
    // felt was passed as operand to an op that expects a boolean.
    //
    // The value is converted by testing for equality against the falsy value (0).
    addTargetMaterialization(
        [](OpBuilder &builder, pcl::BoolType, ValueRange values, Location location) -> Value {
      if (values.size() != 1 || !mlir::isa<pcl::FeltType>(values[0].getType())) {
        return nullptr;
      }

      llvm::APInt zeroValue;
      auto zero = builder.create<pcl::ConstOp>(
          location, pcl::FeltAttr::get(builder.getContext(), zeroValue)
      );
      auto eqOp = builder.create<pcl::CmpEqOp>(location, values[0], zero);
      return builder.create<pcl::NotOp>(location, eqOp);
    }
    );
  }
};

class PassImpl : public pcl::impl::PCLLoweringPassBase<PassImpl> {
  using Base = PCLLoweringPassBase<PassImpl>;
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pcl::PCLDialect, func::FuncDialect>();
  }

  /// The translation only works now on LLZK structs where all the members are felts.
  LogicalResult validateStruct(StructDefOp structDef) {
    for (auto member : structDef.getMemberDefs()) {
      auto memberType = member.getType();
      if (!llvm::isa<FeltType, StructType>(memberType)) {
        return member.emitError() << "Member must be felt or struct type. Found " << memberType
                                  << " for member: " << member.getName();
      }
    }
    return success();
  }

  LogicalResult validateStructs() {
    return failure(
        getOperation()
            ->walk([this](StructDefOp op) {
      if (failed(validateStruct(op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }).wasInterrupted()
    );
  }

  // PCL programs require a module-level attribute specifying the prime.
  LogicalResult setPrime(llvm::APInt &prime) {
    getOperation()->setAttrs(DictionaryAttr::get(&getContext()));
    getOperation()->setAttr("pcl.prime", pcl::PrimeAttr::get(&getContext(), prime));

    return success();
  }

  FailureOr<llvm::APSInt> selectPrime() {
    FieldSet fields;
    // If the collection reports that at least one FeltType did not declare the field and
    // the fields set is empty, then we raise an error.
    if (failed(collectFields(getOperation(), fields)) && fields.empty()) {
      return getOperation()->emitOpError() << "could not deduce the prime field";
    }
    // If the fields is empty and we reached this point it means that the IR we are about to lower
    // does not have a single felt type (because felts without a field will make `collectFields`
    // return failure). We return an error here since we don't have a prime to emit. In practice,
    // this situation it's going to be unlikely.
    if (fields.empty()) {
      return getOperation()->emitOpError()
             << "does not contain felt types and prime field couldn't be deduced";
    }
    // The pass only supports having one field for the whole circuit.
    if (fields.size() > 1) {
      return getOperation()->emitOpError() << "multiple fields is not supported";
    }
    const auto &selectedField = *(fields.begin());
    return toAPSInt(selectedField.get().prime());
  }

  /// Collects all the member names that are already in use.
  llvm::StringSet<> collectUsedNames() {
    llvm::StringSet<> names;
    getOperation()->walk([&names](MemberDefOp op) { names.insert(op.getSymName()); });
    return names;
  }

  /// Collects all the `llzk.nondet` ops that need to be replaced.
  ///
  /// Only `llzk.nondet` ops of type `!felt.type` are considered.
  NonDetOpNames collectNonDetOpNames() {
    uint64_t count = 0;
    auto usedNames = collectUsedNames();
    NonDetOpNames names;
    getOperation()->walk([&count, &names, &usedNames, this](NonDetOp op) {
      if (!mlir::isa<FeltType>(op.getType())) {
        return;
      }
      StringRef nameRef;
      SmallVector<char, 25> nameSto;
      do {
        nameSto.clear();
        Twine name = "_nondet_internal_var__" + Twine(count);
        nameRef = name.toStringRef(nameSto);
        count++;
      } while (usedNames.contains(nameRef));
      names.insert({op, StringAttr::get(&getContext(), nameRef)});
    });
    return names;
  }

  /// Collects all the vars that need to be replaced.
  DupVarsReplacements collectDupVarsReplacements() {
    DupVarsReplacements replacements;
    getOperation()->walk([&replacements](func::FuncOp fn) {
      DominanceInfo dom(fn);
      llvm::StringMap<llvm::SmallVector<pcl::VarOp, 1>> varsByName;
      fn->walk([&varsByName](pcl::VarOp var) { varsByName[var.getName()].push_back(var); });

      for (auto &[_, vars] : varsByName) {
        if (vars.empty()) {
          continue;
        }
        std::stable_sort(vars.begin(), vars.end(), [&dom](pcl::VarOp lhs, pcl::VarOp rhs) {
          return dom.dominates(lhs.getOperation(), rhs);
        });
        auto fst = vars[0];
        for (auto other : ArrayRef(vars).drop_front()) {
          replacements[other] = fst;
        }
      }
    });
    return replacements;
  }

  /// Collects the call-graph from the constrain functions (excluding the `@constrain` functions
  /// themselves). Since we only care if a given `function.def` is part of the graph or not we
  /// return the set of vertices.
  llvm::DenseSet<FuncDefOp> collectConstrainCallGraph() {
    llvm::SmallVector<FuncDefOp> WL;
    llvm::DenseSet<FuncDefOp> funcs;
    mlir::SymbolTableCollection tables;

    // Fill the worklist with the initial set of functions.
    getOperation()->walk([&WL, &funcs, &tables](StructDefOp structOp) {
      auto f = structOp.getConstrainFuncOp();
      if (!f) {
        return;
      }

      f.walk([&WL, &funcs, &tables](CallOp callOp) {
        auto callee = callOp.getCalleeTarget(tables);
        // Ignore it if the lookup failed, is a @constrain function, or is already in the set.
        if (failed(callee) || callee->get().isStructConstrain() || funcs.contains(callee->get())) {
          return;
        }
        funcs.insert(callee->get());
        WL.push_back(callee->get());
      });
    });

    // Complete the graph using the worklist.
    while (!WL.empty()) {
      auto next = WL[WL.size() - 1];
      WL.pop_back();

      next.walk([&WL, &funcs, &tables](CallOp op) {
        auto callee = op.getCalleeTarget(tables);
        if (failed(callee) || funcs.contains(callee->get())) {
          return;
        }
        funcs.insert(callee->get());
        WL.push_back(callee->get());
      });
    }

    return funcs;
  }

  /// Step 1 converts the body of each struct to PCL operations.
  ///
  /// This conversion is performed before moving the body to a function because
  /// that way the IR can access information about the members of the struct.
  LogicalResult runStep1(llvm::DenseSet<FuncDefOp> &usedFreeFunctions) {
    auto nonDetNames = collectNonDetOpNames();
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    PCLTypeConverter tc;
    populateStep1ConversionPatterns(tc, patterns, &getContext(), nonDetNames);
    populateStep1ConversionTarget(target, nonDetNames, usedFreeFunctions);

    return applyFullConversion(getOperation(), target, std::move(patterns));
  }

  /// Step 2 converts the struct to a function, moving the contents of the @constrain function
  /// into the body of the new function.
  LogicalResult runStep2(llvm::DenseSet<FuncDefOp> &usedFreeFunctions) {
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    PCLTypeConverter tc;
    populateStep2ConversionPatterns(tc, patterns, &getContext(), getOperation(), usedFreeFunctions);
    populateStep2ConversionTarget(target);

    return applyFullConversion(getOperation(), target, std::move(patterns));
  }

  /// Step 3 cleans up the IR removing unnecessary ops that may be left over by the previous steps.
  ///
  /// The cleanup operations are:
  ///
  /// - The conversion process may generate multiple copies of the same variable. This is fine since
  /// `VarOp` implements `Pure`. However, for cleaniness we remove these duplicates now, replacing
  /// all extra instances with the value that dominates everyone else.
  ///
  /// - Remove empty non-root module ops.
  LogicalResult runStep3() {
    DupVarsReplacements replacements = collectDupVarsReplacements();

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    populateStep3ConversionPatterns(patterns, &getContext(), replacements);
    populateStep3ConversionTarget(target, replacements, getOperation());

    return applyFullConversion(getOperation(), target, std::move(patterns));
  }

  void runOnOperation() override {
    // check PCLDialect is loaded.
    assert(getContext().getLoadedDialect<pcl::PCLDialect>() && "PCL dialect not loaded");
    auto prime = selectPrime();
    if (failed(prime)) {
      signalPassFailure();
      return;
    }

    if (failed(validateStructs())) {
      signalPassFailure();
      return;
    }
    auto usedFreeFunctions = collectConstrainCallGraph();

    if (failed(runStep1(usedFreeFunctions))) {
      signalPassFailure();
      return;
    }

    if (failed(runStep2(usedFreeFunctions))) {
      signalPassFailure();
      return;
    }

    if (failed(runStep3())) {
      signalPassFailure();
      return;
    }

    if (failed(setPrime(*prime))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace
