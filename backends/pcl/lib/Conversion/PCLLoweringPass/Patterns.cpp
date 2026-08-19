//===-- Patterns.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// This file defines all the conversion patterns used by the `--llzk-to-pcl`
/// pass. The file is organized as follows:
///
/// - Helpers
/// - Patterns
/// - Lowering modes' populate methods
///
/// Please try keep the pattern definitions in alphabetical order so its easier
/// to find them when scrolling through the file.
///
//===----------------------------------------------------------------------===//

#include "Modes.h"

#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <numeric>
#include <utility>

using namespace mlir;
using namespace llzk;
using namespace llzk::cast;
using namespace llzk::boolean;
using namespace llzk::constrain;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::component;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns a flat representation of the fully qualified name of the struct.
template <typename Op> static std::string flatFullyQualifiedName(Op op) {
  auto fqn = op.getFullyQualifiedName();
  auto root = fqn.getRootReference().getValue();
  auto nested = fqn.getNestedReferences();
  StringRef sep("::");
  std::string name;
  name.reserve(
      std::accumulate(
          nested.begin(), nested.end(), root.size() + nested.size() * sep.size(),
          [](auto acc, auto ref) { return acc + ref.getValue().size(); }
      )
  );
  llvm::raw_string_ostream o(name);
  o << root;
  if (!fqn.getNestedReferences().empty()) {
    o << sep;
    interleave(nested, o, [&o](auto ref) { o << ref.getValue(); }, sep);
  }
  return name;
}

/// Copies the body of a function.
template <typename Op>
static LogicalResult copyBody(
    Op op, FuncDefOp srcFuncOp, ConversionPatternRewriter &rewriter, SmallVector<Type> outputs,
    ModuleOp root, unsigned baseOffset = 0
) {

  SmallVector<Type> inputs(
      srcFuncOp.getNumArguments() - baseOffset, pcl::FeltType::get(rewriter.getContext())
  );

  auto funcOp = func::FuncOp::create(
      op.getLoc(), flatFullyQualifiedName(op), rewriter.getFunctionType(inputs, outputs)
  );
  funcOp.addEntryBlock();
  IRMapping mapping;
  for (auto arg : srcFuncOp.getArguments().take_front(baseOffset)) {
    mapping.map(arg, Value());
  }
  for (auto [srcArg, dstArg] :
       llvm::zip_equal(srcFuncOp.getArguments().drop_front(baseOffset), funcOp.getArguments())) {
    auto argType = srcArg.getType();
    if (!llvm::isa<FeltType>(argType)) {
      return srcFuncOp->emitError() << "function's args are expected to be felts. Found " << argType
                                    << "for arg #: " << srcArg.getArgNumber();
    }
    mapping.map(srcArg, dstArg);
  }

  if (!srcFuncOp.getBody().hasOneBlock()) {
    return srcFuncOp->emitError(
        "llzk-to-pcl conversion assumes the constrain function body has 1 block"
    );
  }
  rewriter.cloneRegionBefore(
      srcFuncOp.getRegion(), funcOp.getRegion(), funcOp.getRegion().end(), mapping
  );
  rewriter.mergeBlocks(&funcOp.getRegion().back(), &funcOp.getRegion().front());

  rewriter.eraseOp(op);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(&root.getRegion().front());
    rewriter.insert(funcOp);
  }

  return success();
}

/// Maps the list of struct members that are considered outputs for the pcl module.
///
/// A member is an output if it has the `llzk.pub` attribute and is of type `!felt.type`.
template <typename T, typename Fn> SmallVector<T> mapOutputMembers(StructDefOp op, Fn callback) {
  SmallVector<T> out;
  auto members = op.getMemberDefs();
  out.reserve(members.size());
  for (auto memberDef : members) {
    if (llvm::isa<FeltType>(memberDef.getType()) && memberDef.hasPublicAttr()) {
      out.push_back(callback(memberDef));
    }
  }
  return out;
}

//===----------------------------------------------------------------------===//
// ConvertBinaryOp
//===----------------------------------------------------------------------===//

/// Generic conversion pattern for binary ops that have a 1:1 correspondence with a pcl op.
template <typename SrcOp, typename DstOp>
class ConvertBinaryOp : public OpConversionPattern<SrcOp> {
  using OpAdaptor = typename OpConversionPattern<SrcOp>::OpAdaptor;

  using OpConversionPattern<SrcOp>::getTypeConverter;
  using OpConversionPattern<SrcOp>::getContext;

public:
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<DstOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertBoolXorOp
//===----------------------------------------------------------------------===//

/// Converts a `bool.xor` into a negated `pcl.iff`.
struct ConvertBoolXorOp : public OpConversionPattern<XorBoolOp> {
  using OpConversionPattern<XorBoolOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      XorBoolOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    auto iffOp = rewriter.create<pcl::IffOp>(op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOpWithNewOp<pcl::NotOp>(op, iffOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertCmpOp
//===----------------------------------------------------------------------===//

/// Converts `bool.cmp` ops into their pcl counterparts.
struct ConvertCmpOp : public OpConversionPattern<CmpOp> {
  using OpConversionPattern<CmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
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
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertConstantOp
//===----------------------------------------------------------------------===//

template <typename Op> class ConstantOpValue {};

template <> class ConstantOpValue<FeltConstantOp> {
protected:
  APInt getValue(FeltConstantOp op) const { return op.getValue().getValue(); }
};

template <> class ConstantOpValue<arith::ConstantOp> {
protected:
  APInt getValue(arith::ConstantOp op) const {
    // Extend width by 1 bit to avoid sign issues.
    auto value = llvm::cast<IntegerAttr>(op.getValue()).getValue();
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

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto value = pcl::FeltAttr::get(rewriter.getContext(), getValue(op));
    rewriter.replaceOpWithNewOp<pcl::ConstOp>(op, value);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertConstrainCall
//===----------------------------------------------------------------------===//

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

    auto subcmp = llvm::dyn_cast_if_present<TypedValue<StructType>>(op.getArgOperands().front());
    if (!subcmp) {
      return op->emitOpError() << "expected argument #0 to be a struct type";
    }
    auto subcmpOp = llvm::dyn_cast_if_present<MemberReadOp>(subcmp.getDefiningOp());
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
    auto calleeName = flatFullyQualifiedName(defOp->get());
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

//===----------------------------------------------------------------------===//
// ConvertEmitEqualityOp
//===----------------------------------------------------------------------===//

/// Converts `constrain.eq` ops into an optimized `pcl.assert`.
class ConvertEmitEqualityOp : public OpConversionPattern<EmitEqualityOp> {
public:
  using OpConversionPattern<EmitEqualityOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      EmitEqualityOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {

    auto cmpEqOp = rewriter.create<pcl::CmpEqOp>(op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOpWithNewOp<pcl::AssertOp>(op, cmpEqOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertFreeFunction
//===----------------------------------------------------------------------===//

class ConvertFreeFunction : public OpConversionPattern<FuncDefOp> {
  pcl::lowering::UsedFreeFunctions &funcs;
  ModuleOp root;

public:
  template <typename... Args>
  ConvertFreeFunction(
      pcl::lowering::UsedFreeFunctions &usedFreeFunctions, ModuleOp rootOp, Args &&...args
  )
      : OpConversionPattern(std::forward<Args>(args)...), funcs(usedFreeFunctions), root(rootOp) {}

  LogicalResult
  matchAndRewrite(FuncDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    // Don't convert functions outside the set.
    if (!funcs.contains(op)) {
      return failure();
    }
    auto outputs = SmallVector<Type>(
        op.getFunctionType().getNumResults(), pcl::FeltType::get(op.getContext())
    );
    return copyBody(op, op, rewriter, outputs, root);
  }
};

//===----------------------------------------------------------------------===//
// ConvertFreeFunctionCall
//===----------------------------------------------------------------------===//

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
    auto calleeName = flatFullyQualifiedName(callee->get());
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, calleeName, TypeRange(resultTypes), adaptor.getArgOperands()
    );
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertFreeFunctionIntoStub
//===----------------------------------------------------------------------===//

class ConvertFreeFunctionIntoStub : public OpConversionPattern<FuncDefOp> {
  DenseSet<FuncDefOp> &stubs;
  ModuleOp root;

public:
  template <typename... Args>
  ConvertFreeFunctionIntoStub(DenseSet<FuncDefOp> &stubFunctions, ModuleOp rootOp, Args &&...args)
      : OpConversionPattern(std::forward<Args>(args)...), stubs(stubFunctions), root(rootOp) {}

  LogicalResult
  matchAndRewrite(FuncDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    // Don't convert functions outside the set.
    if (!stubs.contains(op)) {
      return failure();
    }

    auto feltType = pcl::FeltType::get(rewriter.getContext());
    SmallVector<Type> inputs(op.getNumArguments(), feltType);
    SmallVector<Type> outputs(op.getFunctionType().getNumResults(), feltType);

    auto funcOp = func::FuncOp::create(
        op.getLoc(), flatFullyQualifiedName(op), rewriter.getFunctionType(inputs, outputs),
        {NamedAttribute("sym_visibility", rewriter.getStringAttr("private"))}
    );

    rewriter.eraseOp(op);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&root->getRegion(0).front());
      rewriter.insert(funcOp);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertFreeFunctionReturnOp
//===----------------------------------------------------------------------===//

/// Converts `function.return` ops inside free functions into func return ops.
struct ConvertFreeFunctionReturnOp : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (op->getParentOfType<StructDefOp>()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertNonDetOp
//===----------------------------------------------------------------------===//

/// Converts `llzk.nondet` ops into fresh pcl variables.
class ConvertNonDetOp : public OpConversionPattern<NonDetOp> {
  pcl::lowering::NonDetOpNames &names;

public:
  template <typename... Args>
  ConvertNonDetOp(pcl::lowering::NonDetOpNames &opNames, Args &&...args)
      : OpConversionPattern(std::forward<Args>(args)...), names(opNames) {}

  LogicalResult
  matchAndRewrite(NonDetOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto it = names.find(op);
    if (it == names.end()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<pcl::VarOp>(op, it->second, /*public=*/false);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertReturnOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// ConvertSelfMemberReadOpOfFelt
//===----------------------------------------------------------------------===//

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
    if (!llvm::isa<FeltType>(defOp->get().getType())) {
      return failure();
    }

    auto pclVar = rewriter.create<pcl::VarOp>(
        defOp->get().getLoc(), defOp->get().getName(), defOp->get().hasPublicAttr()
    );
    rewriter.replaceOp(op, pclVar);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertSelfMemberReadOpOfSubcmp
//===----------------------------------------------------------------------===//

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

    if (!llvm::isa<StructType>(defOp->get().getType())) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertStructDefOp
//===----------------------------------------------------------------------===//

/// Converts `struct.def` ops into pcl modules (represented with `func.def` ops).
class ConvertStructDefOp : public OpConversionPattern<StructDefOp> {
  ModuleOp root;

public:
  template <typename... Args>
  ConvertStructDefOp(ModuleOp rootMod, Args &&...args)
      : OpConversionPattern(std::forward<Args>(args)...), root(rootMod) {}

  LogicalResult
  matchAndRewrite(StructDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto constrainFuncOp = op.getConstrainFuncOp();
    if (!constrainFuncOp) {
      return op.emitOpError() << "must have a @" << FUNC_NAME_CONSTRAIN
                              << " function for converting to pcl";
    }
    auto outputs = mapOutputMembers<Type>(op, [ctx = op.getContext()](MemberDefOp) {
      return pcl::FeltType::get(ctx);
    });

    return copyBody(op, constrainFuncOp, rewriter, outputs, root, /*baseOffset=*/1);
  }
};

//===----------------------------------------------------------------------===//
// ConvertSubcmpMemberReadOp
//===----------------------------------------------------------------------===//

struct ConvertSubcmpMemberReadOp : public OpConversionPattern<MemberReadOp> {
  using OpConversionPattern<MemberReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MemberReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto subcmp = llvm::dyn_cast_if_present<MemberReadOp>(op.getComponent().getDefiningOp());
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

//===----------------------------------------------------------------------===//
// ConvertUnaryOp
//===----------------------------------------------------------------------===//

/// Generic conversion pattern for unary ops that have a 1:1 correspondence with a pcl op.
template <typename SrcOp, typename DstOp> class ConvertUnaryOp : public OpConversionPattern<SrcOp> {
  using OpAdaptor = typename OpConversionPattern<SrcOp>::OpAdaptor;

public:
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<DstOp>(op, adaptor.getOperand());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RemoveDuplicateVarOp
//===----------------------------------------------------------------------===//

class RemoveDuplicateVarOp : public OpConversionPattern<pcl::VarOp> {
  pcl::lowering::DupVarsReplacements &replacements;

public:
  template <typename... Args>
  RemoveDuplicateVarOp(pcl::lowering::DupVarsReplacements &opReplacements, Args &&...args)
      : OpConversionPattern(std::forward<Args>(args)...), replacements(opReplacements) {}

  LogicalResult
  matchAndRewrite(pcl::VarOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto it = replacements.find(op);
    if (it == replacements.end()) {
      return failure();
    }
    rewriter.replaceOp(op, it->second);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RemoveFreeFunction
//===----------------------------------------------------------------------===//

/// Removes any `function.def` operation that is not in the given set of used free functions.
class RemoveFreeFunction : public OpConversionPattern<FuncDefOp> {
  pcl::lowering::UsedFreeFunctions &funcs;

public:
  template <typename... Args>
  RemoveFreeFunction(pcl::lowering::UsedFreeFunctions &usedFreeFunctions, Args &&...args)
      : OpConversionPattern(std::forward<Args>(args)...), funcs(usedFreeFunctions) {}

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

//===----------------------------------------------------------------------===//
// RemoveIntToFeltOp
//===----------------------------------------------------------------------===//

/// Removes `cast.tofelt` ops since all numerical types in pcl are the same.
struct RemoveIntToFeltOp : public OpConversionPattern<IntToFeltOp> {
  using OpConversionPattern<IntToFeltOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IntToFeltOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RemoveModuleOp
//===----------------------------------------------------------------------===//

/// Removes `builtin.module` operations.
struct RemoveModuleOp : public OpConversionPattern<ModuleOp> {
  using OpConversionPattern<ModuleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ModuleOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Lowering modes populate methods
//===----------------------------------------------------------------------===//

void pcl::lowering::BaseMode::populateStep1ConversionPatterns(
    const TypeConverter &tc, RewritePatternSet &patterns
) {
  patterns.add<
      // clang-format off
        ConvertBinaryOp<AddFeltOp, pcl::AddOp>,
        ConvertBinaryOp<AndBoolOp, pcl::AndOp>,
        ConvertBinaryOp<MulFeltOp, pcl::MulOp>,
        ConvertBinaryOp<OrBoolOp, pcl::OrOp>,
        ConvertBinaryOp<SubFeltOp, pcl::SubOp>,
        ConvertBoolXorOp,
        ConvertCmpOp,
        ConvertConstantOp<FeltConstantOp>,
        ConvertConstantOp<arith::ConstantOp>,
        ConvertConstrainCall,
        ConvertEmitEqualityOp,
        ConvertFreeFunctionCall,
        ConvertFreeFunctionReturnOp,
        ConvertReturnOp,
        ConvertSelfMemberReadOpOfFelt,
        ConvertSelfMemberReadOpOfSubcmp,
        ConvertSubcmpMemberReadOp,
        ConvertUnaryOp<NegFeltOp, pcl::NegOp>,
        ConvertUnaryOp<NotBoolOp, pcl::NotOp>,
        RemoveIntToFeltOp
      // clang-format on
      >(tc, &getContext());
  patterns.add<ConvertNonDetOp>(names, tc, &getContext());
}
void pcl::lowering::BaseMode::populateStep3ConversionPatterns(
    RewritePatternSet &patterns, DupVarsReplacements &replacements
) {
  patterns.add<RemoveDuplicateVarOp>(replacements, &getContext());
  patterns.add<RemoveModuleOp>(&getContext());
}

void pcl::lowering::FullLoweringMode::populateStep2ConversionPatterns(
    const TypeConverter &tc, RewritePatternSet &patterns
) {
  patterns.add<ConvertFreeFunction>(getUsedFreeFunctions(), getOperation(), tc, &getContext());
  patterns.add<ConvertStructDefOp>(getOperation(), tc, &getContext());
  patterns.add<RemoveFreeFunction>(getUsedFreeFunctions(), tc, &getContext());
}

void pcl::lowering::StubbedLoweringMode::populateStep2ConversionPatterns(
    const TypeConverter &tc, RewritePatternSet &patterns
) {
  patterns.add<ConvertFreeFunction>(getUsedFreeFunctions(), getOperation(), tc, &getContext());
  patterns.add<ConvertFreeFunctionIntoStub>(stubs, getOperation(), tc, &getContext());
  patterns.add<ConvertStructDefOp>(getOperation(), tc, &getContext());
  patterns.add<RemoveFreeFunction>(getUsedFreeFunctions(), tc, &getContext());
}
