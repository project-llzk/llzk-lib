//===-- PCLLoweringPass.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-pcl-lowering` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Config/Config.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Transforms/LLZKLoweringUtils.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "r1cs/Dialect/IR/Attrs.h"
#include "r1cs/Dialect/IR/Ops.h"
#include "r1cs/Dialect/IR/Types.h"
#include "r1cs/Transforms/TransformationPasses.h"

#include <pcl/Dialect/IR/Dialect.h>
#include <pcl/Dialect/IR/Ops.h>
#include <pcl/Dialect/IR/Types.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#include <deque>
#include <memory>

// Include the generated base pass class definitions.
namespace r1cs {
#define GEN_PASS_DECL_PCLLOWERINGPASS
#define GEN_PASS_DEF_PCLLOWERINGPASS
#include "r1cs/Transforms/TransformationPasses.h.inc"
} // namespace r1cs

using namespace mlir;
using namespace llzk;
using namespace llzk::cast;
using namespace llzk::boolean;
using namespace llzk::constrain;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::component;

namespace {

static FailureOr<Value> lookup(Value v, llvm::DenseMap<Value, Value> &m, Operation *onError) {
  if (auto it = m.find(v); it != m.end()) {
    return it->second;
  }
  return onError->emitError("missing operand mapping");
}

static void rememberResult(Value from, Value to, llvm::DenseMap<Value, Value> &m) {
  (void)m.try_emplace(from, to);
}

// Convert binary LLZK op to corresponding binary PCL op
template <typename SrcBinOp, typename DstBinOp>
static LogicalResult
lowerBinaryLike(OpBuilder &b, SrcBinOp src, llvm::DenseMap<Value, Value> &mapping) {
  auto loc = src.getLoc();
  auto op = src.getOperation();
  auto lhs = lookup(src.getLhs(), mapping, op);
  if (failed(lhs)) {
    return failure();
  }
  auto rhs = lookup(src.getRhs(), mapping, op);
  if (failed(rhs)) {
    return failure();
  }

  auto dst = b.create<DstBinOp>(loc, *lhs, *rhs);
  rememberResult(src.getResult(), dst.getRes(), mapping);
  return success();
}

static LogicalResult
lowerConst(OpBuilder &b, FeltConstantOp cst, llvm::DenseMap<Value, Value> &mapping) {
  auto attr = pcl::FeltAttr::get(b.getContext(), cst.getValue());
  auto dst = b.create<pcl::ConstOp>(cst.getLoc(), attr);
  rememberResult(cst.getResult(), dst.getRes(), mapping);
  return success();
}

class PCLLoweringPass : public r1cs::impl::PCLLoweringPassBase<PCLLoweringPass> {

private:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pcl::PCLDialect, func::FuncDialect>();
  }

  /// The translation only works now on LLZK structs where all the members are felts.
  LogicalResult validateStruct(StructDefOp structDef) {
    for (auto member : structDef.getMemberDefs()) {
      auto memberType = member.getType();
      if (!llvm::isa<FeltType>(memberType)) {
        return member.emitError() << "Member must be felt type. Found " << memberType
                                  << " for member: " << member.getName();
      }
    }
    return success();
  }

  /// Emit assertions for an equality `lhs == rhs`, with fast paths when one side
  /// is a boolean and the other side is a constant {0,1}.
  ///
  /// Cases handled:
  ///   - bool == 1  → assert(bool)
  ///   - 1 == bool  → assert(bool)
  ///   - bool == 0  → assert(!bool)
  ///   - 0 == bool  → assert(!bool)
  ///   - otherwise  → assert(lhs == rhs)
  ///
  /// Returns success after emitting IR.
  static LogicalResult
  emitAssertEqOptimized(OpBuilder &b, Location loc, Value lhsVal, Value rhsVal) {
    // --- Small helpers --------------------------------------------------------
    auto isBool = [](Value v) { return llvm::isa<pcl::BoolType>(v.getType()); };

    auto getConstAPInt = [](Value v) -> std::optional<llvm::APInt> {
      if (auto c = llvm::dyn_cast_if_present<pcl::ConstOp>(v.getDefiningOp())) {
        // Chain: ConstOp -> FeltAttr (or BoolAttr-as-int) -> IntegerAttr -> APInt
        return c.getValue().getValue().getValue();
      }
      return std::nullopt;
    };

    auto isConstOne = [&](Value v) {
      if (auto ap = getConstAPInt(v)) {
        return ap->isOne();
      }
      return false;
    };
    auto isConstZero = [&](Value v) {
      if (auto ap = getConstAPInt(v)) {
        return ap->isZero();
      }
      return false;
    };

    auto emitEqAssert = [&](Value l, Value r) {
      auto eq = b.create<pcl::CmpEqOp>(loc, l, r);
      b.create<pcl::AssertOp>(loc, eq.getRes());
    };

    auto emitAssertTrue = [&](Value pred) { b.create<pcl::AssertOp>(loc, pred); };

    auto emitAssertFalse = [&](Value pred) {
      auto neg = b.create<pcl::NotOp>(loc, pred);
      b.create<pcl::AssertOp>(loc, neg.getRes());
    };

    // Optimized handling of boolean patterns
    if (isBool(lhsVal) && isConstOne(rhsVal)) {
      // bool == 1 → assert(bool)
      emitAssertTrue(lhsVal);
      return success();
    }
    if (isBool(rhsVal) && isConstOne(lhsVal)) {
      // 1 == bool → assert(bool)
      emitAssertTrue(rhsVal);
      return success();
    }
    if (isBool(lhsVal) && isConstZero(rhsVal)) {
      // bool == 0 → assert(!bool)
      emitAssertFalse(lhsVal);
      return success();
    }
    if (isBool(rhsVal) && isConstZero(lhsVal)) {
      // 0 == bool → assert(!bool)
      emitAssertFalse(rhsVal);
      return success();
    }

    // Fallback to assert(lhs == rhs)
    emitEqAssert(lhsVal, rhsVal);
    return success();
  }

  /// Lower the constraint ops to PCL ops
  LogicalResult lowerStructToPCLBody(StructDefOp structDef, func::FuncOp dstFunc) {
    // As we build, map llzk values to their pcl ones
    llvm::DenseMap<Value, Value> llzkToPcl;
    OpBuilder b(dstFunc.getBody());
    // Map member name to PCL vars; public members are outputs, privates are intermediates
    llvm::DenseMap<StringRef, Value> member2pclvar;
    llvm::SmallVector<Value> outVars;

    auto srcFunc = structDef.getConstrainFuncOp();
    auto srcArgs = srcFunc.getArguments().drop_front();
    auto dstArgs = dstFunc.getArguments();
    if (dstArgs.size() != srcArgs.size()) {
      return srcFunc.emitError("arg count mismatch after dropping self");
    }

    // 1-1 mapping of args from constraint args to PCL args
    for (auto [src, dst] : llvm::zip(srcArgs, dstArgs)) {
      llzkToPcl.try_emplace(src, dst);
    }
    for (auto memberDef : structDef.getMemberDefs()) {
      // Create a PCL var for each struct member. Public members are outputs in PCL
      auto pclVar =
          b.create<pcl::VarOp>(memberDef.getLoc(), memberDef.getName(), memberDef.hasPublicAttr());
      member2pclvar.insert({memberDef.getName(), pclVar});
      if (memberDef.hasPublicAttr()) {
        outVars.push_back(pclVar);
      }
    }
    if (!srcFunc.getBody().hasOneBlock()) {
      return srcFunc.emitError(
          "llzk-to-pcl translation assumes the constrain function body has 1 block"
      );
    }
    Block &srcEntry = srcFunc.getBody().front();
    // Translate each op. Almost 1-1 and currently only support Felt ops.
    // TODO: Support calls, if-else, globals/lookups.
    for (Operation &op : srcEntry) {
      LogicalResult res = success();
      llvm::TypeSwitch<Operation *, void>(&op)
          .Case<FeltConstantOp>([&b, &llzkToPcl, &res](auto c) {
        res = lowerConst(b, c, llzkToPcl);
      })
          .Case<AddFeltOp>([&b, &llzkToPcl, &res](auto a) {
        res = lowerBinaryLike<AddFeltOp, pcl::AddOp>(b, a, llzkToPcl);
      })
          .Case<SubFeltOp>([&b, &llzkToPcl, &res](auto s) {
        res = lowerBinaryLike<SubFeltOp, pcl::SubOp>(b, s, llzkToPcl);
      })
          .Case<MulFeltOp>([&b, &llzkToPcl, &res](auto m) {
        res = lowerBinaryLike<MulFeltOp, pcl::MulOp>(b, m, llzkToPcl);
      })
          .Case<IntToFeltOp>([&llzkToPcl, &res](auto m) {
        auto arg = lookup(m.getValue(), llzkToPcl, m.getOperation());
        if (failed(arg)) {
          res = failure();
          return;
        }
        rememberResult(m.getResult(), arg.value(), llzkToPcl);
      })
          .Case<CmpOp>([&b, &llzkToPcl, &res](auto cmp) {
        auto pred = cmp.getPredicate();
        switch (pred) {
        case FeltCmpPredicate::EQ:
          res = lowerBinaryLike<CmpOp, pcl::CmpEqOp>(b, cmp, llzkToPcl);
          break;
        case FeltCmpPredicate::NE: {
          // Translate not-equals as an equality followed by a negation
          auto eq = lowerBinaryLike<CmpOp, pcl::CmpEqOp>(b, cmp, llzkToPcl);
          if (failed(eq)) {
            res = eq;
            break;
          }
          // Get the result from the `pcl::CmpEqOp` to pass into `Neg`
          auto eqRes = lookup(cmp.getResult(), llzkToPcl, cmp.getOperation());
          if (failed(eqRes)) {
            res = failure();
            break;
          }
          auto loc = cmp.getLoc();
          auto neg = b.create<pcl::NegOp>(loc, *eqRes);
          // Associate the result of the llzk-op with the result of the pcl-neg
          rememberResult(cmp.getResult(), neg.getResult(), llzkToPcl);
          break;
        }
        case FeltCmpPredicate::LT:
          res = lowerBinaryLike<CmpOp, pcl::CmpLtOp>(b, cmp, llzkToPcl);
          break;
        case FeltCmpPredicate::LE:
          res = lowerBinaryLike<CmpOp, pcl::CmpLeOp>(b, cmp, llzkToPcl);
          break;
        case FeltCmpPredicate::GT:
          res = lowerBinaryLike<CmpOp, pcl::CmpGtOp>(b, cmp, llzkToPcl);
          break;
        case FeltCmpPredicate::GE:
          res = lowerBinaryLike<CmpOp, pcl::CmpGeOp>(b, cmp, llzkToPcl);
          break;
        }
      })
          .Case<EmitEqualityOp>([&b, &llzkToPcl, &res](auto eq) {
        auto lhs = lookup(eq.getLhs(), llzkToPcl, eq.getOperation());
        auto rhs = lookup(eq.getRhs(), llzkToPcl, eq.getOperation());
        if (failed(lhs) || failed(rhs)) {
          res = failure();
          return;
        }

        Value lhsVal = *lhs, rhsVal = *rhs;
        auto loc = eq.getLoc();
        if (failed(emitAssertEqOptimized(b, loc, lhsVal, rhsVal))) {
          res = failure();
          return;
        }
      })
          .Case<MemberReadOp>([&member2pclvar, &llzkToPcl, &srcFunc](auto read) {
        // At this point every member in the struct should have a var associated with it
        // so we should simply retrieve the var associated with the member.
        (void)srcFunc; // to silence unused variable warning if asserts are disabled
        assert(read.getComponent() == srcFunc.getArguments()[0]);
        if (auto it = member2pclvar.find(read.getMemberName()); it != member2pclvar.end()) {
          rememberResult(read.getResult(), it->getSecond(), llzkToPcl);
        } else {
          llvm_unreachable("Every member should have been mapped to a pcl var");
        }
      })
          .Case<ReturnOp>([&b, &outVars](auto ret) {
        // We return all the output vars we defined above.
        b.create<pcl::ReturnOp>(
            ret.getLoc(), (llvm::SmallVector<Value>(outVars.begin(), outVars.end()))
        );
      }).Default([](Operation *unknown) {
        unknown->emitError("unsupported op in PCL lowering: ") << unknown->getName();
      });
      if (failed(res)) {
        return failure();
      }
    }
    return success();
  }

  FailureOr<func::FuncOp> buildPCLFunc(StructDefOp structDef) {
    SmallVector<Type> pclInputTypes, pclOutputTypes;
    auto constrainFunc = structDef.getConstrainFuncOp();
    auto ctx = structDef.getContext();
    for (auto arg : constrainFunc.getArguments().drop_front()) {
      auto argType = arg.getType();
      if (!llvm::isa<FeltType>(argType)) {
        return constrainFunc.emitError()
               << "Constrain function's args are expected to be felts. Found " << argType
               << "for arg #: " << arg.getArgNumber();
      }
      pclInputTypes.push_back(pcl::FeltType::get(ctx));
    }
    for (auto member : structDef.getMemberDefs()) {
      auto memberType = member.getType();
      if (!llvm::isa<FeltType>(memberType)) {
        return structDef.emitError() << "Member must be felt type. Found " << memberType
                                     << " for member: " << member.getName();
      }
      if (member.hasPublicAttr()) {
        pclOutputTypes.push_back(pcl::FeltType::get(ctx));
      }
    }
    FunctionType fty = FunctionType::get(ctx, pclInputTypes, pclOutputTypes);
    auto func = func::FuncOp::create(constrainFunc.getLoc(), structDef.getName(), fty);
    func.addEntryBlock();
    return func;
  }

  // PCL programs require a module-level attribute specifying the prime.
  void setPrime(ModuleOp &newMod) {
    // Add an extra bit to avoid the prime being represented as a negative number
    auto newBitWidth = prime.getBitWidth() + 1;
    auto ty = IntegerType::get(newMod.getContext(), newBitWidth);
    auto intAttr = IntegerAttr::get(ty, prime.zext(newBitWidth));
    newMod->setAttr("pcl.prime", pcl::PrimeAttr::get(newMod.getContext(), intAttr));
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    // check PCLDialect is loaded.
    assert(moduleOp->getContext()->getLoadedDialect<pcl::PCLDialect>() && "PCL dialect not loaded");
    // Create the PCL module
    auto newMod = ModuleOp::create(moduleOp.getLoc());
    // Set the prime attribute
    setPrime(newMod);
    // Convert each struct to a PCL function
    auto walkResult = moduleOp.walk([this, &newMod](StructDefOp structDef) -> WalkResult {
      // 1) verify the struct can be converted to PCL
      if (failed(validateStruct(structDef))) {
        return WalkResult::interrupt();
      }
      // 2) Construct the PCL function op but with an empty body
      FailureOr<func::FuncOp> pclFuncOp = buildPCLFunc(structDef);
      if (failed(pclFuncOp)) {
        return WalkResult::interrupt();
      }
      // 3) Fill in the PCL function body
      newMod.getBody()->push_back(*pclFuncOp);
      if (failed(lowerStructToPCLBody(structDef, pclFuncOp.value()))) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    // clear the original ops
    moduleOp.getRegion().takeBody(newMod.getBodyRegion());
    // Replace the module attributes
    moduleOp->setAttrs(newMod->getAttrDictionary());
    newMod.erase();
  }
};
} // namespace

std::unique_ptr<Pass> r1cs::createPCLLoweringPass() { return std::make_unique<PCLLoweringPass>(); }
