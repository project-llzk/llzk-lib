//===-- VerifDialectTests.cpp - Unit tests for verif dialect ----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "OpTestBase.h"

#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Verif/IR/Ops.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::verif;

namespace {

class VerifDialectTests : public OpTests {
protected:
  OwningOpRef<ModuleOp> parseModule(StringRef source) {
    auto parsed = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
    if (!parsed) {
      ADD_FAILURE() << "failed to parse test module";
      return {};
    }
    return parsed;
  }

  template <typename OpTy> SmallVector<OpTy> findOps(ModuleOp module) {
    return walkCollect<OpTy>(module);
  }
};

TEST_F(VerifDialectTests, ContractAllowsReadingPrivateMembersOfItsTargetStruct) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  struct.def @SecretBox {
    struct.member @secret : !felt.type
    function.def @compute(%value: !felt.type) -> !struct.type<@SecretBox> {
      %self = struct.new : !struct.type<@SecretBox>
      struct.writem %self[@secret] = %value : !struct.type<@SecretBox>, !felt.type
      function.return %self : !struct.type<@SecretBox>
    }
    function.def @constrain(%self: !struct.type<@SecretBox>, %value: !felt.type) {
      %secret = struct.readm %self[@secret] : !struct.type<@SecretBox>, !felt.type
      constrain.eq %secret, %value : !felt.type
      function.return
    }
  }

  verif.contract @CheckSecret for @SecretBox (%self: !struct.type<@SecretBox>, %value: !felt.type) {
    %secret = struct.readm %self[@secret] : !struct.type<@SecretBox>, !felt.type
    %isEq = bool.cmp eq(%secret, %value) : !felt.type, !felt.type
    verif.ensure_constrain %isEq
  }
}
)mlir";

  auto parsed = parseModule(source);
  auto contracts = findOps<ContractOp>(*parsed);

  ASSERT_EQ(contracts.size(), 1u);
  ASSERT_TRUE(succeeded(mlir::verify(parsed.get())));
  ASSERT_TRUE(verify(contracts.front(), true));
  ASSERT_TRUE(contracts.front().hasStructTarget());
  ASSERT_TRUE(succeeded(contracts.front().getStructTarget()));
  EXPECT_EQ(contracts.front().getStructTarget()->get().getName(), "SecretBox");
}

TEST_F(VerifDialectTests, FunctionTargetContractsAreNotStructContracts) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  function.def @check(%x: index, %y: index) {
    function.return
  }

  verif.contract @Named for @check (%x: index, %y: index) {
    %ok = arith.constant true
    verif.ensure_compute %ok
  }
}
)mlir";

  auto parsed = parseModule(source);
  auto contracts = findOps<ContractOp>(*parsed);

  ASSERT_EQ(contracts.size(), 1u);
  ASSERT_TRUE(succeeded(mlir::verify(parsed.get())));
  EXPECT_FALSE(contracts.front().hasStructTarget());
}

TEST_F(VerifDialectTests, BuilderCreatesContractBodyWithImplicitTerminator) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  function.def @check() {
    function.return
  }
}
)mlir";

  auto parsed = parseModule(source);
  ASSERT_TRUE(parsed);

  OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(parsed->getBody());
  auto contract = builder.create<ContractOp>(
      builder.getUnknownLoc(), "Built", SymbolRefAttr::get(&ctx, "check"),
      FunctionType::get(&ctx, TypeRange {}, TypeRange {}), ArrayAttr()
  );

  auto &body = contract.getBody();
  ASSERT_FALSE(body.empty());
  ASSERT_EQ(body.getBlocks().size(), 1u);
  ASSERT_NE(body.front().getTerminator(), nullptr);
  EXPECT_TRUE(isa<ContractEndOp>(body.front().getTerminator()));
  EXPECT_TRUE(verify(contract, true));
  EXPECT_TRUE(succeeded(mlir::verify(parsed.get())));
}

TEST_F(VerifDialectTests, IncludeVerifiesKnownCalleeContracts) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  function.def @check(%x: index, %ok: i1) {
    function.return
  }

  verif.contract @Base for @check (%x: index, %ok: i1) {
    verif.require_compute %ok
    verif.ensure_constrain %ok
  }

  verif.contract @Wrapper for @check (%x: index, %ok: i1) {
    verif.include @Base(%x, %ok) : (index, i1) -> ()
  }
}
)mlir";

  auto parsed = parseModule(source);
  auto includes = findOps<IncludeOp>(*parsed);

  ASSERT_EQ(includes.size(), 1u);
  ASSERT_TRUE(succeeded(mlir::verify(parsed.get())));
  ASSERT_TRUE(verify(includes.front(), true));
}

TEST_F(VerifDialectTests, IncludeTypeSignatureMatchesOperandsAndResolvesCallable) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  function.def @check(%x: index, %ok: i1) {
    function.return
  }

  verif.contract @Base for @check (%x: index, %ok: i1) {
    verif.require_compute %ok
  }

  verif.contract @Wrapper for @check (%x: index, %ok: i1) {
    verif.include @Base(%x, %ok) : (index, i1) -> ()
  }
}
)mlir";

  auto parsed = parseModule(source);
  auto includes = findOps<IncludeOp>(*parsed);

  ASSERT_EQ(includes.size(), 1u);
  IncludeOp include = includes.front();

  ASSERT_EQ(include.getTypeSignature().getNumInputs(), 2u);
  EXPECT_EQ(include.getTypeSignature().getInput(0), IndexType::get(&ctx));
  EXPECT_EQ(include.getTypeSignature().getInput(1), IntegerType::get(&ctx, 1));

  auto *callable = include.resolveCallable();
  ASSERT_NE(callable, nullptr);
  auto callee = dyn_cast<ContractOp>(callable);
  ASSERT_TRUE(callee);
  EXPECT_EQ(callee.getSymName(), "Base");
}

TEST_F(VerifDialectTests, TemplateIncludeWithExplicitParametersVerifies) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  poly.template @SpecTemplate {
    poly.param @N : index

    function.def @token() {
      %n = poly.read_const @N : index
      function.return
    }

    verif.contract @Base for @SpecTemplate::@token () {
      %ok = arith.constant true
      verif.ensure_compute %ok
    }

    verif.contract @Wrapper for @SpecTemplate::@token () {
      verif.include @SpecTemplate::@Base<[7]>() : () -> ()
    }
  }
}
)mlir";

  auto parsed = parseModule(source);
  auto includes = findOps<IncludeOp>(*parsed);

  ASSERT_EQ(includes.size(), 1u);
  ASSERT_TRUE(succeeded(mlir::verify(parsed.get())));
  ASSERT_TRUE(verify(includes.front(), true));
}

TEST_F(VerifDialectTests, TemplateIncludeAcceptsFieldedLocalSymbolForFieldlessRestriction) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  poly.template @BoxTemplate {
    poly.param @P
    struct.def @Box {
      function.def @compute() -> !struct.type<@BoxTemplate::@Box<[@P]>> {
        %self = struct.new : <@BoxTemplate::@Box<[@P]>>
        function.return %self : !struct.type<@BoxTemplate::@Box<[@P]>>
      }
      function.def @constrain(%self: !struct.type<@BoxTemplate::@Box<[@P]>>) {
        function.return
      }
    }
  }

  poly.template @Target {
    poly.param @F : !felt.type
    function.def @accept(%value: !struct.type<@BoxTemplate::@Box<[@F]>>) {
      function.return
    }
    verif.contract @Base for @Target::@accept (
        %value: !struct.type<@BoxTemplate::@Box<[@F]>>
    ) {
      %ok = arith.constant true
      verif.ensure_compute %ok
    }
  }

  poly.template @Caller {
    poly.param @G : !felt.type<"bn128">
    function.def @caller(%value: !struct.type<@BoxTemplate::@Box<[@G]>>) {
      function.return
    }
    verif.contract @Wrapper for @Caller::@caller (
        %value: !struct.type<@BoxTemplate::@Box<[@G]>>
    ) {
      verif.include @Target::@Base<[@G]>(%value) :
          (!struct.type<@BoxTemplate::@Box<[@G]>>) -> ()
      verif.include @Target::@Base(%value) :
          (!struct.type<@BoxTemplate::@Box<[@G]>>) -> ()
    }
  }
}
)mlir";

  auto parsed = parseModule(source);
  auto includes = findOps<IncludeOp>(*parsed);

  ASSERT_EQ(includes.size(), 2u);
  ASSERT_TRUE(succeeded(mlir::verify(parsed.get())));
  EXPECT_TRUE(verify(includes[0], true));
  EXPECT_TRUE(verify(includes[1], true));
}

TEST_F(VerifDialectTests, IncludeFeltRestrictionNormalizesEquivalentConstants) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  poly.template @BoxTemplate {
    poly.param @P
    struct.def @Box {
      function.def @compute() -> !struct.type<@BoxTemplate::@Box<[@P]>> {
        %self = struct.new : <@BoxTemplate::@Box<[@P]>>
        function.return %self : !struct.type<@BoxTemplate::@Box<[@P]>>
      }
      function.def @constrain(%self: !struct.type<@BoxTemplate::@Box<[@P]>>) {
        function.return
      }
    }
  }

  poly.template @Target {
    poly.param @F : !felt.type<"bn128">
    function.def @accept(%value: !struct.type<@BoxTemplate::@Box<[@F]>>) {
      function.return
    }
    verif.contract @Base for @Target::@accept (
        %value: !struct.type<@BoxTemplate::@Box<[@F]>>
    ) {
      %ok = arith.constant true
      verif.ensure_compute %ok
    }
  }

  poly.template @Caller {
    function.def @caller(
        %value: !struct.type<@BoxTemplate::@Box<[#felt<const 35 : !felt.type<"bn128">>]>>
    ) {
      function.return
    }
    verif.contract @Wrapper for @Caller::@caller (
        %value: !struct.type<@BoxTemplate::@Box<[#felt<const 35 : !felt.type<"bn128">>]>>
    ) {
      verif.include @Target::@Base<[#felt<const 35>]>(%value) :
          (!struct.type<@BoxTemplate::@Box<[#felt<const 35 : !felt.type<"bn128">>]>>) -> ()
      verif.include @Target::@Base<[35]>(%value) :
          (!struct.type<@BoxTemplate::@Box<[#felt<const 35 : !felt.type<"bn128">>]>>) -> ()
    }
  }
}
)mlir";

  auto parsed = parseModule(source);
  auto includes = findOps<IncludeOp>(*parsed);

  ASSERT_EQ(includes.size(), 2u);
  ASSERT_TRUE(succeeded(mlir::verify(parsed.get())));
  EXPECT_TRUE(verify(includes[0], true));
  EXPECT_TRUE(verify(includes[1], true));
}

TEST_F(VerifDialectTests, IncludeOpVerifyTemplateParamsMatchInferredUsesContractSignature) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  poly.template @Target {
    poly.param @F : !felt.type<"bn128">
    function.def @accept() {
      function.return
    }
    verif.contract @Base for @Target::@accept () {
      %ok = arith.constant true
      verif.ensure_compute %ok
    }

    verif.contract @Wrapper for @Target::@accept () {
      verif.include @Target::@Base<[#felt<const 35>]>() : () -> ()
    }
  }
}
)mlir";

  auto parsed = parseModule(source);
  ASSERT_TRUE(parsed);
  auto includes = findOps<IncludeOp>(*parsed);

  ASSERT_EQ(includes.size(), 1u);

  llzk::polymorphic::TemplateOp targetTemplate;
  parsed->walk([&targetTemplate](llzk::polymorphic::TemplateOp op) {
    if (op.getSymName() == "Target") {
      targetTemplate = op;
    }
  });
  ASSERT_NE(targetTemplate.getOperation(), nullptr);

  auto targetParams = targetTemplate.getConstOps<llzk::polymorphic::TemplateParamOp>();
  UnificationMap unifications;
  unifications.try_emplace(
      {FlatSymbolRefAttr::get(&ctx, "F"), Side::RHS},
      llzk::felt::FeltConstAttr::get(&ctx, APInt(8, 36), llzk::felt::FeltType::get(&ctx, "bn128"))
  );

  EXPECT_DEATH(
      {
        if (failed(
                includes.front().verifyTemplateParamsMatchInferred(targetParams, unifications)
            )) {
          std::abort();
        }
      },
      "contract type signature"
  );
}

TEST_F(VerifDialectTests, TemplateIncludeRejectsIncompatibleLocalSymbols) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  poly.template @BoxTemplate {
    poly.param @P
    struct.def @Box {
      function.def @compute() -> !struct.type<@BoxTemplate::@Box<[@P]>> {
        %self = struct.new : <@BoxTemplate::@Box<[@P]>>
        function.return %self : !struct.type<@BoxTemplate::@Box<[@P]>>
      }
      function.def @constrain(%self: !struct.type<@BoxTemplate::@Box<[@P]>>) {
        function.return
      }
    }
  }

  poly.template @Target {
    poly.param @F : !felt.type<"bn128">
    function.def @accept(%value: !struct.type<@BoxTemplate::@Box<[@F]>>) {
      function.return
    }
    verif.contract @Base for @Target::@accept (
        %value: !struct.type<@BoxTemplate::@Box<[@F]>>
    ) {
      %ok = arith.constant true
      verif.ensure_compute %ok
    }
  }

  poly.template @FieldlessCaller {
    poly.param @G : !felt.type
    function.def @caller(%value: !struct.type<@BoxTemplate::@Box<[@G]>>) {
      function.return
    }
    verif.contract @Wrapper for @FieldlessCaller::@caller (
        %value: !struct.type<@BoxTemplate::@Box<[@G]>>
    ) {
      %ok = arith.constant true
      verif.ensure_compute %ok
    }
  }

  poly.template @MismatchedCaller {
    poly.param @G : !felt.type<"goldilocks">
    function.def @caller(%value: !struct.type<@BoxTemplate::@Box<[@G]>>) {
      function.return
    }
    verif.contract @Wrapper for @MismatchedCaller::@caller (
        %value: !struct.type<@BoxTemplate::@Box<[@G]>>
    ) {
      %ok = arith.constant true
      verif.ensure_compute %ok
    }
  }
}
)mlir";

  auto parsed = parseModule(source);
  ASSERT_TRUE(parsed);
  ASSERT_TRUE(succeeded(mlir::verify(parsed.get())));

  auto callee = SymbolRefAttr::get(
      &ctx, "Target", ArrayRef<FlatSymbolRefAttr> {FlatSymbolRefAttr::get(&ctx, "Base")}
  );
  auto param = FlatSymbolRefAttr::get(&ctx, "G");
  SmallVector<IncludeOp> includes;
  parsed->walk([&](ContractOp contract) {
    if (contract.getSymName() != "Wrapper") {
      return;
    }
    Block &body = contract.getBody().front();
    OpBuilder builder(&body, body.getTerminator()->getIterator());
    includes.push_back(builder.create<IncludeOp>(
        loc, callee, ValueRange {body.getArgument(0)}, ArrayRef<Attribute> {param}
    ));
  });

  ASSERT_EQ(includes.size(), 2u);
  EXPECT_FALSE(verify(includes[0], true));
  EXPECT_FALSE(verify(includes[1], true));
}

TEST_F(VerifDialectTests, ContractOutsideTemplateCanReadTargetFunctionTemplateConstants) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  poly.template @T {
    poly.param @N : index

    function.def @token() {
      function.return
    }
  }

  verif.contract @Outside for @T::@token () {
    %n = poly.read_const @N : index
    %zero = arith.constant 0 : index
    %ok = arith.cmpi eq, %n, %zero : index
    verif.ensure_compute %ok
  }
}
)mlir";

  auto parsed = parseModule(source);
  auto contracts = findOps<ContractOp>(*parsed);

  ASSERT_EQ(contracts.size(), 1u);
  ASSERT_TRUE(succeeded(mlir::verify(parsed.get())));
  EXPECT_TRUE(verify(contracts.front(), true));
  EXPECT_FALSE(
      static_cast<bool>(contracts.front()->getParentOfType<llzk::polymorphic::TemplateOp>())
  );
}

TEST_F(VerifDialectTests, ContractOutsideTemplateCanTargetTemplatedStruct) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  poly.template @T {
    poly.param @N : index

    struct.def @Top {
      struct.member @m : !felt.type
      function.def @compute(%a : !felt.type) -> !struct.type<@T::@Top<[@N]>> {
        %self = struct.new : !struct.type<@T::@Top<[@N]>>
        struct.writem %self[@m] = %a : !struct.type<@T::@Top<[@N]>>, !felt.type
        function.return %self : !struct.type<@T::@Top<[@N]>>
      }
      function.def @constrain(%self: !struct.type<@T::@Top<[@N]>>, %a : !felt.type) {
        function.return
      }
    }
  }

  verif.contract @OutsideStruct for @T::@Top (%self: !struct.type<@T::@Top<[@N]>>, %a: !felt.type) {
    %n = poly.read_const @N : index
    %m = struct.readm %self[@m] : !struct.type<@T::@Top<[@N]>>, !felt.type
    %ok = arith.constant true
    verif.ensure_compute %ok
  }
}
)mlir";

  auto parsed = parseModule(source);
  auto contracts = findOps<ContractOp>(*parsed);

  ASSERT_EQ(contracts.size(), 1u);
  ASSERT_TRUE(succeeded(mlir::verify(parsed.get())));
  ASSERT_TRUE(verify(contracts.front(), true));
  ASSERT_TRUE(contracts.front().hasStructTarget());
  ASSERT_TRUE(succeeded(contracts.front().getStructTarget()));
  EXPECT_EQ(contracts.front().getStructTarget()->get().getName(), "Top");
}

TEST_F(VerifDialectTests, CustomBuilderInfersFunctionTargetContractInvariants) {
  auto modBldr = newBasicFunctionsExample(0, 0, {"TargetFn"});
  auto target = modBldr.getFreeFunc("TargetFn");

  ASSERT_TRUE(succeeded(target));

  OpBuilder builder(&ctx);
  builder.setInsertionPointToStart(mod->getBody());

  auto contract = builder.create<ContractOp>(loc, "BuiltContract", "TargetFn");

  EXPECT_EQ(contract.getSymName(), "BuiltContract");
  ASSERT_TRUE(contract.getTargetAttr());
  EXPECT_EQ(contract.getTarget().getRootReference().getValue(), "TargetFn");
  EXPECT_EQ(contract.getFunctionType(), target->getFunctionType());
  EXPECT_EQ(contract.getArgAttrsAttr(), target->getArgAttrsAttr());
  ASSERT_FALSE(contract.getBody().empty());
  ASSERT_EQ(contract.getBody().getBlocks().size(), 1u);
  EXPECT_TRUE(isa<ContractEndOp>(contract.getBody().front().getTerminator()));

  SymbolTable table(*mod);
  table.insert(contract);

  EXPECT_TRUE(verify(contract, true));
  EXPECT_TRUE(contract.hasFuncTarget());
  EXPECT_FALSE(contract.hasStructTarget());
}

TEST_F(VerifDialectTests, CustomBuilderInfersStructTargetSignatureAndArgAttrs) {
  auto modBldr = newStructExample();
  auto targetStruct = modBldr.getStruct(structNameA);
  auto targetConstrain = modBldr.getConstrainFn(structNameA);

  ASSERT_TRUE(succeeded(targetStruct));
  ASSERT_TRUE(succeeded(targetConstrain));

  OpBuilder builder(&ctx);
  builder.setInsertionPointToStart(mod->getBody());

  auto contract = builder.create<ContractOp>(loc, "StructContract", std::string(structNameA));

  EXPECT_EQ(contract.getSymName(), "StructContract");
  ASSERT_TRUE(contract.getTargetAttr());
  EXPECT_EQ(contract.getTarget().getRootReference().getValue(), structNameA);
  EXPECT_EQ(contract.getFunctionType(), targetConstrain->getFunctionType());
  EXPECT_EQ(contract.getArgAttrsAttr(), targetConstrain->getArgAttrsAttr());
  ASSERT_FALSE(contract.getBody().empty());
  ASSERT_EQ(contract.getBody().getBlocks().size(), 1u);
  EXPECT_TRUE(isa<ContractEndOp>(contract.getBody().front().getTerminator()));

  EXPECT_TRUE(verify(contract, true));
  EXPECT_TRUE(contract.hasStructTarget());
  EXPECT_FALSE(contract.hasFuncTarget());
}

TEST_F(VerifDialectTests, SymbolRefBuilderInfersNestedTargetContractInvariants) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  poly.template @T {
    poly.param @N : index

    function.def @TargetFn(%x: !array.type<@N x index>) {
      function.return
    }
  }
}
)mlir";

  auto parsed = parseModule(source);
  ASSERT_TRUE(parsed);
  auto funcs = findOps<llzk::function::FuncDefOp>(*parsed);
  ASSERT_EQ(funcs.size(), 1u);
  auto target = funcs.front();

  OpBuilder builder(&ctx);
  builder.setInsertionPointToStart(parsed->getBody());
  auto targetAttr = SymbolRefAttr::get(
      &ctx, "T", ArrayRef<FlatSymbolRefAttr> {FlatSymbolRefAttr::get(&ctx, "TargetFn")}
  );

  auto contract = builder.create<ContractOp>(loc, "BuiltContract", targetAttr);

  ASSERT_TRUE(contract.getTargetAttr());
  EXPECT_EQ(contract.getTarget(), targetAttr);
  EXPECT_EQ(contract.getFunctionType(), target.getFunctionType());
  EXPECT_EQ(contract.getArgAttrsAttr(), target.getArgAttrsAttr());
  ASSERT_FALSE(contract.getBody().empty());
  ASSERT_EQ(contract.getBody().getBlocks().size(), 1u);
  EXPECT_TRUE(isa<ContractEndOp>(contract.getBody().front().getTerminator()));

  ASSERT_TRUE(verify(contract, true));
  EXPECT_TRUE(contract.hasFuncTarget());
  EXPECT_FALSE(contract.hasStructTarget());
}

TEST_F(VerifDialectTests, StringAndSymbolRefBuildersInferEquivalentFlatTargetInvariants) {
  auto modBldr = newBasicFunctionsExample(0, 0, {"TargetFn"});
  auto target = modBldr.getFreeFunc("TargetFn");

  ASSERT_TRUE(succeeded(target));

  OpBuilder builder(&ctx);
  builder.setInsertionPointToStart(mod->getBody());

  auto byString = builder.create<ContractOp>(loc, "ByString", "TargetFn");
  auto byAttr = builder.create<ContractOp>(loc, "ByAttr", SymbolRefAttr::get(&ctx, "TargetFn"));

  EXPECT_EQ(byString.getTarget(), byAttr.getTarget());
  EXPECT_EQ(byString.getFunctionType(), byAttr.getFunctionType());
  EXPECT_EQ(byString.getArgAttrsAttr(), byAttr.getArgAttrsAttr());
  EXPECT_EQ(byString.getFunctionType(), target->getFunctionType());
  EXPECT_EQ(byString.getArgAttrsAttr(), target->getArgAttrsAttr());
}

TEST_F(VerifDialectTests, CustomBuilderFailure) {
  auto modBldr = newStructExample();

  OpBuilder builder(&ctx);
  builder.setInsertionPointToStart(mod->getBody());

  // Build does not fail, since we defer to verify
  auto contract = builder.create<ContractOp>(loc, "StructContract", "UnknownTarget");
  ASSERT_NE(contract, nullptr);
  // But verify will fail, since the contract op was not properly built due to
  // target lookup failures.
  ASSERT_FALSE(verify(contract, true));
}

} // namespace
