/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "LLZK", "index.html", [
    [ "Overview", "index.html", "index" ],
    [ "Architecture", "overview.html", [
      [ "Project Overview", "overview.html#project-overview", [
        [ "LLZK IR Dialects", "overview.html#llzk-ir-dialects", null ],
        [ "Frontends", "overview.html#frontends", null ],
        [ "Passes", "overview.html#pass-overview", null ],
        [ "Backends", "overview.html#backends", null ]
      ] ]
    ] ],
    [ "Setup", "setup.html", [
      [ "Nix Setup", "setup.html#nix-setup", null ],
      [ "Manual Build Setup", "setup.html#manual-build-setup", null ],
      [ "Development Workflow", "setup.html#dev-workflow", null ]
    ] ],
    [ "Tool Guides", "tools.html", [
      [ "llzk-opt", "tools.html#llzk-opt", [
        [ "LLZK Pass Documentation", "tools.html#passes", [
          [ "Analysis Passes", "tools.html#analysis-passes", [
            [ "<span class=\"tt\">-llzk-print-call-graph</span>", "tools.html#autotoc_md-llzk-print-call-graph", [
              [ "Options", "tools.html#options", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-print-call-graph-sccs</span>", "tools.html#autotoc_md-llzk-print-call-graph-sccs", [
              [ "Options", "tools.html#options-1", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-print-constraint-dependency-graphs</span>", "tools.html#autotoc_md-llzk-print-constraint-dependency-graphs", [
              [ "Options", "tools.html#options-2", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-print-interval-analysis</span>", "tools.html#autotoc_md-llzk-print-interval-analysis", [
              [ "Options", "tools.html#options-3", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-print-predecessors</span>", "tools.html#autotoc_md-llzk-print-predecessors", [
              [ "Options", "tools.html#options-4", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-print-symbol-def-tree</span>", "tools.html#autotoc_md-llzk-print-symbol-def-tree", [
              [ "Options", "tools.html#options-5", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-print-symbol-use-graph</span>", "tools.html#autotoc_md-llzk-print-symbol-use-graph", [
              [ "Options", "tools.html#options-6", null ]
            ] ]
          ] ],
          [ "Transformation Passes", "tools.html#transformation-passes", [
            [ "<span class=\"tt\">-llzk-compute-constrain-to-product</span>", "tools.html#autotoc_md-llzk-compute-constrain-to-product", [
              [ "Options", "tools.html#options-7", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-duplicate-op-elim</span>", "tools.html#autotoc_md-llzk-duplicate-op-elim", null ],
            [ "<span class=\"tt\">-llzk-duplicate-read-write-elim</span>", "tools.html#autotoc_md-llzk-duplicate-read-write-elim", null ],
            [ "<span class=\"tt\">-llzk-enforce-no-overwrite</span>", "tools.html#autotoc_md-llzk-enforce-no-overwrite", null ],
            [ "<span class=\"tt\">-llzk-fuse-product-control-flow</span>", "tools.html#autotoc_md-llzk-fuse-product-control-flow", null ],
            [ "<span class=\"tt\">-llzk-inline-free-functions</span>", "tools.html#autotoc_md-llzk-inline-free-functions", null ],
            [ "<span class=\"tt\">-llzk-poly-lowering-pass</span>", "tools.html#autotoc_md-llzk-poly-lowering-pass", [
              [ "Options", "tools.html#options-8", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-remove-unused-discardable-allocations</span>", "tools.html#autotoc_md-llzk-remove-unused-discardable-allocations", [
              [ "Options", "tools.html#options-9", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-unused-declaration-elim</span>", "tools.html#autotoc_md-llzk-unused-declaration-elim", [
              [ "Options", "tools.html#options-10", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-while-to-for</span>", "tools.html#autotoc_md-llzk-while-to-for", null ],
            [ "<span class=\"tt\">-llzk-array-to-scalar</span>", "tools.html#autotoc_md-llzk-array-to-scalar", null ],
            [ "<span class=\"tt\">-llzk-straight-line-static-array-promotion</span>", "tools.html#autotoc_md-llzk-straight-line-static-array-promotion", null ],
            [ "<span class=\"tt\">-llzk-lower-bool-quantifiers</span>", "tools.html#autotoc_md-llzk-lower-bool-quantifiers", null ],
            [ "<span class=\"tt\">-llzk-inline-includes</span>", "tools.html#autotoc_md-llzk-inline-includes", null ],
            [ "<span class=\"tt\">-pcl-trim-expression-size</span>", "tools.html#autotoc_md-pcl-trim-expression-size", [
              [ "Options", "tools.html#options-11", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-pod-to-scalar</span>", "tools.html#autotoc_md-llzk-pod-to-scalar", null ],
            [ "<span class=\"tt\">-llzk-drop-empty-templates</span>", "tools.html#autotoc_md-llzk-drop-empty-templates", null ],
            [ "<span class=\"tt\">-llzk-flatten</span>", "tools.html#autotoc_md-llzk-flatten", [
              [ "Options", "tools.html#options-12", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-infer-tvar</span>", "tools.html#autotoc_md-llzk-infer-tvar", null ],
            [ "<span class=\"tt\">-llzk-specialize-wildcard-arrays</span>", "tools.html#autotoc_md-llzk-specialize-wildcard-arrays", [
              [ "Options", "tools.html#options-13", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-inline-structs</span>", "tools.html#autotoc_md-llzk-inline-structs", [
              [ "Options", "tools.html#options-14", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-to-pcl</span>", "tools.html#autotoc_md-llzk-to-pcl", [
              [ "Options", "tools.html#options-15", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-r1cs-lowering</span>", "tools.html#autotoc_md-llzk-r1cs-lowering", null ]
          ] ],
          [ "Validation Passes", "tools.html#validation-passes", [
            [ "<span class=\"tt\">-llzk-validate-member-writes</span>", "tools.html#autotoc_md-llzk-validate-member-writes", null ]
          ] ],
          [ "Predefined Pipelines", "tools.html#predefined-pipelines", [
            [ "<span class=\"tt\">-llzk-remove-unnecessary-ops</span>", "tools.html#autotoc_md-llzk-remove-unnecessary-ops", null ],
            [ "<span class=\"tt\">-llzk-remove-unnecessary-ops-and-defs</span>", "tools.html#autotoc_md-llzk-remove-unnecessary-ops-and-defs", null ],
            [ "<span class=\"tt\">-llzk-product-program</span>", "tools.html#autotoc_md-llzk-product-program", null ],
            [ "<span class=\"tt\">-llzk-full-struct-inlining</span>", "tools.html#autotoc_md-llzk-full-struct-inlining", [
              [ "Options", "tools.html#options-16", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-full-inlining</span>", "tools.html#autotoc_md-llzk-full-inlining", null ],
            [ "<span class=\"tt\">-llzk-full-poly-lowering</span>", "tools.html#autotoc_md-llzk-full-poly-lowering", [
              [ "Options", "tools.html#options-17", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-full-r1cs-lowering</span>", "tools.html#autotoc_md-llzk-full-r1cs-lowering", null ]
          ] ]
        ] ]
      ] ],
      [ "llzk-translate", "tools.html#llzk-translate", null ],
      [ "llzk-witgen", "tools.html#llzk-witgen", null ],
      [ "llzk-smt-check", "tools.html#llzk-smt-check", null ],
      [ "llzk-lsp-server", "tools.html#llzk-lsp-server", null ]
    ] ],
    [ "LLZK Language Specification", "syntax.html", [
      [ "Syntax", "syntax.html#syntax-1", null ],
      [ "Types", "syntax.html#types", [
        [ "Pseudo-homogeneous arrays", "syntax.html#pseudo-homogeneous", null ]
      ] ],
      [ "Semantic Rules", "syntax.html#semantic-rules", null ],
      [ "Translation Guidelines", "syntax.html#translation-guidelines", null ]
    ] ],
    [ "Contribution Guide", "contribution-guide.html", "contribution-guide" ],
    [ "Maintenance Guide", "maintanence.html", [
      [ "Tracking a New Version", "maintanence.html#tracking-a-new-version", [
        [ "Release version", "maintanence.html#release-version", null ],
        [ "Patches", "maintanence.html#patches", null ]
      ] ],
      [ "Releasing a New Version", "maintanence.html#releasing-a-new-version", [
        [ "Preparing a new release", "maintanence.html#preparing-a-new-release", null ],
        [ "Creating the Release Candidate", "maintanence.html#creating-the-release-candidate", null ],
        [ "Create the Release", "maintanence.html#create-the-release", null ]
      ] ]
    ] ],
    [ "LLZK IR Dialects", "dialects.html", [
      [ "'array' Dialect", "dialects.html#array-dialect", [
        [ "Operations", "dialects.html#operations", [
          [ "<span class=\"tt\">array.extract</span> (llzk::array::ExtractArrayOp)", "dialects.html#arrayextract-llzkarrayextractarrayop", [
            [ "Operands:", "dialects.html#operands", null ],
            [ "Results:", "dialects.html#results", null ]
          ] ],
          [ "<span class=\"tt\">array.insert</span> (llzk::array::InsertArrayOp)", "dialects.html#arrayinsert-llzkarrayinsertarrayop", [
            [ "Operands:", "dialects.html#operands-1", null ]
          ] ],
          [ "<span class=\"tt\">array.len</span> (llzk::array::ArrayLengthOp)", "dialects.html#arraylen-llzkarrayarraylengthop", [
            [ "Operands:", "dialects.html#operands-2", null ],
            [ "Results:", "dialects.html#results-1", null ]
          ] ],
          [ "<span class=\"tt\">array.new</span> (llzk::array::CreateArrayOp)", "dialects.html#arraynew-llzkarraycreatearrayop", [
            [ "Attributes:", "dialects.html#attributes", null ],
            [ "Operands:", "dialects.html#operands-3", null ],
            [ "Results:", "dialects.html#results-2", null ]
          ] ],
          [ "<span class=\"tt\">array.read</span> (llzk::array::ReadArrayOp)", "dialects.html#arrayread-llzkarrayreadarrayop", [
            [ "Operands:", "dialects.html#operands-4", null ],
            [ "Results:", "dialects.html#results-3", null ]
          ] ],
          [ "<span class=\"tt\">array.write</span> (llzk::array::WriteArrayOp)", "dialects.html#arraywrite-llzkarraywritearrayop", [
            [ "Operands:", "dialects.html#operands-5", null ]
          ] ]
        ] ],
        [ "Types", "dialects.html#types-1", [
          [ "ArrayType", "dialects.html#arraytype", [
            [ "Parameters:", "dialects.html#parameters", null ]
          ] ]
        ] ]
      ] ],
      [ "'bool' Dialect", "dialects.html#bool-dialect", [
        [ "Operations", "dialects.html#operations-1", [
          [ "<span class=\"tt\">bool.and</span> (llzk::boolean::AndBoolOp)", "dialects.html#booland-llzkbooleanandboolop", [
            [ "Operands:", "dialects.html#operands-6", null ],
            [ "Results:", "dialects.html#results-4", null ]
          ] ],
          [ "<span class=\"tt\">bool.assert</span> (llzk::boolean::AssertOp)", "dialects.html#boolassert-llzkbooleanassertop", [
            [ "Attributes:", "dialects.html#attributes-1", null ],
            [ "Operands:", "dialects.html#operands-7", null ]
          ] ],
          [ "<span class=\"tt\">bool.cmp</span> (llzk::boolean::CmpOp)", "dialects.html#boolcmp-llzkbooleancmpop", [
            [ "Attributes:", "dialects.html#attributes-2", null ],
            [ "Operands:", "dialects.html#operands-8", null ],
            [ "Results:", "dialects.html#results-5", null ]
          ] ],
          [ "<span class=\"tt\">bool.exists</span> (llzk::boolean::ExistsOp)", "dialects.html#boolexists-llzkbooleanexistsop", [
            [ "Operands:", "dialects.html#operands-9", null ],
            [ "Results:", "dialects.html#results-6", null ]
          ] ],
          [ "<span class=\"tt\">bool.forall</span> (llzk::boolean::ForAllOp)", "dialects.html#boolforall-llzkbooleanforallop", [
            [ "Operands:", "dialects.html#operands-10", null ],
            [ "Results:", "dialects.html#results-7", null ]
          ] ],
          [ "<span class=\"tt\">bool.not</span> (llzk::boolean::NotBoolOp)", "dialects.html#boolnot-llzkbooleannotboolop", [
            [ "Operands:", "dialects.html#operands-11", null ],
            [ "Results:", "dialects.html#results-8", null ]
          ] ],
          [ "<span class=\"tt\">bool.or</span> (llzk::boolean::OrBoolOp)", "dialects.html#boolor-llzkbooleanorboolop", [
            [ "Operands:", "dialects.html#operands-12", null ],
            [ "Results:", "dialects.html#results-9", null ]
          ] ],
          [ "<span class=\"tt\">bool.xor</span> (llzk::boolean::XorBoolOp)", "dialects.html#boolxor-llzkbooleanxorboolop", [
            [ "Operands:", "dialects.html#operands-13", null ],
            [ "Results:", "dialects.html#results-10", null ]
          ] ],
          [ "<span class=\"tt\">bool.yield</span> (llzk::boolean::YieldOp)", "dialects.html#boolyield-llzkbooleanyieldop", [
            [ "Operands:", "dialects.html#operands-14", null ]
          ] ]
        ] ],
        [ "Attributes", "dialects.html#attributes-3", [
          [ "FeltCmpPredicateAttr", "dialects.html#feltcmppredicateattr", [
            [ "Parameters:", "dialects.html#parameters-1", null ]
          ] ]
        ] ],
        [ "Enums", "dialects.html#enums", [
          [ "FeltCmpPredicate", "dialects.html#feltcmppredicate", [
            [ "Cases:", "dialects.html#cases", null ]
          ] ]
        ] ]
      ] ],
      [ "'cast' Dialect", "dialects.html#cast-dialect", [
        [ "Operations", "dialects.html#operations-2", [
          [ "<span class=\"tt\">cast.tofelt</span> (llzk::cast::IntToFeltOp)", "dialects.html#casttofelt-llzkcastinttofeltop", [
            [ "Attributes:", "dialects.html#attributes-4", null ],
            [ "Operands:", "dialects.html#operands-15", null ],
            [ "Results:", "dialects.html#results-11", null ]
          ] ],
          [ "<span class=\"tt\">cast.toindex</span> (llzk::cast::FeltToIndexOp)", "dialects.html#casttoindex-llzkcastfelttoindexop", [
            [ "Attributes:", "dialects.html#attributes-5", null ],
            [ "Operands:", "dialects.html#operands-16", null ],
            [ "Results:", "dialects.html#results-12", null ]
          ] ]
        ] ],
        [ "Attributes", "dialects.html#attributes-6", [
          [ "OverflowSemanticsAttr", "dialects.html#overflowsemanticsattr", [
            [ "Parameters:", "dialects.html#parameters-2", null ]
          ] ]
        ] ],
        [ "Enums", "dialects.html#enums-1", [
          [ "OverflowSemantics", "dialects.html#overflowsemantics", [
            [ "Cases:", "dialects.html#cases-1", null ]
          ] ]
        ] ]
      ] ],
      [ "'constrain' Dialect", "dialects.html#constrain-dialect", [
        [ "Operations", "dialects.html#operations-3", [
          [ "<span class=\"tt\">constrain.eq</span> (llzk::constrain::EmitEqualityOp)", "dialects.html#constraineq-llzkconstrainemitequalityop", [
            [ "Operands:", "dialects.html#operands-17", null ]
          ] ],
          [ "<span class=\"tt\">constrain.in</span> (llzk::constrain::EmitContainmentOp)", "dialects.html#constrainin-llzkconstrainemitcontainmentop", [
            [ "Operands:", "dialects.html#operands-18", null ]
          ] ]
        ] ]
      ] ],
      [ "'felt' Dialect", "dialects.html#felt-dialect", [
        [ "Operations", "dialects.html#operations-4", [
          [ "<span class=\"tt\">felt.add</span> (llzk::felt::AddFeltOp)", "dialects.html#feltadd-llzkfeltaddfeltop", [
            [ "Operands:", "dialects.html#operands-19", null ],
            [ "Results:", "dialects.html#results-13", null ]
          ] ],
          [ "<span class=\"tt\">felt.bit_and</span> (llzk::felt::AndFeltOp)", "dialects.html#feltbit_and-llzkfeltandfeltop", [
            [ "Operands:", "dialects.html#operands-20", null ],
            [ "Results:", "dialects.html#results-14", null ]
          ] ],
          [ "<span class=\"tt\">felt.bit_not</span> (llzk::felt::NotFeltOp)", "dialects.html#feltbit_not-llzkfeltnotfeltop", [
            [ "Operands:", "dialects.html#operands-21", null ],
            [ "Results:", "dialects.html#results-15", null ]
          ] ],
          [ "<span class=\"tt\">felt.bit_or</span> (llzk::felt::OrFeltOp)", "dialects.html#feltbit_or-llzkfeltorfeltop", [
            [ "Operands:", "dialects.html#operands-22", null ],
            [ "Results:", "dialects.html#results-16", null ]
          ] ],
          [ "<span class=\"tt\">felt.bit_xor</span> (llzk::felt::XorFeltOp)", "dialects.html#feltbit_xor-llzkfeltxorfeltop", [
            [ "Operands:", "dialects.html#operands-23", null ],
            [ "Results:", "dialects.html#results-17", null ]
          ] ],
          [ "<span class=\"tt\">felt.const</span> (llzk::felt::FeltConstantOp)", "dialects.html#feltconst-llzkfeltfeltconstantop", [
            [ "Attributes:", "dialects.html#attributes-7", null ],
            [ "Results:", "dialects.html#results-18", null ]
          ] ],
          [ "<span class=\"tt\">felt.div</span> (llzk::felt::DivFeltOp)", "dialects.html#feltdiv-llzkfeltdivfeltop", [
            [ "Operands:", "dialects.html#operands-24", null ],
            [ "Results:", "dialects.html#results-19", null ]
          ] ],
          [ "<span class=\"tt\">felt.inv</span> (llzk::felt::InvFeltOp)", "dialects.html#feltinv-llzkfeltinvfeltop", [
            [ "Operands:", "dialects.html#operands-25", null ],
            [ "Results:", "dialects.html#results-20", null ]
          ] ],
          [ "<span class=\"tt\">felt.mul</span> (llzk::felt::MulFeltOp)", "dialects.html#feltmul-llzkfeltmulfeltop", [
            [ "Operands:", "dialects.html#operands-26", null ],
            [ "Results:", "dialects.html#results-21", null ]
          ] ],
          [ "<span class=\"tt\">felt.neg</span> (llzk::felt::NegFeltOp)", "dialects.html#feltneg-llzkfeltnegfeltop", [
            [ "Operands:", "dialects.html#operands-27", null ],
            [ "Results:", "dialects.html#results-22", null ]
          ] ],
          [ "<span class=\"tt\">felt.pow</span> (llzk::felt::PowFeltOp)", "dialects.html#feltpow-llzkfeltpowfeltop", [
            [ "Operands:", "dialects.html#operands-28", null ],
            [ "Results:", "dialects.html#results-23", null ]
          ] ],
          [ "<span class=\"tt\">felt.shl</span> (llzk::felt::ShlFeltOp)", "dialects.html#feltshl-llzkfeltshlfeltop", [
            [ "Operands:", "dialects.html#operands-29", null ],
            [ "Results:", "dialects.html#results-24", null ]
          ] ],
          [ "<span class=\"tt\">felt.shr</span> (llzk::felt::ShrFeltOp)", "dialects.html#feltshr-llzkfeltshrfeltop", [
            [ "Operands:", "dialects.html#operands-30", null ],
            [ "Results:", "dialects.html#results-25", null ]
          ] ],
          [ "<span class=\"tt\">felt.sintdiv</span> (llzk::felt::SignedIntDivFeltOp)", "dialects.html#feltsintdiv-llzkfeltsignedintdivfeltop", [
            [ "Operands:", "dialects.html#operands-31", null ],
            [ "Results:", "dialects.html#results-26", null ]
          ] ],
          [ "<span class=\"tt\">felt.smod</span> (llzk::felt::SignedModFeltOp)", "dialects.html#feltsmod-llzkfeltsignedmodfeltop", [
            [ "Operands:", "dialects.html#operands-32", null ],
            [ "Results:", "dialects.html#results-27", null ]
          ] ],
          [ "<span class=\"tt\">felt.sub</span> (llzk::felt::SubFeltOp)", "dialects.html#feltsub-llzkfeltsubfeltop", [
            [ "Operands:", "dialects.html#operands-33", null ],
            [ "Results:", "dialects.html#results-28", null ]
          ] ],
          [ "<span class=\"tt\">felt.uintdiv</span> (llzk::felt::UnsignedIntDivFeltOp)", "dialects.html#feltuintdiv-llzkfeltunsignedintdivfeltop", [
            [ "Operands:", "dialects.html#operands-34", null ],
            [ "Results:", "dialects.html#results-29", null ]
          ] ],
          [ "<span class=\"tt\">felt.umod</span> (llzk::felt::UnsignedModFeltOp)", "dialects.html#feltumod-llzkfeltunsignedmodfeltop", [
            [ "Operands:", "dialects.html#operands-35", null ],
            [ "Results:", "dialects.html#results-30", null ]
          ] ]
        ] ],
        [ "Attributes", "dialects.html#attributes-8", [
          [ "FeltConstAttr", "dialects.html#feltconstattr", [
            [ "Parameters:", "dialects.html#parameters-3", null ]
          ] ],
          [ "FieldSpecAttr", "dialects.html#fieldspecattr", [
            [ "Parameters:", "dialects.html#parameters-4", null ]
          ] ]
        ] ],
        [ "Types", "dialects.html#types-2", [
          [ "FeltType", "dialects.html#felttype", [
            [ "Parameters:", "dialects.html#parameters-5", null ]
          ] ]
        ] ]
      ] ],
      [ "'function' Dialect", "dialects.html#function-dialect", [
        [ "Operations", "dialects.html#operations-5", [
          [ "<span class=\"tt\">function.call</span> (llzk::function::CallOp)", "dialects.html#functioncall-llzkfunctioncallop", [
            [ "Attributes:", "dialects.html#attributes-9", null ],
            [ "Operands:", "dialects.html#operands-36", null ],
            [ "Results:", "dialects.html#results-31", null ]
          ] ],
          [ "<span class=\"tt\">function.def</span> (llzk::function::FuncDefOp)", "dialects.html#functiondef-llzkfunctionfuncdefop", [
            [ "Attributes:", "dialects.html#attributes-10", null ]
          ] ],
          [ "<span class=\"tt\">function.return</span> (llzk::function::ReturnOp)", "dialects.html#functionreturn-llzkfunctionreturnop", [
            [ "Operands:", "dialects.html#operands-37", null ]
          ] ]
        ] ]
      ] ],
      [ "'global' Dialect", "dialects.html#global-dialect", [
        [ "Operations", "dialects.html#operations-6", [
          [ "<span class=\"tt\">global.def</span> (llzk::global::GlobalDefOp)", "dialects.html#globaldef-llzkglobalglobaldefop", [
            [ "Attributes:", "dialects.html#attributes-11", null ]
          ] ],
          [ "<span class=\"tt\">global.read</span> (llzk::global::GlobalReadOp)", "dialects.html#globalread-llzkglobalglobalreadop", [
            [ "Attributes:", "dialects.html#attributes-12", null ],
            [ "Results:", "dialects.html#results-32", null ]
          ] ],
          [ "<span class=\"tt\">global.write</span> (llzk::global::GlobalWriteOp)", "dialects.html#globalwrite-llzkglobalglobalwriteop", [
            [ "Attributes:", "dialects.html#attributes-13", null ],
            [ "Operands:", "dialects.html#operands-38", null ]
          ] ]
        ] ]
      ] ],
      [ "'include' Dialect", "dialects.html#include-dialect", [
        [ "Operations", "dialects.html#operations-7", [
          [ "<span class=\"tt\">include.from</span> (llzk::include::IncludeOp)", "dialects.html#includefrom-llzkincludeincludeop", [
            [ "Attributes:", "dialects.html#attributes-14", null ]
          ] ]
        ] ]
      ] ],
      [ "'llzk' Dialect", "dialects.html#llzk-dialect", [
        [ "Operations", "dialects.html#operations-8", [
          [ "<span class=\"tt\">llzk.nondet</span> (llzk::NonDetOp)", "dialects.html#llzknondet-llzknondetop", [
            [ "Results:", "dialects.html#results-33", null ]
          ] ]
        ] ],
        [ "Attributes", "dialects.html#attributes-15", [
          [ "LoopBoundsAttr", "dialects.html#loopboundsattr", [
            [ "Parameters:", "dialects.html#parameters-6", null ]
          ] ],
          [ "PublicAttr", "dialects.html#publicattr", null ]
        ] ]
      ] ],
      [ "'pod' Dialect", "dialects.html#pod-dialect", [
        [ "Operations", "dialects.html#operations-9", [
          [ "<span class=\"tt\">pod.new</span> (llzk::pod::NewPodOp)", "dialects.html#podnew-llzkpodnewpodop", [
            [ "Attributes:", "dialects.html#attributes-16", null ],
            [ "Operands:", "dialects.html#operands-39", null ],
            [ "Results:", "dialects.html#results-34", null ]
          ] ],
          [ "<span class=\"tt\">pod.read</span> (llzk::pod::ReadPodOp)", "dialects.html#podread-llzkpodreadpodop", [
            [ "Attributes:", "dialects.html#attributes-17", null ],
            [ "Operands:", "dialects.html#operands-40", null ],
            [ "Results:", "dialects.html#results-35", null ]
          ] ],
          [ "<span class=\"tt\">pod.write</span> (llzk::pod::WritePodOp)", "dialects.html#podwrite-llzkpodwritepodop", [
            [ "Attributes:", "dialects.html#attributes-18", null ],
            [ "Operands:", "dialects.html#operands-41", null ]
          ] ]
        ] ],
        [ "Attributes", "dialects.html#attributes-19", [
          [ "RecordAttr", "dialects.html#recordattr", [
            [ "Parameters:", "dialects.html#parameters-7", null ]
          ] ]
        ] ],
        [ "Types", "dialects.html#types-3", [
          [ "PodType", "dialects.html#podtype", [
            [ "Parameters:", "dialects.html#parameters-8", null ]
          ] ]
        ] ]
      ] ],
      [ "'poly' Dialect", "dialects.html#poly-dialect", [
        [ "Operations", "dialects.html#operations-10", [
          [ "<span class=\"tt\">poly.applymap</span> (llzk::polymorphic::ApplyMapOp)", "dialects.html#polyapplymap-llzkpolymorphicapplymapop", [
            [ "Attributes:", "dialects.html#attributes-20", null ],
            [ "Operands:", "dialects.html#operands-42", null ],
            [ "Results:", "dialects.html#results-36", null ]
          ] ],
          [ "<span class=\"tt\">poly.expr</span> (llzk::polymorphic::TemplateExprOp)", "dialects.html#polyexpr-llzkpolymorphictemplateexprop", [
            [ "Attributes:", "dialects.html#attributes-21", null ]
          ] ],
          [ "<span class=\"tt\">poly.param</span> (llzk::polymorphic::TemplateParamOp)", "dialects.html#polyparam-llzkpolymorphictemplateparamop", [
            [ "Attributes:", "dialects.html#attributes-22", null ]
          ] ],
          [ "<span class=\"tt\">poly.read_const</span> (llzk::polymorphic::ConstReadOp)", "dialects.html#polyread_const-llzkpolymorphicconstreadop", [
            [ "Attributes:", "dialects.html#attributes-23", null ],
            [ "Results:", "dialects.html#results-37", null ]
          ] ],
          [ "<span class=\"tt\">poly.template</span> (llzk::polymorphic::TemplateOp)", "dialects.html#polytemplate-llzkpolymorphictemplateop", [
            [ "Attributes:", "dialects.html#attributes-24", null ]
          ] ],
          [ "<span class=\"tt\">poly.unifiable_cast</span> (llzk::polymorphic::UnifiableCastOp)", "dialects.html#polyunifiable_cast-llzkpolymorphicunifiablecastop", [
            [ "Operands:", "dialects.html#operands-43", null ],
            [ "Results:", "dialects.html#results-38", null ]
          ] ],
          [ "<span class=\"tt\">poly.yield</span> (llzk::polymorphic::YieldOp)", "dialects.html#polyyield-llzkpolymorphicyieldop", [
            [ "Operands:", "dialects.html#operands-44", null ]
          ] ]
        ] ],
        [ "Types", "dialects.html#types-4", [
          [ "TypeVarType", "dialects.html#typevartype", [
            [ "Parameters:", "dialects.html#parameters-9", null ]
          ] ]
        ] ]
      ] ],
      [ "'string' Dialect", "dialects.html#string-dialect", [
        [ "Operations", "dialects.html#operations-11", [
          [ "<span class=\"tt\">string.new</span> (llzk::string::LitStringOp)", "dialects.html#stringnew-llzkstringlitstringop", [
            [ "Attributes:", "dialects.html#attributes-25", null ],
            [ "Results:", "dialects.html#results-39", null ]
          ] ]
        ] ],
        [ "Types", "dialects.html#types-5", [
          [ "StringType", "dialects.html#stringtype", null ]
        ] ]
      ] ],
      [ "'struct' Dialect", "dialects.html#struct-dialect", [
        [ "Operations", "dialects.html#operations-12", [
          [ "<span class=\"tt\">struct.def</span> (llzk::component::StructDefOp)", "dialects.html#structdef-llzkcomponentstructdefop", [
            [ "Attributes:", "dialects.html#attributes-26", null ]
          ] ],
          [ "<span class=\"tt\">struct.member</span> (llzk::component::MemberDefOp)", "dialects.html#structmember-llzkcomponentmemberdefop", [
            [ "Attributes:", "dialects.html#attributes-27", null ]
          ] ],
          [ "<span class=\"tt\">struct.new</span> (llzk::component::CreateStructOp)", "dialects.html#structnew-llzkcomponentcreatestructop", [
            [ "Results:", "dialects.html#results-40", null ]
          ] ],
          [ "<span class=\"tt\">struct.readm</span> (llzk::component::MemberReadOp)", "dialects.html#structreadm-llzkcomponentmemberreadop", [
            [ "Attributes:", "dialects.html#attributes-28", null ],
            [ "Operands:", "dialects.html#operands-45", null ],
            [ "Results:", "dialects.html#results-41", null ]
          ] ],
          [ "<span class=\"tt\">struct.writem</span> (llzk::component::MemberWriteOp)", "dialects.html#structwritem-llzkcomponentmemberwriteop", [
            [ "Attributes:", "dialects.html#attributes-29", null ],
            [ "Operands:", "dialects.html#operands-46", null ]
          ] ]
        ] ],
        [ "Types", "dialects.html#types-6", [
          [ "StructType", "dialects.html#structtype", [
            [ "Parameters:", "dialects.html#parameters-10", null ]
          ] ]
        ] ]
      ] ]
    ] ],
    [ "Backends", "backends-page.html", "backends-page" ],
    [ "WTNS witness output", "md_doc_2doxygen_214__wtns__format.html", [
      [ "File schema", "md_doc_2doxygen_214__wtns__format.html#file-schema", null ],
      [ "Wire ordering", "md_doc_2doxygen_214__wtns__format.html#wire-ordering", null ]
    ] ],
    [ "Todo List", "todo.html", null ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", "namespacemembers_dup" ],
        [ "Functions", "namespacemembers_func.html", "namespacemembers_func" ],
        [ "Variables", "namespacemembers_vars.html", null ],
        [ "Typedefs", "namespacemembers_type.html", null ],
        [ "Enumerations", "namespacemembers_enum.html", null ]
      ] ]
    ] ],
    [ "Concepts", "concepts.html", "concepts" ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", "functions_vars" ],
        [ "Typedefs", "functions_type.html", "functions_type" ],
        [ "Enumerations", "functions_enum.html", null ],
        [ "Related Symbols", "functions_rela.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", "globals_dup" ],
        [ "Functions", "globals_func.html", "globals_func" ],
        [ "Variables", "globals_vars.html", null ],
        [ "Typedefs", "globals_type.html", null ],
        [ "Enumerations", "globals_enum.html", null ],
        [ "Enumerator", "globals_eval.html", null ],
        [ "Macros", "globals_defs.html", "globals_defs" ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"AbstractLatticeValue_8h.html",
"Array_2IR_2Ops_8capi_8test_8cpp_8inc.html#aeb7f43022e6c7eb93d6e3043875f6ac8",
"Bool_2IR_2Ops_8capi_8h_8inc.html#a55482a01e0780eb0cd2a876ebcfcdde7",
"Cast_2IR_2Enums_8capi_8test_8cpp_8inc.html",
"Constrain_2IR_2Ops_8h.html#a52c883a38e577b8b12282ff70c95a16d",
"Felt_2IR_2Ops_8capi_8h_8inc.html",
"Felt_2IR_2Ops_8capi_8test_8cpp_8inc.html#af0779dbfa9cfca6732a1e85f86e3203e",
"Function_2IR_2Ops_8capi_8h_8inc.html#a427c96019a2ae1dc2f02022462638299",
"Global_2IR_2Ops_8capi_8h_8inc.html#ad2afa6d0e2d660f6f65b4b99d01dc539",
"LLZKTransformationPasses_8capi_8cpp_8inc.html#aae30325e9ba10693aff79390d57ed64f",
"POD_2IR_2Dialect_8h_8inc.html",
"Poly_8h_source.html",
"RAM_2IR_2Ops_8capi_8cpp_8inc.html#a23cb6472e5a48a8fbcdc94978a25b879",
"Struct_2IR_2Ops_8capi_8cpp_8inc.html#a0ae4a7c367353c8846241c63a744df95",
"Struct_8h.html#a22e2006591a89a7e8604d19703014881",
"Verif_2IR_2Ops_8capi_8cpp_8inc.html#ad7c2140ce4dbc615e51941865484ff9b",
"Verif_8cpp.html#a280966befd8991bb34589375d6c54f9e",
"classllzk_1_1CallGraphNode_1_1Edge.html#acf84371a995da42becbcda5585f063a4",
"classllzk_1_1InFlightDiagnosticWrapper.html#a9b46dfdc9b44c6b80ccce64e4768794d",
"classllzk_1_1MemberOverwriteLattice.html#a2314434027ef2b8b65f603a5d7f9986e",
"classllzk_1_1PredecessorAnalysis.html#a0d287a6af61462f40faebf62b42397d4",
"classllzk_1_1SourceRefLatticeValue.html#aceaa981713550fcca3e5fcaea6b788d3",
"classllzk_1_1SymbolUseGraph.html#aeab6a93bbcd3a8536194b6c7fb891267",
"classllzk_1_1array_1_1ArrayLengthOpAdaptor.html#a2d84f073dddc00263412bc74b71ed341",
"classllzk_1_1array_1_1ExtractArrayOpGenericAdaptor.html#a9ab1f4f17ae9acbe35e8a582b213afae",
"classllzk_1_1array_1_1WriteArrayOpAdaptor.html#a2e12b4ecf11e120909888927193781d5",
"classllzk_1_1boolean_1_1AndBoolOp.html#acd87d450011cda7eb06c5edf7f0ebc41",
"classllzk_1_1boolean_1_1ExistsOp.html#a6a7dab6f4ec36fdd33b2503718722b68",
"classllzk_1_1boolean_1_1OrBoolOpGenericAdaptor.html#aa3eb8e3dca52e4e5ccc81da95ea378b0",
"classllzk_1_1boolean_1_1detail_1_1NotBoolOpGenericAdaptorBase.html#a437768cab81a8a1d2ececdea54fd60c7",
"classllzk_1_1cast_1_1IntToFeltOpGenericAdaptor.html#a9b24f6164b4b02c860fd7680b60801b7",
"classllzk_1_1component_1_1MemberReadOp.html#a20153649b20e0bd40c097d47a96b775a",
"classllzk_1_1component_1_1StructDefOp.html#a7946bb4e70453580f91581957d81bcf2",
"classllzk_1_1component_1_1detail_1_1StructDefOpGenericAdaptorBase.html#a68de71f891ebdf28600e2bcc043192d3",
"classllzk_1_1dataflow_1_1AbstractLatticeValue.html#a01a92eec1dcf10670a66a857b4e2bb65",
"classllzk_1_1felt_1_1AndFeltOp.html#abdd23bc2d62e962808e742d3d450e9c8",
"classllzk_1_1felt_1_1InvFeltOp.html#ab5953bed9cc70783aea24c2bc67dee1e",
"classllzk_1_1felt_1_1OrFeltOp.html#a1d7dae2c3e4e81c6f8fdd1b80db2aceb",
"classllzk_1_1felt_1_1ShrFeltOpAdaptor.html#a12eb8df52b8cf04c96150db672e2ec08",
"classllzk_1_1felt_1_1UnsignedIntDivFeltOp.html#a07f69a638aff4ca53c5da83e7791eeb8",
"classllzk_1_1felt_1_1detail_1_1DivFeltOpGenericAdaptorBase.html#a365564089314688c0e927903ddead040",
"classllzk_1_1felt_1_1detail_1_1UnsignedModFeltOpGenericAdaptorBase.html#a4a09a65a1ced60b04b40c76bc093fc5e",
"classllzk_1_1function_1_1FuncDefOp.html#a5e340f3ca8deb3c212244693b9c115be",
"classllzk_1_1function_1_1detail_1_1FuncDefOpGenericAdaptorBase.html#a3f4bae15ac01772ed805dc1bce6e3466",
"classllzk_1_1global_1_1GlobalRefOpInterface.html#a8f6c94ff7e16f5d2734f7a317a4f23b4",
"classllzk_1_1impl_1_1CallGraphPrinterPassBase.html#ad827302e9c44498b37b080f4e852293c",
"classllzk_1_1impl_1_1MemberWriteValidatorPassBase.html#a82e1ba212b4b0c373936badd5063c07e",
"classllzk_1_1impl_1_1UnusedDeclarationEliminationPassBase.html#a015bfb8ed56ce5e95bc1261b9bff0db3",
"classllzk_1_1pod_1_1NewPodOp.html#a75be9d07990fe650c06231c8816e4b1d",
"classllzk_1_1pod_1_1WritePodOp.html#a02a6f96139946421fd890599656abf26",
"classllzk_1_1polymorphic_1_1ApplyMapOp.html#a007ab817f8897ebed4e03464ef53c5e0",
"classllzk_1_1polymorphic_1_1TemplateExprOpAdaptor.html#aca3395b4c409a0436cce031d5f5cd7c8",
"classllzk_1_1polymorphic_1_1TypeVarType.html#a31a39812602b82e2e8c8f2b46e092891",
"classllzk_1_1polymorphic_1_1detail_1_1TemplateOpGenericAdaptorBase.html#a1b861784cc5b7a108c523386e169f354",
"classllzk_1_1ram_1_1LoadOp.html#aa4d743c1684ea8dcdb679b1d2b4837a6",
"classllzk_1_1smt_1_1ApplyFuncOpAdaptor.html#a2ebe656b0853b02ee49ae3e81a02250e",
"classllzk_1_1smt_1_1AssertOp.html",
"classllzk_1_1smt_1_1BVAddOpAdaptor.html",
"classllzk_1_1smt_1_1BVConstantOpAdaptor.html#a69f3b62038166f44a3a28944b54926a8",
"classllzk_1_1smt_1_1BVNotOp.html#abf43c68b12d23da5f9a3eee2d2049989",
"classllzk_1_1smt_1_1BVSModOpGenericAdaptor.html#a9c8eef894732e7b72e6d28245fade229",
"classllzk_1_1smt_1_1BVURemOpAdaptor.html",
"classllzk_1_1smt_1_1CheckOpGenericAdaptor.html#a8303ac4144cb43db66ef6d401d1d978e",
"classllzk_1_1smt_1_1EqOp.html#ac84f69c9b24ae4a4baf5a65124d58b51",
"classllzk_1_1smt_1_1ForallOp.html#a63088106aaab077e028f3adc840e7ddc",
"classllzk_1_1smt_1_1IntAbsOpAdaptor.html#a18abb4c2472d6dc6026c40a351e425be",
"classllzk_1_1smt_1_1IntConstantOpGenericAdaptor.html#a6c72e01b6e2a01d1db73a28c0526c7ea",
"classllzk_1_1smt_1_1IntNegOpAdaptor.html#a6b000b0211a0e6bc63acfe44ecab6b7c",
"classllzk_1_1smt_1_1OrOp.html#a09f8b63a5c1cd6d66d7067568fa44a8c",
"classllzk_1_1smt_1_1RepeatOpAdaptor.html#a82328545d02e012b783aaf3705e4ee44",
"classllzk_1_1smt_1_1SetInfoOp.html#a20aa33711baec3637f594a646aa3b50c",
"classllzk_1_1smt_1_1XOrOpAdaptor.html#a3bc7c98e318abb55bdfad89891f52f52",
"classllzk_1_1smt_1_1detail_1_1BVAndOpGenericAdaptorBase.html#a4f4f0ec9dd5a760ff5b3d350e13cf98f",
"classllzk_1_1smt_1_1detail_1_1BVXOrOpGenericAdaptorBase.html#af2038f944d59e7312fa0fd72f1cfab8b",
"classllzk_1_1smt_1_1detail_1_1Int2BVOpGenericAdaptorBase.html#af7c79548210469c967971fc8ba4b2647",
"classllzk_1_1smt_1_1detail_1_1PushOpGenericAdaptorBase.html#a4c1164071b3b7c42e4370796f74e9c34",
"classllzk_1_1string_1_1detail_1_1LitStringOpGenericAdaptorBase.html#ae68ac0a607a684d14d0771500afda088",
"classllzk_1_1verif_1_1ContractOp.html#afe81094ff4f9e13dc4a4caf9b1ad27e6",
"classllzk_1_1verif_1_1EnsureConstrainOpGenericAdaptor.html#a08a1462c0b3310e23c41e4f8e672ea6e",
"classllzk_1_1verif_1_1InvariantOp.html#abf2d3fe647600c9a1a808c97266e6894",
"classllzk_1_1verif_1_1RequireComputeOpGenericAdaptor.html#a53dd0a3826f1ee18c4520ee2bafe33d2",
"classllzk_1_1verif_1_1VerifAssertOpGenericAdaptor.html#aee72d2d58998d204ffbc7aa79ed696a3",
"classllzk_1_1verif_1_1detail_1_1EnsureComputeOpGenericAdaptorBase.html",
"classllzk_1_1verif_1_1detail_1_1StepOpGenericAdaptorBase.html#a18ef0c8002c8292d443f4be1a5dce570",
"dialects.html#operands-26",
"dir_ba0b87a1c71c7656624deb1072f340d5.html",
"llzk-c_2Support_8h.html#ac0162a315ebcfa7f54b93e0916f00c28",
"namespacellzk.html#a619bc971771be50cd90fe6219aa47aa2",
"namespacellzk.html#af64bf6226bffcb86733347072c5255e9",
"namespacellzk_1_1pod.html#ae82b1794e38c9bf3e11798e9abe9b3bc",
"pcl-backend.html#feltattr-1",
"smt-backend.html#smtbvextract-llzksmtextractop",
"structMulFeltOpBuildFuncHelper.html",
"structllvm_1_1GraphTraits_3_01const_01llzk_1_1CallGraphNode_01_5_01_4.html#aeecc588a5355acffa8906bdea1ad2fff",
"structllzk_1_1PredecessorPrinterPassOptions.html",
"structllzk_1_1component_1_1detail_1_1MemberDefOpGenericAdaptorBase_1_1Properties.html#a768ae66330b8eb5178aa9d29272d3ce4",
"structllzk_1_1function_1_1detail_1_1CallOpGenericAdaptorBase_1_1Properties.html#a4c3f8855239cf7b48a743b4ae39937bb",
"structllzk_1_1pod_1_1detail_1_1PodRefOpInterfaceInterfaceTraits.html",
"structllzk_1_1smt_1_1detail_1_1BitVectorTypeStorage.html#afb79b7b242c6b54626f545ba9f02112e",
"structllzk_1_1verif_1_1InvariantTargetOpInterface_1_1Trait.html",
"structllzk_1_1witgen_1_1WitgenOptions.html#a742654891f871f65682a4b99c5c0667d"
];

const SYNCONMSG = 'click to disable panel synchronization';
const SYNCOFFMSG = 'click to enable panel synchronization';
const LISTOFALLMEMBERS = 'List of all members';