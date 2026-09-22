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
"Array_2IR_2Ops_8capi_8test_8cpp_8inc.html#aec553ab569bfa77e48302cb4e923f7c9",
"Bool_2IR_2Ops_8capi_8h_8inc.html#a6ddde97f40c6cbe9015be66112eaa7f2",
"Cast_2IR_2Enums_8cpp_8inc_source.html",
"Constrain_8cpp.html",
"Felt_2IR_2Ops_8capi_8h_8inc.html#a17f1f52e5dc5a5f97fbd442f8445957c",
"Felt_2IR_2Ops_8capi_8test_8cpp_8inc_source.html",
"Function_2IR_2Ops_8capi_8h_8inc.html#a5719e32a2ad1b3f6a2cc48bb266013db",
"Global_2IR_2Ops_8capi_8test_8cpp_8inc.html#a1655a591e12af393c6aaa8db86d0728f",
"LLZKTransformationPasses_8capi_8h_8inc.html#a2361ad75b149e1bf6e2213c73dcd4da1",
"POD_2IR_2OpInterfaces_8td.html",
"Polymorphic_2IR_2Dialect_8cpp_8inc_source.html",
"RAM_2IR_2Ops_8capi_8h_8inc.html#a0fa4ef370e0f205497e30d908ac4a451",
"Struct_2IR_2Ops_8capi_8cpp_8inc.html#a4aebd0d79a696cfd0c9d926f4b764a8a",
"Struct_8h.html#a80b49b2f96cc12c23b34d149c6d0d07b",
"Verif_2IR_2Ops_8capi_8cpp_8inc.html#af32716b5179e8c9b48d7207a03ddeeb2",
"Verif_8h.html",
"classllzk_1_1CallGraphReachabilityAnalysis.html#a9ed72535ac0564b70cb8b00ccecf1377",
"classllzk_1_1InFlightDiagnosticWrapper.html#aca622bcf133ce35e08504853a6dd67a9",
"classllzk_1_1MemberOverwriteLattice.html#ab0d123e3f904fda299a09175a653f4eb",
"classllzk_1_1PredecessorLattice.html",
"classllzk_1_1SourceRefLatticeValue.html#af1fa4e0f52ece55441306e17db2b9aff",
"classllzk_1_1SymbolUseGraphNode.html#a1a525aee077424832a631e6acba2c156",
"classllzk_1_1array_1_1ArrayLengthOpGenericAdaptor.html#a0a8928b5354af7b76b8ec8c290d19714",
"classllzk_1_1array_1_1InsertArrayOp.html#a208500d4a5bad6b84b1ebfd66dd5dee9",
"classllzk_1_1array_1_1WriteArrayOpGenericAdaptor.html#a0b14f0b44cb64cb8128cf12ab111e4df",
"classllzk_1_1boolean_1_1AndBoolOpGenericAdaptor.html#ab66807c08f9ba266eeca6f1a666fa668",
"classllzk_1_1boolean_1_1ExistsOpGenericAdaptor.html#a36f88bb3fe8413e9e2238fade89bb9b5",
"classllzk_1_1boolean_1_1XorBoolOp.html#a9d25a6e21978ada9b70952736f860ec7",
"classllzk_1_1boolean_1_1detail_1_1XorBoolOpGenericAdaptorBase.html#add4f1370bbaec7407d5b87e65740d651",
"classllzk_1_1cast_1_1detail_1_1IntToFeltOpGenericAdaptorBase.html#a255054212b86515ce84aea61f6f16c57",
"classllzk_1_1component_1_1MemberReadOp.html#a8180f811990d1195b04a8270061ac2bd",
"classllzk_1_1component_1_1StructDefOp.html#aeb83bc589542b51bb815520b993177a6",
"classllzk_1_1component_1_1impl_1_1InlineStructsPassBase.html#aa7553909df24d9a2be3cb86459de49d3",
"classllzk_1_1dataflow_1_1AbstractLatticeValue.html#ad444c34dc1742e86f8522b350d726c79",
"classllzk_1_1felt_1_1DivFeltOp.html#a1d342a6a762c4dd3a21306fafb6fdb11",
"classllzk_1_1felt_1_1MulFeltOp.html#a114620d3bd07afa95baf97965eecd32a",
"classllzk_1_1felt_1_1OrFeltOpAdaptor.html#a9aee8c7b6d44b834bc9f14f9e75c06fc",
"classllzk_1_1felt_1_1SignedIntDivFeltOp.html#a5790a5758fba74f4cc25b683180ebef5",
"classllzk_1_1felt_1_1UnsignedIntDivFeltOpAdaptor.html#a71ec414396976bf3d6432fb0217616cc",
"classllzk_1_1felt_1_1detail_1_1FeltConstantOpGenericAdaptorBase.html#a7b786ae7087881874e99bc4faf7bd1d1",
"classllzk_1_1function_1_1CallOp.html#a173ffbd4ef665b319f869d83cc8d2c95",
"classllzk_1_1function_1_1FuncDefOp.html#ac70f62fa96d1567a287e528eef9ddaba",
"classllzk_1_1global_1_1GlobalDefOp.html",
"classllzk_1_1global_1_1GlobalWriteOp.html#ada986560fa5705041c6962b11979b3d3",
"classllzk_1_1impl_1_1ComputeConstrainToProductPassBase.html#a2b3ca2129370796c522a6ef980e1ee6c",
"classllzk_1_1impl_1_1PolyLoweringPassBase.html#aee9261f2f8b316b0dd03cfd1fdc10b19",
"classllzk_1_1impl_1_1WhileToForPassBase.html#a5e9e4390d7fad37647cfacd58ae3d8a9",
"classllzk_1_1pod_1_1NewPodOp.html#af5ca49480ce29f2d27001eaea1e6ee1c",
"classllzk_1_1pod_1_1WritePodOp.html#aab8ac542c6ed7e63cfb114465852912c",
"classllzk_1_1polymorphic_1_1ApplyMapOp.html#aac2a04205dfcd96fe35904d5c8b8ae3c",
"classllzk_1_1polymorphic_1_1TemplateOp.html#a56a04f5245cafd01dbd731ca44d6ae7a",
"classllzk_1_1polymorphic_1_1UnifiableCastOpAdaptor.html#a51588a4f6f17a57ff8f3b45903a95c45",
"classllzk_1_1polymorphic_1_1detail_1_1TemplateParamOpGenericAdaptorBase.html#a9e78c356dcf9083c0dbb1b1f77e1448d",
"classllzk_1_1ram_1_1RAMDialect.html#a16ba9f9fd76da17f6786169cf6c6079e",
"classllzk_1_1smt_1_1ArrayBroadcastOp.html#a8cc17b8e8f29468db33be4bb6afb4f52",
"classllzk_1_1smt_1_1AssertOpGenericAdaptor.html#a3a2708a6ef7cd0942114d56679ecf449",
"classllzk_1_1smt_1_1BVAndOp.html#a1b0d41743da3d174970824b669a7ba24",
"classllzk_1_1smt_1_1BVLShrOp.html#a65bde51b7d12eeaa1862ac8c7586c8b7",
"classllzk_1_1smt_1_1BVOrOp.html",
"classllzk_1_1smt_1_1BVSRemOpAdaptor.html#a2743a30eea30dc2e36b7ec56bbbfcd20",
"classllzk_1_1smt_1_1BVXOrOp.html#a5f50eb4235e60bcc7ca9e5fd4a4b17df",
"classllzk_1_1smt_1_1ConcatOpAdaptor.html",
"classllzk_1_1smt_1_1ExistsOp.html#a1afc09d2925e8f9fd2bf857c3649a4c2",
"classllzk_1_1smt_1_1ForallOp.html#aeb55bd648145fb928c4be688e5bb764f",
"classllzk_1_1smt_1_1IntAddOp.html#a9c5da1be2d62cfce96cf58fecccccbbc",
"classllzk_1_1smt_1_1IntDivOp.html#af24c45c2e2210a7db442be6205214ecd",
"classllzk_1_1smt_1_1IntSubOp.html#a652c5316a38e9d3372d92754654eda1c",
"classllzk_1_1smt_1_1OrOpGenericAdaptor.html",
"classllzk_1_1smt_1_1ResetOp.html#ae46080a5d921fe15e4a714b802b51703",
"classllzk_1_1smt_1_1SetInfoOpAdaptor.html#a3389405ad55d218ad0020710bf7b8bbb",
"classllzk_1_1smt_1_1YieldOp.html#aae2650740c3b496b4c3e60ee31610131",
"classllzk_1_1smt_1_1detail_1_1BVConstantOpGenericAdaptorBase.html#a728704687e03620642e8faa21e3029f5",
"classllzk_1_1smt_1_1detail_1_1CheckOpGenericAdaptorBase.html#ac4f988f45bf1639a89d9d00d2acd9086",
"classllzk_1_1smt_1_1detail_1_1IntCmpOpGenericAdaptorBase.html#a5691a6f623f1b09f599d16a0f002b36f",
"classllzk_1_1smt_1_1detail_1_1SetInfoOpGenericAdaptorBase.html#a0f12cbfb43ad45dbe68c95c79913ef8f",
"classllzk_1_1verif_1_1AssumeDetOpAdaptor.html#a3637deaf979abedd7d1a3dfc9dfe42e4",
"classllzk_1_1verif_1_1ContractTargetOpInterface.html#a20f747c19b938519f9d217aea8ad6555",
"classllzk_1_1verif_1_1IncludeOp.html#a47f91835878aefbe0015e7ae19515b5b",
"classllzk_1_1verif_1_1InvariantOpGenericAdaptor.html#ae61a69c4eee9ccb4b9adcd08ca15cf1f",
"classllzk_1_1verif_1_1RequireConstrainOpAdaptor.html#a7676870fafd7851b597864fe3cf9a142",
"classllzk_1_1verif_1_1VerifProveOpAdaptor.html#a42d30671eaf06c98f23f96d10603f568",
"classllzk_1_1verif_1_1detail_1_1ForbiddenInfluenceAnalyzer.html#ae636672f787c03ca268e1d4022328912",
"classllzk_1_1verif_1_1detail_1_1VerifProveOpGenericAdaptorBase.html",
"dialects.html#parameters",
"dir_fbb5839660df89f0c7872b7fe52b5d6c.html",
"maintanence.html#release-version",
"namespacellzk.html#a7d13a019e952052198f1b49196e67111",
"namespacellzk_1_1affineMapHelpers.html#ad6a727262e168eb23aa752c9acd2c805",
"namespacellzk_1_1polymorphic_1_1detail.html#aa2c080b1b59715494eafe735b1081aec",
"pcl-backend.html#pcltrue-pcltrueop",
"structAddFeltOpBuildFuncHelper.html",
"structOpImplementationGenerator.html#a80fd640b34c7f15fc09b8715bcad6ec1",
"structllzk_1_1CallGraphSCCsPrinterPassOptions.html",
"structllzk_1_1SplitFunctionNameInfo.html#afd749618e7dff455f220778cbc14ad16",
"structllzk_1_1component_1_1detail_1_1MemberReadOpGenericAdaptorBase_1_1Properties.html#ab08ac8a8e431a54451e4b8186995b9db",
"structllzk_1_1function_1_1detail_1_1FuncDefOpGenericAdaptorBase_1_1Properties.html#a9056da176d9b2452703227604c209252",
"structllzk_1_1polymorphic_1_1TemplateSymbolBindingOpInterface_1_1Trait.html",
"structllzk_1_1smt_1_1detail_1_1ExistsOpGenericAdaptorBase_1_1Properties.html#aae623dd0d5f9eb076c46b638f409bb26",
"structllzk_1_1verif_1_1detail_1_1ContractOpGenericAdaptorBase_1_1Properties.html#acb2b40dc0e42c75f781ce7ab74de47b7",
"tools.html#autotoc_md-llzk-enforce-no-overwrite"
];

const SYNCONMSG = 'click to disable panel synchronization';
const SYNCOFFMSG = 'click to enable panel synchronization';
const LISTOFALLMEMBERS = 'List of all members';