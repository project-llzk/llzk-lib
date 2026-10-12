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
            [ "<span class=\"tt\">-llzk-monomorphize-templates</span>", "tools.html#autotoc_md-llzk-monomorphize-templates", [
              [ "Options", "tools.html#options-13", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-specialize-wildcard-arrays</span>", "tools.html#autotoc_md-llzk-specialize-wildcard-arrays", [
              [ "Options", "tools.html#options-14", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-inline-structs</span>", "tools.html#autotoc_md-llzk-inline-structs", [
              [ "Options", "tools.html#options-15", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-to-pcl</span>", "tools.html#autotoc_md-llzk-to-pcl", [
              [ "Options", "tools.html#options-16", null ]
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
              [ "Options", "tools.html#options-17", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-full-inlining</span>", "tools.html#autotoc_md-llzk-full-inlining", null ],
            [ "<span class=\"tt\">-llzk-full-poly-lowering</span>", "tools.html#autotoc_md-llzk-full-poly-lowering", [
              [ "Options", "tools.html#options-18", null ]
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
      [ "Translation Guidelines", "syntax.html#translation-guidelines", [
        [ "Mathematical integers and fixed-width encodings", "syntax.html#mathematical-integers-and-fixed-width-encodings", null ]
      ] ]
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
"Array_2IR_2Ops_8capi_8test_8cpp_8inc.html#a2ba71cb9bb2b9e07ec3e315305720747",
"Bool_2IR_2Ops_8capi_8cpp_8inc.html#a93d1b6e61e7eeaa68a4a55f080ed0b0f",
"CallLikeOpInterfaces_8cpp_8inc.html",
"Constrain_2IR_2OpInterfaces_8cpp_8inc_source.html",
"Felt_2IR_2Ops_8capi_8cpp_8inc.html#a7f2bade7b93358051b4721b495d67ec0",
"Felt_2IR_2Ops_8capi_8test_8cpp_8inc.html#a7e7013186b76f2b27e6a44a3ab9091ac",
"Function_2IR_2Ops_8capi_8cpp_8inc.html#aa37d5c1b540a3c5eaeffacfe9705a867",
"Global_2IR_2Dialect_8cpp_source.html",
"JSON_8cpp.html",
"LightweightSignalEquivalenceAnalysis_8h_source.html",
"POD_2Transforms_2TransformationPasses_8h.html#a50fd6f4ec277edd1b62f2efe4a35eede",
"Polymorphic_2IR_2Ops_8td.html",
"String_2IR_2Ops_8capi_8test_8cpp_8inc.html#a170f2a9371aab49b42246316c6763e93",
"Struct_2IR_2Types_8capi_8cpp_8inc.html#a5d2654d8bd3c3a9a7c7bcf244a5fd3f2",
"Verif_2IR_2Ops_8capi_8cpp_8inc.html",
"Verif_2IR_2Ops_8capi_8test_8cpp_8inc.html#a40a907c8df152643d92a64923b37aa8f",
"classllzk_1_1BinaryBuffer.html#a02ac44f59f084f391a40d29a16ed30c1",
"classllzk_1_1Field.html#a28f0e6e597b3f6023040ef703971f44e",
"classllzk_1_1IntervalAnalysisLatticeValue.html#a598ca3498a26c58a985aa8db71b90772",
"classllzk_1_1ModuleLikeBuilder.html#a7b78418a92df92dc2709465f44fcfe9b",
"classllzk_1_1SourceRef.html#ae77745b0ef066aded6e384055f75c8f1",
"classllzk_1_1SymbolDefTreeNode.html#a8aab81b6e404750be7d227ad0ffb857a",
"classllzk_1_1Within.html",
"classllzk_1_1array_1_1CreateArrayOp.html#aaf6380eeff520cc428fdee7458641c22",
"classllzk_1_1array_1_1ReadArrayOp.html#a2a66a4248a8d2c331aeeeb31b403b435",
"classllzk_1_1array_1_1detail_1_1CreateArrayOpGenericAdaptorBase.html#a39dffdb331f1e3c1947226d4b2528938",
"classllzk_1_1boolean_1_1AssertOp.html#a0640284a0ef860923dee032d3fd1536d",
"classllzk_1_1boolean_1_1ExistsOpGenericAdaptor.html#abcb423d312c9f3610b17d81b425dc4ec",
"classllzk_1_1boolean_1_1XorBoolOp.html#a76404d24dd81b74f9c7ff5c108df461a",
"classllzk_1_1boolean_1_1detail_1_1NotBoolOpGenericAdaptorBase.html#a477e6733c5f59f529f057c18c4036d25",
"classllzk_1_1cast_1_1IntToFeltOpAdaptor.html",
"classllzk_1_1component_1_1MemberDefOpGenericAdaptor.html#a20a28003d1da2a100ff0605e6311d2ea",
"classllzk_1_1component_1_1MemberWriteOpGenericAdaptor.html#aabc569d9c908b7560e6dd21a2b30187c",
"classllzk_1_1component_1_1detail_1_1MemberRefOpInterfaceInterfaceTraits_1_1Model.html#abe7df1ff75bb5e5bd40557700bee3e28",
"classllzk_1_1constrain_1_1detail_1_1ConstraintOpInterfaceInterfaceTraits_1_1ExternalModel.html#a4c7862d5e9a6095305e1abd98c55a9b2",
"classllzk_1_1felt_1_1AddFeltOp.html#abe15104ce50be9e0a16a41eca3ebc63e",
"classllzk_1_1felt_1_1FeltConstantOpAdaptor.html#a157207e6e1a6fa6c113fd32ff1a6c032",
"classllzk_1_1felt_1_1NegFeltOpAdaptor.html#a71264860697f2f4609e805d637f5fbde",
"classllzk_1_1felt_1_1PowFeltOpGenericAdaptor.html#ad3400fda4c1ad47a22de6482bfdde019",
"classllzk_1_1felt_1_1SignedModFeltOp.html#a4a72174d69d5e8f798e510ca40284061",
"classllzk_1_1felt_1_1UnsignedModFeltOp.html#a8e811889711dc341bd86fa1929ee01e8",
"classllzk_1_1felt_1_1detail_1_1InvFeltOpGenericAdaptorBase.html#acd6ac643f800b35955ae16bd169eccab",
"classllzk_1_1felt_1_1detail_1_1UnsignedIntDivFeltOpGenericAdaptorBase.html#ab9df940942ab97d71edc0166b085434f",
"classllzk_1_1function_1_1FuncDefOp.html#a2c6888cd468ee9450e0a26d4b0d6bee8",
"classllzk_1_1function_1_1detail_1_1CallOpGenericAdaptorBase.html#a0b49c8680c040b056d1d4bc2d94d518f",
"classllzk_1_1global_1_1GlobalReadOp.html#aaaad1cadaddf98320252e1dac12b7e0a",
"classllzk_1_1global_1_1impl_1_1ConstGlobalPropagationPassBase.html",
"classllzk_1_1impl_1_1IntervalAnalysisPrinterPassBase.html",
"classllzk_1_1impl_1_1SymbolDefTreePrinterPassBase.html#aae9033fb5d9b41c42e11d47f6affddfd",
"classllzk_1_1include_1_1impl_1_1InlineIncludesPassBase.html#a8aeffa2aad2dfd47f7de8a3630964212",
"classllzk_1_1pod_1_1ReadPodOp.html#ab8514d77ea4fc6b6c8b9866085279131",
"classllzk_1_1pod_1_1detail_1_1ReadPodOpGenericAdaptorBase.html#aae8334f76dde12f0bd2cee77af3b92f0",
"classllzk_1_1polymorphic_1_1TemplateExprOp.html#a06ffd6ee66b8e7770193527a45822d27",
"classllzk_1_1polymorphic_1_1TemplateParamOpGenericAdaptor.html#a25dcf9969af363bca1c77e50f5acc221",
"classllzk_1_1polymorphic_1_1detail_1_1ConstReadOpGenericAdaptorBase.html#a52a9d03f6c33cd16f07ac4355c216e02",
"classllzk_1_1polymorphic_1_1impl_1_1FlatteningPassBase.html#a60514483ca28b65d98edc0c36b244039",
"classllzk_1_1ram_1_1StoreOpGenericAdaptor.html#a3a972152bf2ee267b43623c3364263c6",
"classllzk_1_1string_1_1LitStringOpGenericAdaptor.html#a60df45094447d9df280a6673a2c75606",
"classllzk_1_1verif_1_1ContractOp.html#a851da8ba70f388557d6b08c6a09e6079",
"classllzk_1_1verif_1_1EnsureComputeOpAdaptor.html#aee82cac6dbec958dd4b7f6b023ed8c4f",
"classllzk_1_1verif_1_1IncreasesOp.html#a660b716f3c3ac4b03c12ac9d49ff21da",
"classllzk_1_1verif_1_1ProveDetOp.html#a3286a3f68e566928d0406688bddab1e8",
"classllzk_1_1verif_1_1StepOpGenericAdaptor.html#a1e6ab42b868d0200328fecdb23db1033",
"classllzk_1_1verif_1_1VerifSMTProveOpAdaptor.html",
"classllzk_1_1verif_1_1detail_1_1IncludeOpGenericAdaptorBase.html#a6b03f2aaa99ae75f60e0a5cad3f68856",
"classllzk_1_1verif_1_1detail_1_1StepYieldOpGenericAdaptorBase.html#a229a9679e07d9440680a18227262fbfa",
"dialects.html#operands-17",
"dir_adfe3cc28faefdcc2b149bf5452f8464.html",
"llzk-c_2DialectRegistration_8h.html#ae33e7733866550d325ac77bff79abff0",
"namespacellzk.html#a573de69c46da2c30863428ac17deeae8",
"namespacellzk.html#ae25f83930559cf0ea3e386db68f54598",
"namespacellzk_1_1impl.html#a43b34038f2aae69d95362c5bbcd9a535",
"namespacemembers_func_d.html",
"structEmitEqualityOpBuildFuncHelper.html#ac6e510b8157fde62c5346011dd31d186",
"structUnifiableCastOpBuildFuncHelper.html",
"structllzk_1_1IntervalAnalysisContext.html",
"structllzk_1_1array_1_1detail_1_1ArrayTypeStorage.html#a83972eee46c98be1064ccc72f9fcf979",
"structllzk_1_1constrain_1_1detail_1_1ConstraintOpInterfaceInterfaceTraits.html",
"structllzk_1_1global_1_1detail_1_1GlobalDefOpGenericAdaptorBase_1_1Properties.html#aa1c686eeeec7d994c4b9885e1f6217e8",
"structllzk_1_1polymorphic_1_1detail_1_1InstantiationLayout.html#a65456921b57398e3844ac01ca8a280f2",
"structllzk_1_1verif_1_1detail_1_1IncludeOpGenericAdaptorBase_1_1Properties.html#a0d70edda904677f12bfbce16d5bfe20d",
"tools.html#autotoc_md-llzk-product-program"
];

const SYNCONMSG = 'click to disable panel synchronization';
const SYNCOFFMSG = 'click to enable panel synchronization';
const LISTOFALLMEMBERS = 'List of all members';