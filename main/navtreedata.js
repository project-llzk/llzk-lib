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
"Array_2IR_2Ops_8capi_8test_8cpp_8inc.html#a2ba71cb9bb2b9e07ec3e315305720747",
"Bool_2IR_2Ops_8capi_8cpp_8inc.html#a93d1b6e61e7eeaa68a4a55f080ed0b0f",
"CallLikeOpInterfaces_8cpp_8inc.html",
"Constrain_2IR_2OpInterfaces_8cpp_8inc_source.html",
"Felt_2IR_2Ops_8capi_8cpp_8inc.html#a7f2bade7b93358051b4721b495d67ec0",
"Felt_2IR_2Ops_8capi_8test_8cpp_8inc.html#a7b8bbefafea3fcb29011df01967d9aea",
"Function_2IR_2Ops_8capi_8cpp_8inc.html#a887f98c869539b2107423eadc888c20d",
"Global_2IR_2Dialect_8capi_8test_8cpp_8inc.html",
"IntervalAnalysis_8cpp_source.html",
"LightweightSignalEquivalenceAnalysis_8h.html",
"POD_2Transforms_2TransformationPasses_8h.html#a197de990137bba19a1e0623808f75e5d",
"Polymorphic_2IR_2Ops_8h_source.html",
"String_2IR_2Ops_8capi_8test_8cpp_8inc_source.html",
"Struct_2IR_2Types_8capi_8test_8cpp_8inc.html",
"Verif_2IR_2Ops_8capi_8cpp_8inc.html#a24bfedfb62beff85ad5a2b6955dc714b",
"Verif_2IR_2Ops_8capi_8test_8cpp_8inc.html#a60ee721f1ffc4bc9109c5201a4e042e2",
"classllzk_1_1BaseBuilder.html#abf3ccf80941e19e5bee5c8f0a78906ee",
"classllzk_1_1Field.html#a371d71b386b8387a592a9967311a91ab",
"classllzk_1_1IntervalAnalysisLatticeValue.html#a73fa4b2fa35656374233180fa4d9ddea",
"classllzk_1_1ModuleLikeBuilder.html#a88eaeb924e1447b6a8cd3629ff89b8b9",
"classllzk_1_1SourceRefAnalysis.html#a95f9ec2b825b270c0fa65459cbd99a06",
"classllzk_1_1SymbolLookupResult.html#a28c49a5ae5d23836b51db3e34716aa0a",
"classllzk_1_1Within.html#aae4b1d794f8dd2cc972eea1c991af90f",
"classllzk_1_1array_1_1CreateArrayOp.html#ad79d513c221868409d9a7605ebff327c",
"classllzk_1_1array_1_1ReadArrayOp.html#a6691898171ad5133de6a34e04ebb674b",
"classllzk_1_1array_1_1detail_1_1CreateArrayOpGenericAdaptorBase.html#ac775ef4393e62a6d3096300616c30795",
"classllzk_1_1boolean_1_1AssertOp.html#a513e1b637401f380b6bd0d20c80bc13f",
"classllzk_1_1boolean_1_1ForAllOp.html#a29b6a1fbaa475096cb9a47114e80d519",
"classllzk_1_1boolean_1_1XorBoolOp.html#ab78fc72d0d18d9c576e426ec509fcfc3",
"classllzk_1_1boolean_1_1detail_1_1NotBoolOpGenericAdaptorBase.html#ad8aa49d193d38322be7c9c91671ac251",
"classllzk_1_1cast_1_1IntToFeltOpAdaptor.html#aba947248a478ef981d206e5266643613",
"classllzk_1_1component_1_1MemberDefOpGenericAdaptor.html#ae782c55c962578519a5bd80ad1a0e4fc",
"classllzk_1_1component_1_1StructDefOp.html#a19d639929d4054a60f9a469e23cfac74",
"classllzk_1_1component_1_1detail_1_1MemberWriteOpGenericAdaptorBase.html#a2f8ff3e48c914d869878392713e23d4d",
"classllzk_1_1constrain_1_1detail_1_1EmitContainmentOpGenericAdaptorBase.html",
"classllzk_1_1felt_1_1AddFeltOpAdaptor.html#acbd120a4025e2da1f95a88dd98c9b045",
"classllzk_1_1felt_1_1FeltConstantOpAdaptor.html#afadc81e954bcd0e6f83ddce87e7df35e",
"classllzk_1_1felt_1_1NegFeltOpGenericAdaptor.html#a008a7f5606ce5bbd3280660f1c2afcfb",
"classllzk_1_1felt_1_1ShlFeltOp.html#a0d4b23d0a26fa4f742d8e0345e52240f",
"classllzk_1_1felt_1_1SignedModFeltOp.html#a8ef6d67f0a086370740309450b07857d",
"classllzk_1_1felt_1_1UnsignedModFeltOp.html#adf7137a685256431f1f3e0026b57e1a4",
"classllzk_1_1felt_1_1detail_1_1MulFeltOpGenericAdaptorBase.html#a2a722ab6f526bb413967485cc89979fc",
"classllzk_1_1felt_1_1detail_1_1UnsignedModFeltOpGenericAdaptorBase.html#a7491909bfcaedb4ff001b500a6bb7bac",
"classllzk_1_1function_1_1FuncDefOp.html#a3b824744d5a4d4a7e668e3e360a03b84",
"classllzk_1_1function_1_1detail_1_1CallOpGenericAdaptorBase.html#a49c05a012f2f09cbade19d4f86570086",
"classllzk_1_1global_1_1GlobalReadOp.html#ad520e6c5a5f41e492aebd6d945995d82",
"classllzk_1_1global_1_1impl_1_1ConstGlobalPropagationPassBase.html#a885b05320b09d0e09e10abd2b7edc928",
"classllzk_1_1impl_1_1IntervalAnalysisPrinterPassBase.html#a670413664f7e1b6386d8ff16540e0b04",
"classllzk_1_1impl_1_1SymbolDefTreePrinterPassBase.html#ae0942ecef092d91d92d3b7fa4f40ea3d",
"classllzk_1_1include_1_1impl_1_1InlineIncludesPassBase.html#add78893f9c06bd8795418f4dd9baf932",
"classllzk_1_1pod_1_1ReadPodOp.html#ae224b101afac01d40a50deff5de5d079",
"classllzk_1_1pod_1_1detail_1_1WritePodOpGenericAdaptorBase.html#a35d5f272a1e318f2d4d53fab3a98ceef",
"classllzk_1_1polymorphic_1_1TemplateExprOp.html#a38d86f45d4c6426e7dcc6999e8bf0f83",
"classllzk_1_1polymorphic_1_1TemplateSymbolBindingOpInterface.html",
"classllzk_1_1polymorphic_1_1detail_1_1ConstReadOpGenericAdaptorBase.html#ae7ffb8532374ecd3224a960f6d14fed2",
"classllzk_1_1polymorphic_1_1impl_1_1TypeVarInferencePassBase.html#ab69bb2176c1aa29549956a611b8f620b",
"classllzk_1_1smt__info_1_1SMTInfoDialect.html#ae1d74d7b37ca93cd19e544d06aece777",
"classllzk_1_1verif_1_1AssumeDetOp.html#af3435ed5defcef7459b4d4762fbc7bad",
"classllzk_1_1verif_1_1ContractOpAdaptor.html#aa48c207eed9491bd177dd000f09f0e5a",
"classllzk_1_1verif_1_1EnsureConstrainOpGenericAdaptor.html",
"classllzk_1_1verif_1_1InvariantOp.html#a453c76024d50a079e71f262e4f558229",
"classllzk_1_1verif_1_1RequireComputeOp.html#a7f90c0f1d7c92548d861b749d0846717",
"classllzk_1_1verif_1_1StepYieldOpGenericAdaptor.html#ae731d2e2a8405c839db3db9372d70ef9",
"classllzk_1_1verif_1_1detail_1_1ConditionOpInterfaceInterfaceTraits_1_1Model.html#a579eaff055e9604ab89407e64b3a4b42",
"classllzk_1_1verif_1_1detail_1_1InvariantOpGenericAdaptorBase.html#ac781bccb06111d1ddd479e39163cfb9b",
"classllzk_1_1verif_1_1detail_1_1VerifSMTProveOpGenericAdaptorBase.html#a61fe076f4f0ce22fc0a06e477ac6cade",
"dialects.html#polytemplate-llzkpolymorphictemplateop",
"functions_func_i.html",
"namespacellvm.html",
"namespacellzk.html#a7f9e5ec8059996c90a89f6de02767394",
"namespacellzk_1_1affineMapHelpers.html#ac87c7276a74f7ca378ad932599e54113",
"namespacellzk_1_1polymorphic.html#afec084874e5df37a9774039696031b8f",
"pcl-backend.html#results-65",
"structMlirOpBuilderInsertPoint.html#a670dfda6e5fa742b0c8b38414f79512f",
"structllvm_1_1GraphTraits_3_01const_01llzk_1_1SymbolDefTreeNode_01_5_01_4.html#a4c325bfbfdbf7c0e4c0941f7bbc8ace2",
"structllzk_1_1RefValueCapture.html",
"structllzk_1_1component_1_1detail_1_1MemberDefOpGenericAdaptorBase_1_1Properties.html#aa9a5d0f573b9aa7563acc41b1a8efb3e",
"structllzk_1_1function_1_1detail_1_1CallOpGenericAdaptorBase_1_1Properties.html#a57372879ff1773eb05e71a8d45784c81",
"structllzk_1_1pod_1_1detail_1_1PodAccessOpInterfaceInterfaceTraits_1_1Concept.html#a7a21c46491d88f5992f08ac64157b6c0",
"structllzk_1_1verif_1_1detail_1_1CallableSummaryKey.html#a362bcb530bfdb0d2b3c76891f3bb818e",
"structllzk_1_1witgen_1_1PodValue.html#af7443fe54b540dd7a50d4c50bee20c85"
];

const SYNCONMSG = 'click to disable panel synchronization';
const SYNCOFFMSG = 'click to enable panel synchronization';
const LISTOFALLMEMBERS = 'List of all members';