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
      [ "Building with PCL", "setup.html#building-with-pcl", null ],
      [ "Formatting hooks", "setup.html#formatting-hooks", null ],
      [ "Development Workflow", "setup.html#dev-workflow", null ],
      [ "Python Bindings", "setup.html#python-bindings", null ]
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
            [ "<span class=\"tt\">-llzk-flatten-constraints</span>", "tools.html#autotoc_md-llzk-flatten-constraints", null ],
            [ "<span class=\"tt\">-llzk-infer-tvar</span>", "tools.html#autotoc_md-llzk-infer-tvar", null ],
            [ "<span class=\"tt\">-llzk-monomorphize-templates</span>", "tools.html#autotoc_md-llzk-monomorphize-templates", [
              [ "Options", "tools.html#options-13", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-specialize-wildcard-arrays</span>", "tools.html#autotoc_md-llzk-specialize-wildcard-arrays", [
              [ "Options", "tools.html#options-14", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-expose-child-members</span>", "tools.html#autotoc_md-llzk-expose-child-members", [
              [ "Options", "tools.html#options-15", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-inline-structs</span>", "tools.html#autotoc_md-llzk-inline-structs", [
              [ "Options", "tools.html#options-16", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-to-pcl</span>", "tools.html#autotoc_md-llzk-to-pcl", [
              [ "Options", "tools.html#options-17", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-r1cs-lowering</span>", "tools.html#autotoc_md-llzk-r1cs-lowering", [
              [ "Options", "tools.html#options-18", null ]
            ] ]
          ] ],
          [ "Validation Passes", "tools.html#validation-passes", [
            [ "<span class=\"tt\">-llzk-validate-member-writes</span>", "tools.html#autotoc_md-llzk-validate-member-writes", null ]
          ] ],
          [ "Predefined Pipelines", "tools.html#predefined-pipelines", [
            [ "<span class=\"tt\">-llzk-remove-unnecessary-ops</span>", "tools.html#autotoc_md-llzk-remove-unnecessary-ops", null ],
            [ "<span class=\"tt\">-llzk-remove-unnecessary-ops-and-defs</span>", "tools.html#autotoc_md-llzk-remove-unnecessary-ops-and-defs", null ],
            [ "<span class=\"tt\">-llzk-product-program</span>", "tools.html#autotoc_md-llzk-product-program", null ],
            [ "<span class=\"tt\">-llzk-full-struct-inlining</span>", "tools.html#autotoc_md-llzk-full-struct-inlining", [
              [ "Options", "tools.html#options-19", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-full-inlining</span>", "tools.html#autotoc_md-llzk-full-inlining", null ],
            [ "<span class=\"tt\">-llzk-full-poly-lowering</span>", "tools.html#autotoc_md-llzk-full-poly-lowering", [
              [ "Options", "tools.html#options-20", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-full-r1cs-lowering</span>", "tools.html#autotoc_md-llzk-full-r1cs-lowering", null ],
            [ "<span class=\"tt\">-llzk-constraint-flattening-r1cs-lowering</span>", "tools.html#autotoc_md-llzk-constraint-flattening-r1cs-lowering", null ],
            [ "<span class=\"tt\">-llzk-constraint-flattening</span>", "tools.html#autotoc_md-llzk-constraint-flattening", null ]
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
"Felt_2IR_2Ops_8capi_8cpp_8inc.html#a4bd3fbcc2516b563416b90e9af92cb12",
"Felt_2IR_2Ops_8capi_8test_8cpp_8inc.html#a5a8fcbc7864319fada4f5bdf88482ac2",
"Function_2IR_2Ops_8capi_8cpp_8inc.html#a209ecb518695c2625cf58492f1cdaddb",
"Function_8cpp.html#abbeb9307274cd6224b5994f9013e93bb",
"InlineIncludesPass_8h.html#a27ba7d4664f92c7febd8d7a2b48db707",
"LLZK_2IR_2Ops_8capi_8cpp_8inc.html",
"POD_2IR_2Types_8capi_8cpp_8inc.html#a4d5e3445db42a90d5bcaddda537caae1",
"Polymorphic_2IR_2Ops_8capi_8test_8cpp_8inc.html#a5edbd073f351798df20c6d0004eef485",
"SourceRefLattice_8cpp_source.html",
"Struct_2IR_2Ops_8capi_8test_8cpp_8inc.html#a2ba30089275c26ab5719cb1fc6ef5e8a",
"Typing_8cpp.html#a8cd7f12b88f6f6d0de94c5f53006f2d2",
"Verif_2IR_2Ops_8capi_8h_8inc.html#a9653851f75a08b67e75bbe9387c34442",
"classLlzkDialectLinkTests.html",
"classllzk_1_1ExpressionValue.html#a47404227873395a42a1ca28be0016b60",
"classllzk_1_1Interval.html#adf403ad8a32b3ce99f950f2db365f6ea",
"classllzk_1_1ModuleBuilder.html#af98bcd56048df520b5f2be8c960d2d0b",
"classllzk_1_1SourceRef.html#a38f4c7d6b01daf6563cdfcaedbdc3256",
"classllzk_1_1StructAnalysis.html#ad69a19dcea3329ace5239afcf754903a",
"classllzk_1_1TemplateBuilder.html#a05602e5c63ebd6b871c01f67dd802e87",
"classllzk_1_1array_1_1ArrayRefOpInterface.html#aaeed5d4597c061932c6efb23cf77486e",
"classllzk_1_1array_1_1InsertArrayOp.html#aaa634b97c80a2b36f4ad1c3226745183",
"classllzk_1_1array_1_1WriteArrayOpGenericAdaptor.html#a4de6000ee4b6399bf53040bd0f23aa47",
"classllzk_1_1boolean_1_1AndBoolOp.html#a1557d70b5f4981b83a711ac79c97bce4",
"classllzk_1_1boolean_1_1CmpOpGenericAdaptor.html#a989b3751b705ce12b1c308e10fa8c7d7",
"classllzk_1_1boolean_1_1OrBoolOp.html#a96e0c17386d40d491a7de508791852eb",
"classllzk_1_1boolean_1_1detail_1_1CmpOpGenericAdaptorBase.html#a13b5ce6db7a1cc4ebc22fb86d7b26e58",
"classllzk_1_1cast_1_1FeltToIndexOpAdaptor.html",
"classllzk_1_1component_1_1MemberDefOp.html#a19bd59afaf9d16006c7c08b62cfb2ae4",
"classllzk_1_1component_1_1MemberRefOpInterface.html#ac3ca70d653c775c817f7c92405ccf21c",
"classllzk_1_1component_1_1detail_1_1CreateStructOpGenericAdaptorBase.html#af7530d2214c366ebde218eaea979b724",
"classllzk_1_1constrain_1_1EmitContainmentOp.html#ac70a5a3a1bf815a6bffbef08e5954491",
"classllzk_1_1dataflow_1_1SparseForwardDataFlowAnalysis.html#ac073da339fa3258e4e13a0efc229e331",
"classllzk_1_1felt_1_1DivFeltOp.html#a38823a841fdb6dac49ec38d1ddd5692a",
"classllzk_1_1felt_1_1MulFeltOp.html",
"classllzk_1_1felt_1_1OrFeltOp.html#abd03ab70c04bde33d4f54c5994161192",
"classllzk_1_1felt_1_1ShrFeltOpAdaptor.html#a3b8835f05469a75f37f0dc91ceeb4be9",
"classllzk_1_1felt_1_1SubFeltOpGenericAdaptor.html#a3091bde245cf978ab661e1cb5cc76102",
"classllzk_1_1felt_1_1XorFeltOpGenericAdaptor.html#ad23d3b6fec6ea2aa6ae1cc52a38428a7",
"classllzk_1_1felt_1_1detail_1_1ShlFeltOpGenericAdaptorBase.html#a7672d40d226d3284bc72d4f75799a291",
"classllzk_1_1function_1_1CallOp.html#a986dfde6236945588647617af08b4f36",
"classllzk_1_1function_1_1FuncDefOpAdaptor.html#a4aa35d9cdbddc2c25bd84935cf6a9c7d",
"classllzk_1_1global_1_1GlobalDefOp.html#a5651f00ca916ecdd5b867e7b57b82108",
"classllzk_1_1global_1_1GlobalWriteOpGenericAdaptor.html#a297c98f9c782662b49694f1b94461c24",
"classllzk_1_1impl_1_1ConstraintDependencyGraphPrinterPassBase.html",
"classllzk_1_1impl_1_1PredecessorPrinterPassBase.html#a71709edf96ac076e26d1b833adf5fbe6",
"classllzk_1_1include_1_1IncludeOp.html#a0f197cbb31e40fd9f1e0a0f8bd3f7e92",
"classllzk_1_1pod_1_1NewPodOpGenericAdaptor.html#a39f0808cb7671770bfb6184dcfd01fd1",
"classllzk_1_1pod_1_1WritePodOp.html#af8eb423c135bf7f80c51cbdaa84ecdbc",
"classllzk_1_1polymorphic_1_1ApplyMapOpAdaptor.html#a3211ce4b5368b5a0bcd516f7fd5392a6",
"classllzk_1_1polymorphic_1_1TemplateOp.html#a717beda1ad6483759f16fa67a71d1a23",
"classllzk_1_1polymorphic_1_1UnifiableCastOpGenericAdaptor.html#a4f9b350763cdadfa6d92f53231e5fb1d",
"classllzk_1_1polymorphic_1_1detail_1_1TemplateExprOpGenericAdaptorBase.html#a1dfe07bc09564b2bd55a9cc4895bf031",
"classllzk_1_1polymorphic_1_1impl_1_1EmptyTemplateRemovalPassBase.html#ae44288cd0abeaa4820e7893c756138ff",
"classllzk_1_1ram_1_1StoreOp.html#af4919aeeb790d393e382a3051e3dd497",
"classllzk_1_1string_1_1LitStringOpAdaptor.html",
"classllzk_1_1verif_1_1ContractOp.html#a5c5e910fb59a415da7c52af67734440b",
"classllzk_1_1verif_1_1EnsureComputeOp.html#a7a76ab604bb91febe2983f232a2884ef",
"classllzk_1_1verif_1_1IncludeOpGenericAdaptor.html#a497ad7314cfb6c4c9c48ac92f8ee62d7",
"classllzk_1_1verif_1_1PostconditionOpInterface.html#a00089b950e953078931eefdab35a8708",
"classllzk_1_1verif_1_1StepOp.html#ae5a73b2fbcbff63639e3742b504ab7da",
"classllzk_1_1verif_1_1VerifSMTProveOp.html#a149b827ca8daeb169f76fafc3cdd958d",
"classllzk_1_1verif_1_1detail_1_1ForbiddenInfluenceAnalyzer.html#a2029100a16656a19e97c0a11e3957526",
"classllzk_1_1verif_1_1detail_1_1StepOpGenericAdaptorBase.html#a18ef0c8002c8292d443f4be1a5dce570",
"dialects.html#fieldspecattr",
"dir_8824517f61f65466f9470639a7e8c03b.html",
"llzk-c_2Builder_8h.html#aa1480bc854ed5fd0ae26e0714f914ab8",
"namespacellzk.html#a449db750fee6d61e2ec3aa36a969d483",
"namespacellzk.html#acdac41547ae59c95df683ecca710db95",
"namespacellzk_1_1debug.html#a0352c114ac94203cb61cc0a0aa5d5ed3",
"namespacellzk_1_1polymorphic_1_1impl.html#ae39de184abcb3634585c00b25e46e3f4",
"r1cs-backend.html#operations-13",
"structNotBoolOpBuildFuncHelper.html#a4d343e63f64cded4c9e60a6685987e29",
"structllvm_1_1GraphTraits_3_01const_01llzk_1_1SymbolUseGraph_01_5_01_4.html#a354e59a3f4b9005c38a46226e6b85403",
"structllzk_1_1RefValueCapture.html",
"structllzk_1_1component_1_1detail_1_1MemberDefOpGenericAdaptorBase_1_1Properties.html#a91b66b3b7cb396be749634c82bfffdfa",
"structllzk_1_1function_1_1detail_1_1CallOpGenericAdaptorBase_1_1Properties.html#a3b37d50339436ee9559d62bb42535531",
"structllzk_1_1pod_1_1detail_1_1PodAccessOpInterfaceInterfaceTraits.html",
"structllzk_1_1polymorphic_1_1detail_1_1TemplateSymbolBindingOpInterfaceInterfaceTraits_1_1Concept.html#a004d9b9c8d486f88308d247bc5f7b3f4",
"structllzk_1_1verif_1_1detail_1_1InvariantOpGenericAdaptorBase_1_1Properties.html#a33e13d2ebd96e4377db4ab4c8c1f9568",
"zklean-backend.html#zkexprliteral-llzkzkexprliteralop"
];

const SYNCONMSG = 'click to disable panel synchronization';
const SYNCOFFMSG = 'click to enable panel synchronization';
const LISTOFALLMEMBERS = 'List of all members';