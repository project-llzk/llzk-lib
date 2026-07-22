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
          [ "General Transformation Passes", "tools.html#general-transformation-passes", [
            [ "<span class=\"tt\">-llzk-compute-constrain-to-product</span>", "tools.html#autotoc_md-llzk-compute-constrain-to-product", [
              [ "Options", "tools.html#options-7", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-duplicate-op-elim</span>", "tools.html#autotoc_md-llzk-duplicate-op-elim", null ],
            [ "<span class=\"tt\">-llzk-duplicate-read-write-elim</span>", "tools.html#autotoc_md-llzk-duplicate-read-write-elim", null ],
            [ "<span class=\"tt\">-llzk-enforce-no-overwrite</span>", "tools.html#autotoc_md-llzk-enforce-no-overwrite", null ],
            [ "<span class=\"tt\">-llzk-fuse-product-loops</span>", "tools.html#autotoc_md-llzk-fuse-product-loops", null ],
            [ "<span class=\"tt\">-llzk-poly-lowering-pass</span>", "tools.html#autotoc_md-llzk-poly-lowering-pass", [
              [ "Options", "tools.html#options-8", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-remove-unused-discardable-allocations</span>", "tools.html#autotoc_md-llzk-remove-unused-discardable-allocations", [
              [ "Options", "tools.html#options-9", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-unused-declaration-elim</span>", "tools.html#autotoc_md-llzk-unused-declaration-elim", [
              [ "Options", "tools.html#options-10", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-while-to-for</span>", "tools.html#autotoc_md-llzk-while-to-for", null ]
          ] ],
          [ "'array' Dialect Transformation Passes", "tools.html#array-dialect-transformation-passes", [
            [ "<span class=\"tt\">-llzk-array-to-scalar</span>", "tools.html#autotoc_md-llzk-array-to-scalar", null ]
          ] ],
          [ "'polymorphic' Dialect Transformation Passes", "tools.html#polymorphic-dialect-transformation-passes", [
            [ "<span class=\"tt\">-llzk-drop-empty-templates</span>", "tools.html#autotoc_md-llzk-drop-empty-templates", null ],
            [ "<span class=\"tt\">-llzk-flatten</span>", "tools.html#autotoc_md-llzk-flatten", [
              [ "Options", "tools.html#options-11", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-specialize-wildcard-arrays</span>", "tools.html#autotoc_md-llzk-specialize-wildcard-arrays", [
              [ "Options", "tools.html#options-12", null ]
            ] ]
          ] ],
          [ "Validation Passes", "tools.html#validation-passes", [
            [ "<span class=\"tt\">-llzk-validate-member-writes</span>", "tools.html#autotoc_md-llzk-validate-member-writes", null ]
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
    [ "Backend Dialects", "backend-dialects.html", [
      [ "'r1cs' Dialect", "backend-dialects.html#r1cs-dialect", [
        [ "Operations", "backend-dialects.html#operations-13", [
          [ "<span class=\"tt\">r1cs.add</span> (r1cs::AddOp)", "backend-dialects.html#r1csadd-r1csaddop", [
            [ "Operands:", "backend-dialects.html#operands-47", null ],
            [ "Results:", "backend-dialects.html#results-42", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.circuit</span> (r1cs::CircuitDefOp)", "backend-dialects.html#r1cscircuit-r1cscircuitdefop", [
            [ "Attributes:", "backend-dialects.html#attributes-30", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.const</span> (r1cs::ConstOp)", "backend-dialects.html#r1csconst-r1csconstop", [
            [ "Attributes:", "backend-dialects.html#attributes-31", null ],
            [ "Results:", "backend-dialects.html#results-43", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.constrain</span> (r1cs::ConstrainOp)", "backend-dialects.html#r1csconstrain-r1csconstrainop", [
            [ "Operands:", "backend-dialects.html#operands-48", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.def</span> (r1cs::SignalDefOp)", "backend-dialects.html#r1csdef-r1cssignaldefop", [
            [ "Attributes:", "backend-dialects.html#attributes-32", null ],
            [ "Results:", "backend-dialects.html#results-44", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.mul_const</span> (r1cs::MulConstOp)", "backend-dialects.html#r1csmul_const-r1csmulconstop", [
            [ "Attributes:", "backend-dialects.html#attributes-33", null ],
            [ "Operands:", "backend-dialects.html#operands-49", null ],
            [ "Results:", "backend-dialects.html#results-45", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.neg</span> (r1cs::NegOp)", "backend-dialects.html#r1csneg-r1csnegop", [
            [ "Operands:", "backend-dialects.html#operands-50", null ],
            [ "Results:", "backend-dialects.html#results-46", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.to_linear</span> (r1cs::ToLinearOp)", "backend-dialects.html#r1csto_linear-r1cstolinearop", [
            [ "Operands:", "backend-dialects.html#operands-51", null ],
            [ "Results:", "backend-dialects.html#results-47", null ]
          ] ]
        ] ],
        [ "Attributes", "backend-dialects.html#attributes-34", [
          [ "FeltAttr", "backend-dialects.html#feltattr", [
            [ "Parameters:", "backend-dialects.html#parameters-11", null ]
          ] ],
          [ "PublicAttr", "backend-dialects.html#publicattr-1", null ]
        ] ],
        [ "Types", "backend-dialects.html#types-7", [
          [ "LinearType", "backend-dialects.html#lineartype", null ],
          [ "SignalType", "backend-dialects.html#signaltype", null ]
        ] ]
      ] ]
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
"Bool_2IR_2Ops_8capi_8h_8inc.html#a83821c3dc8726dfab91dce6a2bbb4965",
"Cast_2IR_2Ops_8capi_8cpp_8inc.html#a492a8e82d8cd87fb4ac4124e25141812",
"Dialect_2InitDialects_8cpp_source.html",
"Felt_2IR_2Ops_8capi_8h_8inc.html#a489c5f340dc2401bd6c7ae6f57e401c8",
"Felt_2IR_2Types_8capi_8h_8inc.html#a56d5d831063c23265eaa6db5d75a320d",
"Function_2IR_2Ops_8capi_8h_8inc.html#aea5764a807db8d85bcb38eb5cfc00299",
"Include_2IR_2Dialect_8cpp.html",
"LLZK_2IR_2Attrs_8capi_8h_8inc.html#a0f5690d0a2c6a5da4f689665512c7bcd",
"POD_2IR_2Ops_8capi_8test_8cpp_8inc.html#a5f6b3a91feb42232094b53f22415ab3b",
"Polymorphic_2IR_2Ops_8capi_8h_8inc.html#aa5dae8a79bfcefb3762bbb19bb6dabf7",
"SMTTypes_8cpp.html#a69e2a4688847620f9e0b19eb7d719156",
"Struct_2IR_2Ops_8capi_8h_8inc.html#ad50f81a11e316b23968e6fc5b5b62c7a",
"Typing_8cpp.html#a5474acb433e02be6a84d634007e6f699",
"Verif_2IR_2Ops_8capi_8test_8cpp_8inc.html#a40942ea40642ea6ff98e57a86affed56",
"classllzk_1_1APIntValue.html",
"classllzk_1_1Field.html#a2abf8d65ff6ae1f4778cec3b164b5d08",
"classllzk_1_1IntervalAnalysisLatticeValue.html",
"classllzk_1_1NonDetOp.html#a10f34b3f7d82372061873a9e1925b014",
"classllzk_1_1SourceRefIndex.html#ad17cf8610d0a9deb5193f59fe8bd886c",
"classllzk_1_1SymbolLookupResultUntyped.html#aa3ab62d44927b2e88fbb11289e8147e8",
"classllzk_1_1array_1_1ArrayIndexGen.html#af299e13df8ee265e88b7f2a95eec2c1e",
"classllzk_1_1array_1_1ExtractArrayOp.html#ac246baafeca3b5c3afbd2c832397a8df",
"classllzk_1_1array_1_1WriteArrayOp.html#a693c2b6197bdec1cda9d6850769e305f",
"classllzk_1_1boolean_1_1AndBoolOp.html#a27a1865c2ddca39dc826bf0be608bb1a",
"classllzk_1_1boolean_1_1ExistsOp.html",
"classllzk_1_1boolean_1_1OrBoolOpAdaptor.html#acbf41887019c844ff1c4f5325d78920b",
"classllzk_1_1boolean_1_1detail_1_1ForAllOpGenericAdaptorBase.html#a73e7afac2ede2d65d91ca6921fc429ed",
"classllzk_1_1cast_1_1IntToFeltOpAdaptor.html#a9a05d041f73a5175c3b812bd0ef37f17",
"classllzk_1_1component_1_1MemberDefOpGenericAdaptor.html#ae782c55c962578519a5bd80ad1a0e4fc",
"classllzk_1_1component_1_1StructDefOp.html#a26f0834eef8efdad44b20af8cc85cdb6",
"classllzk_1_1component_1_1detail_1_1MemberWriteOpGenericAdaptorBase.html#ad4fb3782502cf26249d04671c050159a",
"classllzk_1_1constrain_1_1detail_1_1EmitEqualityOpGenericAdaptorBase.html#a079009b8e30f03c9bd85ee9e411908e1",
"classllzk_1_1felt_1_1AndFeltOpAdaptor.html#ac48e54c4d2b5c2639ce0156b5ed37e7e",
"classllzk_1_1felt_1_1InvFeltOpAdaptor.html#a538f7f29da976cc9c2ec6933b3d21999",
"classllzk_1_1felt_1_1OrFeltOp.html#a8bed652ab966b2e00bff2a414c9a3983",
"classllzk_1_1felt_1_1ShrFeltOpGenericAdaptor.html#a08e3d10bf6fd32558f0d4b53899aa99f",
"classllzk_1_1felt_1_1UnsignedIntDivFeltOp.html#a9f1ffd1fef273b23cdc8a112080cbbbe",
"classllzk_1_1felt_1_1detail_1_1FeltBinaryOpInterfaceInterfaceTraits_1_1ExternalModel.html#a86237995807888e512732f1c6a29dc3d",
"classllzk_1_1felt_1_1detail_1_1XorFeltOpGenericAdaptorBase.html#a21f65e01e9fb246ea5b75280333fbeeb",
"classllzk_1_1function_1_1FuncDefOp.html#a8567e2b05118dd899c133b4115632488",
"classllzk_1_1function_1_1detail_1_1FuncDefOpGenericAdaptorBase.html#ab48c3f12d171630cd840a554a0577346",
"classllzk_1_1global_1_1GlobalWriteOp.html#a9c2135fc4fbef1fa894d3a8c968d0f3b",
"classllzk_1_1impl_1_1ComputeConstrainToProductPassBase.html#ad73251238b30d0ce575c9ea433715a66",
"classllzk_1_1impl_1_1RedundantOperationEliminationPassBase.html#a53ea9aec50f4031ccab7fcfbfedf789c",
"classllzk_1_1include_1_1IncludeOp.html#a7ab8dbbef53dbdee7fb76e4a00da79d6",
"classllzk_1_1pod_1_1PODDialect.html#a4ef085a858cccbf328436ab22de82bd0",
"classllzk_1_1pod_1_1WritePodOpGenericAdaptor.html#a8a47f67327daf8183db88c08725cbc52",
"classllzk_1_1polymorphic_1_1ApplyMapOpGenericAdaptor.html#aa8bcff11552c788a0b01d7d8e227d4f7",
"classllzk_1_1polymorphic_1_1TemplateOpAdaptor.html#ae85fb29fd7042badc7559ddfc4f1fa15",
"classllzk_1_1polymorphic_1_1YieldOp.html#adfc3eee72caf4e17f9291e711296a705",
"classllzk_1_1polymorphic_1_1impl_1_1EmptyTemplateRemovalPassBase.html#aaf5cea5d405d57aa945e89bfccaa34de",
"classllzk_1_1ram_1_1detail_1_1StoreOpGenericAdaptorBase.html#aa27e98413d52e6c70dc02d36d8864616",
"classllzk_1_1smt_1_1ArraySelectOpGenericAdaptor.html",
"classllzk_1_1smt_1_1BVAShrOp.html#a2fcfb20485fe5b3c3f9b4120751ec546",
"classllzk_1_1smt_1_1BVCmpOp.html#abbe8b4891a0930e0489e799bd3f48aae",
"classllzk_1_1smt_1_1BVMulOpGenericAdaptor.html",
"classllzk_1_1smt_1_1BVSDivOp.html#ae264c0063d1691797e27cb8aeaf16694",
"classllzk_1_1smt_1_1BVUDivOp.html#a185c2e7d6fb69b561666b6ff680db59d",
"classllzk_1_1smt_1_1BoolConstantOp.html#abc844424155f2a9615dc857fecc45064",
"classllzk_1_1smt_1_1DeclareFunOpGenericAdaptor.html#a7ac2ccbe23c8e6f7a464de6021db13a8",
"classllzk_1_1smt_1_1ExtractOp.html#a1237526515f2f4aa430d576023e6206d",
"classllzk_1_1smt_1_1ImpliesOpGenericAdaptor.html#acc52833c6667525dad862908f42151c2",
"classllzk_1_1smt_1_1IntCmpOpAdaptor.html#a0ca79f3366f3b568dda3d7ca8452e0e1",
"classllzk_1_1smt_1_1IntModOpGenericAdaptor.html#acc52e3e44110a34ebc19aae0fef45ca9",
"classllzk_1_1smt_1_1IteOp.html#af04601b9392d9f8a917f5f867832af63",
"classllzk_1_1smt_1_1PushOp.html#a3e6c0a09f4c6703e090c9171fa633336",
"classllzk_1_1smt_1_1SMTOpVisitor.html#a4ace37b30dba569d65e1ebcf100c626a",
"classllzk_1_1smt_1_1SolverOp.html#a1eb1db7dcb2eff4d4c803d7d384b2ddb",
"classllzk_1_1smt_1_1detail_1_1ArraySelectOpGenericAdaptorBase.html#a60e2e57b33c175f16531affcfb6b832f",
"classllzk_1_1smt_1_1detail_1_1BVSDivOpGenericAdaptorBase.html#a82353f910a90f09fff218fd915305028",
"classllzk_1_1smt_1_1detail_1_1ExistsOpGenericAdaptorBase.html#ac1bc7aa28f290175d92350d8c4e26491",
"classllzk_1_1smt_1_1detail_1_1IntNegOpGenericAdaptorBase.html#aeb418ff0567d79b96920b8dc12a1a2f2",
"classllzk_1_1string_1_1LitStringOp.html",
"classllzk_1_1verif_1_1ContractOp.html#a9d8adf87f62b6698da990ce043788c92",
"classllzk_1_1verif_1_1EnsureConstrainOp.html#a00c8d95b2b216f9d4b8911fd0293db21",
"classllzk_1_1verif_1_1IncreasesOpGenericAdaptor.html#a0db688ebea9dcfa623ed80aa2cedf3c5",
"classllzk_1_1verif_1_1RequireComputeOpGenericAdaptor.html#a9e0a6748f7b4384e517e01bd5fa71543",
"classllzk_1_1verif_1_1detail_1_1ContractOpGenericAdaptorBase.html#aa615cfa8348d3f9ba28885993c618514",
"classllzk_1_1verif_1_1detail_1_1RequireComputeOpGenericAdaptorBase.html#a758d411e8b20098fb623243b6ed9cbc3",
"dialects.html#operands-43",
"dir_e0c8ca7768f72850137af2d53bd2396e.html",
"llzk-translate_8cpp.html#a3c04138a5bfe5d72780bb7e82a18e627",
"namespacellzk.html#a79c7df33a29f599757f22c2920d8addf",
"namespacellzk_1_1array.html#a2305a2dbd092e1bb3bfd90a2a6f7142a",
"namespacellzk_1_1smt.html#a7d6c4a8036ae417a39e27a6956963dba",
"structExtraMethod.html#adece370eb95e462adfa9e610bcaf3ec9",
"structXorFeltOpBuildFuncHelper.html#a73996088fa75d12cc273b295983eabeb",
"structllzk_1_1IntervalAnalysisContext.html#ad400aa87427b2d052f8d41e667696493",
"structllzk_1_1array_1_1detail_1_1CreateArrayOpGenericAdaptorBase_1_1Properties.html#aebd62dc523eaab8d21d05d0d03b12885",
"structllzk_1_1felt_1_1detail_1_1FeltBinaryOpInterfaceTrait.html",
"structllzk_1_1pod_1_1detail_1_1NewPodOpGenericAdaptorBase_1_1Properties.html#adbcc81217dfd174d0e753c52a3b9338f",
"structllzk_1_1smt_1_1detail_1_1BVConstantOpGenericAdaptorBase_1_1Properties.html#ad2b4a74a7b391021088d1925aa3fd1f5",
"structllzk_1_1verif_1_1ConditionOpInterface_1_1Trait.html",
"structllzk_1_1witgen_1_1StructValue.html"
];

var SYNCONMSG = 'click to disable panel synchronization';
var SYNCOFFMSG = 'click to enable panel synchronization';
var LISTOFALLMEMBERS = 'List of all members';