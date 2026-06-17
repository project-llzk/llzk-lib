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
            [ "<span class=\"tt\">-llzk-inline-structs</span>", "tools.html#autotoc_md-llzk-inline-structs", [
              [ "Options", "tools.html#options-8", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-poly-lowering-pass</span>", "tools.html#autotoc_md-llzk-poly-lowering-pass", [
              [ "Options", "tools.html#options-9", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-remove-unused-discardable-allocations</span>", "tools.html#autotoc_md-llzk-remove-unused-discardable-allocations", [
              [ "Options", "tools.html#options-10", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-unused-declaration-elim</span>", "tools.html#autotoc_md-llzk-unused-declaration-elim", [
              [ "Options", "tools.html#options-11", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-while-to-for</span>", "tools.html#autotoc_md-llzk-while-to-for", null ]
          ] ],
          [ "'array' Dialect Transformation Passes", "tools.html#array-dialect-transformation-passes", [
            [ "<span class=\"tt\">-llzk-array-to-scalar</span>", "tools.html#autotoc_md-llzk-array-to-scalar", null ]
          ] ],
          [ "'polymorphic' Dialect Transformation Passes", "tools.html#polymorphic-dialect-transformation-passes", [
            [ "<span class=\"tt\">-llzk-drop-empty-templates</span>", "tools.html#autotoc_md-llzk-drop-empty-templates", null ],
            [ "<span class=\"tt\">-llzk-flatten</span>", "tools.html#autotoc_md-llzk-flatten", [
              [ "Options", "tools.html#options-12", null ]
            ] ]
          ] ],
          [ "Validation Passes", "tools.html#validation-passes", [
            [ "<span class=\"tt\">-llzk-validate-member-writes</span>", "tools.html#autotoc_md-llzk-validate-member-writes", null ]
          ] ]
        ] ]
      ] ],
      [ "llzk-witgen", "tools.html#llzk-witgen", null ],
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
          [ "<span class=\"tt\">bool.not</span> (llzk::boolean::NotBoolOp)", "dialects.html#boolnot-llzkbooleannotboolop", [
            [ "Operands:", "dialects.html#operands-9", null ],
            [ "Results:", "dialects.html#results-6", null ]
          ] ],
          [ "<span class=\"tt\">bool.or</span> (llzk::boolean::OrBoolOp)", "dialects.html#boolor-llzkbooleanorboolop", [
            [ "Operands:", "dialects.html#operands-10", null ],
            [ "Results:", "dialects.html#results-7", null ]
          ] ],
          [ "<span class=\"tt\">bool.xor</span> (llzk::boolean::XorBoolOp)", "dialects.html#boolxor-llzkbooleanxorboolop", [
            [ "Operands:", "dialects.html#operands-11", null ],
            [ "Results:", "dialects.html#results-8", null ]
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
            [ "Operands:", "dialects.html#operands-12", null ],
            [ "Results:", "dialects.html#results-9", null ]
          ] ],
          [ "<span class=\"tt\">cast.toindex</span> (llzk::cast::FeltToIndexOp)", "dialects.html#casttoindex-llzkcastfelttoindexop", [
            [ "Attributes:", "dialects.html#attributes-5", null ],
            [ "Operands:", "dialects.html#operands-13", null ],
            [ "Results:", "dialects.html#results-10", null ]
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
            [ "Operands:", "dialects.html#operands-14", null ]
          ] ],
          [ "<span class=\"tt\">constrain.in</span> (llzk::constrain::EmitContainmentOp)", "dialects.html#constrainin-llzkconstrainemitcontainmentop", [
            [ "Operands:", "dialects.html#operands-15", null ]
          ] ]
        ] ]
      ] ],
      [ "'felt' Dialect", "dialects.html#felt-dialect", [
        [ "Operations", "dialects.html#operations-4", [
          [ "<span class=\"tt\">felt.add</span> (llzk::felt::AddFeltOp)", "dialects.html#feltadd-llzkfeltaddfeltop", [
            [ "Operands:", "dialects.html#operands-16", null ],
            [ "Results:", "dialects.html#results-11", null ]
          ] ],
          [ "<span class=\"tt\">felt.bit_and</span> (llzk::felt::AndFeltOp)", "dialects.html#feltbit_and-llzkfeltandfeltop", [
            [ "Operands:", "dialects.html#operands-17", null ],
            [ "Results:", "dialects.html#results-12", null ]
          ] ],
          [ "<span class=\"tt\">felt.bit_not</span> (llzk::felt::NotFeltOp)", "dialects.html#feltbit_not-llzkfeltnotfeltop", [
            [ "Operands:", "dialects.html#operands-18", null ],
            [ "Results:", "dialects.html#results-13", null ]
          ] ],
          [ "<span class=\"tt\">felt.bit_or</span> (llzk::felt::OrFeltOp)", "dialects.html#feltbit_or-llzkfeltorfeltop", [
            [ "Operands:", "dialects.html#operands-19", null ],
            [ "Results:", "dialects.html#results-14", null ]
          ] ],
          [ "<span class=\"tt\">felt.bit_xor</span> (llzk::felt::XorFeltOp)", "dialects.html#feltbit_xor-llzkfeltxorfeltop", [
            [ "Operands:", "dialects.html#operands-20", null ],
            [ "Results:", "dialects.html#results-15", null ]
          ] ],
          [ "<span class=\"tt\">felt.const</span> (llzk::felt::FeltConstantOp)", "dialects.html#feltconst-llzkfeltfeltconstantop", [
            [ "Attributes:", "dialects.html#attributes-7", null ],
            [ "Results:", "dialects.html#results-16", null ]
          ] ],
          [ "<span class=\"tt\">felt.div</span> (llzk::felt::DivFeltOp)", "dialects.html#feltdiv-llzkfeltdivfeltop", [
            [ "Operands:", "dialects.html#operands-21", null ],
            [ "Results:", "dialects.html#results-17", null ]
          ] ],
          [ "<span class=\"tt\">felt.inv</span> (llzk::felt::InvFeltOp)", "dialects.html#feltinv-llzkfeltinvfeltop", [
            [ "Operands:", "dialects.html#operands-22", null ],
            [ "Results:", "dialects.html#results-18", null ]
          ] ],
          [ "<span class=\"tt\">felt.mul</span> (llzk::felt::MulFeltOp)", "dialects.html#feltmul-llzkfeltmulfeltop", [
            [ "Operands:", "dialects.html#operands-23", null ],
            [ "Results:", "dialects.html#results-19", null ]
          ] ],
          [ "<span class=\"tt\">felt.neg</span> (llzk::felt::NegFeltOp)", "dialects.html#feltneg-llzkfeltnegfeltop", [
            [ "Operands:", "dialects.html#operands-24", null ],
            [ "Results:", "dialects.html#results-20", null ]
          ] ],
          [ "<span class=\"tt\">felt.pow</span> (llzk::felt::PowFeltOp)", "dialects.html#feltpow-llzkfeltpowfeltop", [
            [ "Operands:", "dialects.html#operands-25", null ],
            [ "Results:", "dialects.html#results-21", null ]
          ] ],
          [ "<span class=\"tt\">felt.shl</span> (llzk::felt::ShlFeltOp)", "dialects.html#feltshl-llzkfeltshlfeltop", [
            [ "Operands:", "dialects.html#operands-26", null ],
            [ "Results:", "dialects.html#results-22", null ]
          ] ],
          [ "<span class=\"tt\">felt.shr</span> (llzk::felt::ShrFeltOp)", "dialects.html#feltshr-llzkfeltshrfeltop", [
            [ "Operands:", "dialects.html#operands-27", null ],
            [ "Results:", "dialects.html#results-23", null ]
          ] ],
          [ "<span class=\"tt\">felt.sintdiv</span> (llzk::felt::SignedIntDivFeltOp)", "dialects.html#feltsintdiv-llzkfeltsignedintdivfeltop", [
            [ "Operands:", "dialects.html#operands-28", null ],
            [ "Results:", "dialects.html#results-24", null ]
          ] ],
          [ "<span class=\"tt\">felt.smod</span> (llzk::felt::SignedModFeltOp)", "dialects.html#feltsmod-llzkfeltsignedmodfeltop", [
            [ "Operands:", "dialects.html#operands-29", null ],
            [ "Results:", "dialects.html#results-25", null ]
          ] ],
          [ "<span class=\"tt\">felt.sub</span> (llzk::felt::SubFeltOp)", "dialects.html#feltsub-llzkfeltsubfeltop", [
            [ "Operands:", "dialects.html#operands-30", null ],
            [ "Results:", "dialects.html#results-26", null ]
          ] ],
          [ "<span class=\"tt\">felt.uintdiv</span> (llzk::felt::UnsignedIntDivFeltOp)", "dialects.html#feltuintdiv-llzkfeltunsignedintdivfeltop", [
            [ "Operands:", "dialects.html#operands-31", null ],
            [ "Results:", "dialects.html#results-27", null ]
          ] ],
          [ "<span class=\"tt\">felt.umod</span> (llzk::felt::UnsignedModFeltOp)", "dialects.html#feltumod-llzkfeltunsignedmodfeltop", [
            [ "Operands:", "dialects.html#operands-32", null ],
            [ "Results:", "dialects.html#results-28", null ]
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
            [ "Operands:", "dialects.html#operands-33", null ],
            [ "Results:", "dialects.html#results-29", null ]
          ] ],
          [ "<span class=\"tt\">function.def</span> (llzk::function::FuncDefOp)", "dialects.html#functiondef-llzkfunctionfuncdefop", [
            [ "Attributes:", "dialects.html#attributes-10", null ]
          ] ],
          [ "<span class=\"tt\">function.return</span> (llzk::function::ReturnOp)", "dialects.html#functionreturn-llzkfunctionreturnop", [
            [ "Operands:", "dialects.html#operands-34", null ]
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
            [ "Results:", "dialects.html#results-30", null ]
          ] ],
          [ "<span class=\"tt\">global.write</span> (llzk::global::GlobalWriteOp)", "dialects.html#globalwrite-llzkglobalglobalwriteop", [
            [ "Attributes:", "dialects.html#attributes-13", null ],
            [ "Operands:", "dialects.html#operands-35", null ]
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
            [ "Results:", "dialects.html#results-31", null ]
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
            [ "Operands:", "dialects.html#operands-36", null ],
            [ "Results:", "dialects.html#results-32", null ]
          ] ],
          [ "<span class=\"tt\">pod.read</span> (llzk::pod::ReadPodOp)", "dialects.html#podread-llzkpodreadpodop", [
            [ "Attributes:", "dialects.html#attributes-17", null ],
            [ "Operands:", "dialects.html#operands-37", null ],
            [ "Results:", "dialects.html#results-33", null ]
          ] ],
          [ "<span class=\"tt\">pod.write</span> (llzk::pod::WritePodOp)", "dialects.html#podwrite-llzkpodwritepodop", [
            [ "Attributes:", "dialects.html#attributes-18", null ],
            [ "Operands:", "dialects.html#operands-38", null ]
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
            [ "Operands:", "dialects.html#operands-39", null ],
            [ "Results:", "dialects.html#results-34", null ]
          ] ],
          [ "<span class=\"tt\">poly.expr</span> (llzk::polymorphic::TemplateExprOp)", "dialects.html#polyexpr-llzkpolymorphictemplateexprop", [
            [ "Attributes:", "dialects.html#attributes-21", null ]
          ] ],
          [ "<span class=\"tt\">poly.param</span> (llzk::polymorphic::TemplateParamOp)", "dialects.html#polyparam-llzkpolymorphictemplateparamop", [
            [ "Attributes:", "dialects.html#attributes-22", null ]
          ] ],
          [ "<span class=\"tt\">poly.read_const</span> (llzk::polymorphic::ConstReadOp)", "dialects.html#polyread_const-llzkpolymorphicconstreadop", [
            [ "Attributes:", "dialects.html#attributes-23", null ],
            [ "Results:", "dialects.html#results-35", null ]
          ] ],
          [ "<span class=\"tt\">poly.template</span> (llzk::polymorphic::TemplateOp)", "dialects.html#polytemplate-llzkpolymorphictemplateop", [
            [ "Attributes:", "dialects.html#attributes-24", null ]
          ] ],
          [ "<span class=\"tt\">poly.unifiable_cast</span> (llzk::polymorphic::UnifiableCastOp)", "dialects.html#polyunifiable_cast-llzkpolymorphicunifiablecastop", [
            [ "Operands:", "dialects.html#operands-40", null ],
            [ "Results:", "dialects.html#results-36", null ]
          ] ],
          [ "<span class=\"tt\">poly.yield</span> (llzk::polymorphic::YieldOp)", "dialects.html#polyyield-llzkpolymorphicyieldop", [
            [ "Operands:", "dialects.html#operands-41", null ]
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
            [ "Results:", "dialects.html#results-37", null ]
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
            [ "Results:", "dialects.html#results-38", null ]
          ] ],
          [ "<span class=\"tt\">struct.readm</span> (llzk::component::MemberReadOp)", "dialects.html#structreadm-llzkcomponentmemberreadop", [
            [ "Attributes:", "dialects.html#attributes-28", null ],
            [ "Operands:", "dialects.html#operands-42", null ],
            [ "Results:", "dialects.html#results-39", null ]
          ] ],
          [ "<span class=\"tt\">struct.writem</span> (llzk::component::MemberWriteOp)", "dialects.html#structwritem-llzkcomponentmemberwriteop", [
            [ "Attributes:", "dialects.html#attributes-29", null ],
            [ "Operands:", "dialects.html#operands-43", null ]
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
            [ "Operands:", "backend-dialects.html#operands-44", null ],
            [ "Results:", "backend-dialects.html#results-40", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.circuit</span> (r1cs::CircuitDefOp)", "backend-dialects.html#r1cscircuit-r1cscircuitdefop", [
            [ "Attributes:", "backend-dialects.html#attributes-30", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.const</span> (r1cs::ConstOp)", "backend-dialects.html#r1csconst-r1csconstop", [
            [ "Attributes:", "backend-dialects.html#attributes-31", null ],
            [ "Results:", "backend-dialects.html#results-41", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.constrain</span> (r1cs::ConstrainOp)", "backend-dialects.html#r1csconstrain-r1csconstrainop", [
            [ "Operands:", "backend-dialects.html#operands-45", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.def</span> (r1cs::SignalDefOp)", "backend-dialects.html#r1csdef-r1cssignaldefop", [
            [ "Attributes:", "backend-dialects.html#attributes-32", null ],
            [ "Results:", "backend-dialects.html#results-42", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.mul_const</span> (r1cs::MulConstOp)", "backend-dialects.html#r1csmul_const-r1csmulconstop", [
            [ "Attributes:", "backend-dialects.html#attributes-33", null ],
            [ "Operands:", "backend-dialects.html#operands-46", null ],
            [ "Results:", "backend-dialects.html#results-43", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.neg</span> (r1cs::NegOp)", "backend-dialects.html#r1csneg-r1csnegop", [
            [ "Operands:", "backend-dialects.html#operands-47", null ],
            [ "Results:", "backend-dialects.html#results-44", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.to_linear</span> (r1cs::ToLinearOp)", "backend-dialects.html#r1csto_linear-r1cstolinearop", [
            [ "Operands:", "backend-dialects.html#operands-48", null ],
            [ "Results:", "backend-dialects.html#results-45", null ]
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
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"AbstractLatticeValue_8h.html",
"Array_2IR_2Ops_8capi_8test_8cpp_8inc.html#aec553ab569bfa77e48302cb4e923f7c9",
"Bool_2IR_2Ops_8capi_8test_8cpp_8inc.html",
"CommonCAPIGen_8cpp.html",
"Felt_2IR_2Dialect_8capi_8test_8cpp_8inc.html#a7434e3b0728d861a70a0d57e068d5c0f",
"Felt_2IR_2Ops_8capi_8h_8inc.html#adf6ad9a096c1602565582e23c6a1b549",
"Function_2IR_2Attrs_8capi_8h_8inc.html#aefc7e27e2f57096041a24fc9caf2228d",
"Function_2IR_2Ops_8h_source.html",
"LLZKComputeConstrainToProductPass_8cpp.html",
"OpCAPITestGen_8cpp.html",
"POD_8h.html#a1cc9b53108fefbe0ff632b8a94d60988",
"Polymorphic_2IR_2Types_8h_8inc.html",
"String_2IR_2Types_8capi_8cpp_8inc_source.html",
"Struct_8cpp.html",
"Verif_2IR_2Ops_8capi_8h_8inc.html#a5880aa4b2cb36768c3198cc84a9628dc",
"classllzk_1_1CallGraph.html#a215961c1cd28b8c777f8d8ff5bd51bc5",
"classllzk_1_1Field.html#a8431930191dedec16a151765541a3b57",
"classllzk_1_1IntervalDataFlowAnalysis.html#ad4cd6edf9c7ea6137de8ae33c8c0fd9b",
"classllzk_1_1NonDetOp.html#afc4d545b1e3765a34566215e146d4877",
"classllzk_1_1SourceRefLatticeValue.html#a9d4360bfd5246c3dc32a8835538bd66d",
"classllzk_1_1SymbolUseGraph.html#a75930e3924df0e706b0594348063b0f4",
"classllzk_1_1array_1_1ArrayLengthOp.html#afb4448e4f992a3c4a4d747135aeaf47a",
"classllzk_1_1array_1_1ExtractArrayOpGenericAdaptor.html#a7c784c03b57a49c3691113de502c4e15",
"classllzk_1_1array_1_1WriteArrayOp.html#af7676d7cdcb7a46a6d78a9622057c887",
"classllzk_1_1boolean_1_1AndBoolOpAdaptor.html#a58b7539a930f4d765483696bdc031fb9",
"classllzk_1_1boolean_1_1NotBoolOpAdaptor.html#a34736131ea4771b852199e97e1be03f5",
"classllzk_1_1boolean_1_1detail_1_1NotBoolOpGenericAdaptorBase.html#af7aae6ac2d07e9f2b7842800d9677574",
"classllzk_1_1cast_1_1detail_1_1IntToFeltOpGenericAdaptorBase.html#abd7d7c6ff1be3f4f4fb647d800334290",
"classllzk_1_1component_1_1MemberReadOp.html#acb5d0ea88f710ef59080b0a5981307e0",
"classllzk_1_1component_1_1StructDefOpAdaptor.html#a9ef3fbbafa71395237574c52cd076743",
"classllzk_1_1constrain_1_1EmitContainmentOp.html#ae8d4f1a32251e7d9e39fb709ec31c756",
"classllzk_1_1detail_1_1DiscardableAllocationAccessorOpInterfaceInterfaceTraits_1_1ExternalModel.html#a503f7740455e357cebb46369b415d16b",
"classllzk_1_1felt_1_1DivFeltOpGenericAdaptor.html#aca6c04db280d489c6c736aa988f7ba3a",
"classllzk_1_1felt_1_1MulFeltOpGenericAdaptor.html#ab3052d315d1690e9e2f35d02d2b8fa77",
"classllzk_1_1felt_1_1PowFeltOpAdaptor.html#a0b4ada2d34f94b7baa167ff1578876a0",
"classllzk_1_1felt_1_1SignedModFeltOp.html#a21ef4ad74390f0438d4e7d933ba666c2",
"classllzk_1_1felt_1_1UnsignedModFeltOp.html#aed46d0a1f7662d65e1e00c49fa1a27b0",
"classllzk_1_1felt_1_1detail_1_1NotFeltOpGenericAdaptorBase.html#a59d27198c26d666bddaab65994436a15",
"classllzk_1_1function_1_1CallOp.html#aad15350e2920b08df199e1e49b00a609",
"classllzk_1_1function_1_1FunctionDialect.html",
"classllzk_1_1global_1_1GlobalDefOpAdaptor.html#a5c76abc42bcf62eb4c19a656f59a4a05",
"classllzk_1_1global_1_1detail_1_1GlobalReadOpGenericAdaptorBase.html#a7f206d3620ef35693fdfe7a666fa092a",
"classllzk_1_1impl_1_1InlineStructsPassBase.html#a4c2edcb8be95f0d8814b55690330a962",
"classllzk_1_1impl_1_1RemoveUnusedDiscardableAllocationsPassBase.html#abf5c2ab3673ecd36e01af6306743f5fb",
"classllzk_1_1include_1_1detail_1_1IncludeOpGenericAdaptorBase.html#ac32b21f03442f3c7aff3eba5dcefb028",
"classllzk_1_1pod_1_1ReadPodOp.html#a505ca920ddfd5073c23d74ec17dc48ab",
"classllzk_1_1pod_1_1detail_1_1PodRefOpInterfaceInterfaceTraits_1_1FallbackModel.html#af097b2ab9f8e5bce8ab3f94d43a16c64",
"classllzk_1_1polymorphic_1_1ConstReadOpGenericAdaptor.html#a8b5b2fab23a804a1b6355d12c9a1bfbf",
"classllzk_1_1polymorphic_1_1TemplateParamOpAdaptor.html",
"classllzk_1_1polymorphic_1_1detail_1_1ConstReadOpGenericAdaptorBase.html#a52a9d03f6c33cd16f07ac4355c216e02",
"classllzk_1_1ram_1_1LoadOp.html#aae6981a00c5120a1ee041c40feb39b08",
"classllzk_1_1smt_1_1ApplyFuncOpAdaptor.html#a6a1ee73ad7a820c319b60ebf34e46179",
"classllzk_1_1smt_1_1AssertOp.html#a0fa7daea7c5cfa582834687d98545869",
"classllzk_1_1smt_1_1BVAddOpAdaptor.html#a1d4f1c95001de66a85ddb70031847066",
"classllzk_1_1smt_1_1BVConstantOpAdaptor.html#a7c638aac992489b64c2de99e6fea7ce0",
"classllzk_1_1smt_1_1BVNotOp.html#ad0488a546b991079d4786edb8c841176",
"classllzk_1_1smt_1_1BVSModOpGenericAdaptor.html#ad43fc18abfa3bc46a1e1bd8b31f2ad52",
"classllzk_1_1smt_1_1BVURemOpAdaptor.html#a0ae8d32b5f282d30298c0123d9ac6d5f",
"classllzk_1_1smt_1_1CheckOpGenericAdaptor.html#a9fdc7124e01637fc8585d55bc46572cb",
"classllzk_1_1smt_1_1EqOp.html#ad799d9a0e05a97cda9031ae2edf3dcbf",
"classllzk_1_1smt_1_1ForallOp.html#a635b55e28943eedd7f4a0189270913bf",
"classllzk_1_1smt_1_1IntAbsOpAdaptor.html#a5010c35f02e2bdb8f09f13b4257c6c76",
"classllzk_1_1smt_1_1IntConstantOpGenericAdaptor.html#a85166c5d449a3484b4e853491e092a93",
"classllzk_1_1smt_1_1IntNegOpAdaptor.html#aa3d7e3b98efe986fd044912baeb2b4ec",
"classllzk_1_1smt_1_1OrOp.html#a21742790318fe8ccd501827b7a7780ce",
"classllzk_1_1smt_1_1RepeatOpAdaptor.html#aae96eb6ff05bcbc21cebf5265a8e3db8",
"classllzk_1_1smt_1_1SetLogicOp.html#a569bda78c864a21f6df2f447061d93f6",
"classllzk_1_1smt_1_1YieldOpGenericAdaptor.html#a4fd935b6d3147bedd6b97dbf41ec2d55",
"classllzk_1_1smt_1_1detail_1_1BVMulOpGenericAdaptorBase.html#a47935e285bcb367820c12e8e0a898018",
"classllzk_1_1smt_1_1detail_1_1DeclareFunOpGenericAdaptorBase.html#a8a7b88e113f41f9d99c3059ab5403c14",
"classllzk_1_1smt_1_1detail_1_1IntConstantOpGenericAdaptorBase.html#ab46126ed6a0d6f33643d75fce8466177",
"classllzk_1_1smt_1_1detail_1_1SolverOpGenericAdaptorBase.html#af048e484d1a687caf6661945b4cfae57",
"classllzk_1_1verif_1_1ContractOp.html#a58f385733873e6cca1ff286c3ce53c52",
"classllzk_1_1verif_1_1IncludeOp.html#a1d1327493504c58f2ab1eca69a92caf0",
"classllzk_1_1verif_1_1RequireConstrainOpAdaptor.html#ad2d951a691b873d040be6f13435eff5b",
"classllzk_1_1witgen_1_1FunctionInterpreter.html#ab700212fa9dc041e2a23e7395abbf8c7",
"dialects.html#results-24",
"functions_type_e.html",
"namespacellzk.html#a2176aa1303939b2957b8385543246aea",
"namespacellzk.html#ac64174fed27e22bfcf57e4332be0b8f2ad546f09aaf7aeef1f946a31089d7387d",
"namespacellzk_1_1function.html#af06d600ccce5f2a82dcf17c5437f1ac9a0b0a718d4ce7399e6723abe9f1e5f996",
"namespacemembers_func_w.html",
"structOpHeaderGenerator.html#a65528049bf947680634ad8f5d207207f",
"structllzk_1_1CommutativeMatcher.html#a1a4db8bcdacbd1fc1240e0e5cc09a468",
"structllzk_1_1array_1_1detail_1_1ArrayRefOpInterfaceInterfaceTraits.html",
"structllzk_1_1component_1_1detail_1_1StructTypeStorage.html",
"structllzk_1_1include_1_1detail_1_1IncludeOpGenericAdaptorBase_1_1Properties.html#a91fe7d040cfcd1777b2ba1ff12e09193",
"structllzk_1_1smt_1_1detail_1_1BV2IntOpGenericAdaptorBase_1_1Properties.html",
"structllzk_1_1smt_1_1detail_1_1SortTypeStorage.html#abb038f99100f697a7d8d032f0bbce9dc",
"structmlir_1_1FieldParser_3_01std_1_1optional_3_1_1llzk_1_1cast_1_1OverflowSemantics_01_4_00_01se5833e255e5fc640d3165f8450b95442.html"
];

var SYNCONMSG = 'click to disable panel synchronization';
var SYNCOFFMSG = 'click to enable panel synchronization';
var LISTOFALLMEMBERS = 'List of all members';