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
            [ "<span class=\"tt\">-llzk-print-call-graph</span>", "tools.html#autotoc_md-llzk-print-call-graph", null ],
            [ "<span class=\"tt\">-llzk-print-call-graph-sccs</span>", "tools.html#autotoc_md-llzk-print-call-graph-sccs", null ],
            [ "<span class=\"tt\">-llzk-print-constraint-dependency-graphs</span>", "tools.html#autotoc_md-llzk-print-constraint-dependency-graphs", [
              [ "Options", "tools.html#options", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-print-interval-analysis</span>", "tools.html#autotoc_md-llzk-print-interval-analysis", [
              [ "Options", "tools.html#options-1", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-print-predecessors</span>", "tools.html#autotoc_md-llzk-print-predecessors", [
              [ "Options", "tools.html#options-2", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-print-symbol-def-tree</span>", "tools.html#autotoc_md-llzk-print-symbol-def-tree", [
              [ "Options", "tools.html#options-3", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-print-symbol-use-graph</span>", "tools.html#autotoc_md-llzk-print-symbol-use-graph", [
              [ "Options", "tools.html#options-4", null ]
            ] ]
          ] ],
          [ "General Transformation Passes", "tools.html#general-transformation-passes", [
            [ "<span class=\"tt\">-llzk-compute-constrain-to-product</span>", "tools.html#autotoc_md-llzk-compute-constrain-to-product", [
              [ "Options", "tools.html#options-5", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-duplicate-op-elim</span>", "tools.html#autotoc_md-llzk-duplicate-op-elim", null ],
            [ "<span class=\"tt\">-llzk-duplicate-read-write-elim</span>", "tools.html#autotoc_md-llzk-duplicate-read-write-elim", null ],
            [ "<span class=\"tt\">-llzk-enforce-no-overwrite</span>", "tools.html#autotoc_md-llzk-enforce-no-overwrite", null ],
            [ "<span class=\"tt\">-llzk-fuse-product-loops</span>", "tools.html#autotoc_md-llzk-fuse-product-loops", null ],
            [ "<span class=\"tt\">-llzk-inline-structs</span>", "tools.html#autotoc_md-llzk-inline-structs", [
              [ "Options", "tools.html#options-6", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-poly-lowering-pass</span>", "tools.html#autotoc_md-llzk-poly-lowering-pass", [
              [ "Options", "tools.html#options-7", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-unused-declaration-elim</span>", "tools.html#autotoc_md-llzk-unused-declaration-elim", [
              [ "Options", "tools.html#options-8", null ]
            ] ],
            [ "<span class=\"tt\">-llzk-while-to-for</span>", "tools.html#autotoc_md-llzk-while-to-for", null ]
          ] ],
          [ "'array' Dialect Transformation Passes", "tools.html#array-dialect-transformation-passes", [
            [ "<span class=\"tt\">-llzk-array-to-scalar</span>", "tools.html#autotoc_md-llzk-array-to-scalar", null ]
          ] ],
          [ "'polymorphic' Dialect Transformation Passes", "tools.html#polymorphic-dialect-transformation-passes", [
            [ "<span class=\"tt\">-llzk-drop-empty-templates</span>", "tools.html#autotoc_md-llzk-drop-empty-templates", null ],
            [ "<span class=\"tt\">-llzk-flatten</span>", "tools.html#autotoc_md-llzk-flatten", [
              [ "Options", "tools.html#options-9", null ]
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
"Array_2IR_2Ops_8cpp_source.html",
"Bool_2IR_2Ops_8capi_8test_8cpp_8inc.html#a2bef18d8b6aba0cf9d90df662862873b",
"CommonCAPIGen_8cpp.html#a5149c52488bfc9f817d626f09a1738fb",
"Felt_2IR_2OpInterfaces_8h_8inc.html",
"Felt_2IR_2Ops_8capi_8test_8cpp_8inc.html#a106b37ffa8a82c24b6b5414f80e4b879",
"Function_2IR_2Dialect_8cpp.html",
"Function_8h.html#a6f5189b8e68ab4545d46db506a71efea",
"LLZKInlineStructsPass_8cpp.html#ad78e062f62e0d6e453941fb4ca843e4d",
"POD_2IR_2Attrs_8capi_8test_8cpp_8inc.html#ab9a65ec6d35328bb918254bab1b8d258",
"Poly_8cpp.html#addea62220081c5b73852c2781e8594fc",
"RAM_2IR_2Dialect_8cpp_8inc_source.html",
"Struct_2IR_2Dialect_8td_source.html",
"Struct_8h.html#abb9c486f519da6505a74784b076ff8d0",
"Verif_2IR_2Ops_8capi_8test_8cpp_8inc.html#a0b5ab00f6fa77256f49924468b06e05d",
"classllzk_1_1CallGraphNode.html#a2f2179c1cf322575df713c0baab9f871",
"classllzk_1_1FunctionTypeConverter.html#ab7f9abff2079f8bca7b9d3fb387a6573",
"classllzk_1_1LLZKSymbolTableImplTrait.html",
"classllzk_1_1NonDetOpGenericAdaptor.html",
"classllzk_1_1SourceRefLatticeValue.html#aceaa981713550fcca3e5fcaea6b788d3",
"classllzk_1_1SymbolUseGraph.html#ab669d5d2b10bf093195cb35e048ccc74",
"classllzk_1_1array_1_1ArrayLengthOp.html#ad8295f825eb7eca78e470961520a43b9",
"classllzk_1_1array_1_1ExtractArrayOpGenericAdaptor.html#a544945d51bd3c5115978ae013ce69a80",
"classllzk_1_1array_1_1WriteArrayOpAdaptor.html#a8692c7267e753de2a54de32fdfa6e103",
"classllzk_1_1boolean_1_1AndBoolOpGenericAdaptor.html#a2f0ed0e9a345649f86c9b26ec13270b3",
"classllzk_1_1boolean_1_1NotBoolOpGenericAdaptor.html",
"classllzk_1_1boolean_1_1detail_1_1OrBoolOpGenericAdaptorBase.html#ab0133ead19a71a69bad04b6efdf99b3e",
"classllzk_1_1component_1_1CreateStructOp.html#a0bcfe71cf082bb1fe81da835b9587154",
"classllzk_1_1component_1_1MemberReadOp.html#adfb1ea2fa7abd13e4e9a7ec4c0cc63de",
"classllzk_1_1component_1_1StructDefOpAdaptor.html#aeab2da7cdc872363cc4f98af205ad8d8",
"classllzk_1_1constrain_1_1EmitContainmentOpAdaptor.html#a80f0226123e24149bbe9c7228432bfd9",
"classllzk_1_1detail_1_1NonDetOpGenericAdaptorBase.html#abdc8744ddd054e999b4c1593cee74049",
"classllzk_1_1felt_1_1FeltConstantOp.html#a78a9d6f2e3cd5199638232c0e6a4998e",
"classllzk_1_1felt_1_1NegFeltOp.html#ad9766ae409306ff34b7653409f2a875b",
"classllzk_1_1felt_1_1ShlFeltOp.html#a085f260d09f4d287199f1095123f2da6",
"classllzk_1_1felt_1_1SignedModFeltOp.html#afc4b53324e7880ec9d2be005139a7561",
"classllzk_1_1felt_1_1UnsignedModFeltOpGenericAdaptor.html#afec0974258a4d6ce9526a8f0ea83b77d",
"classllzk_1_1felt_1_1detail_1_1PowFeltOpGenericAdaptorBase.html#a59c81b07adc95f80286b7a4b90c203b6",
"classllzk_1_1function_1_1CallOp.html#ae6d849f736448db91b1ede97a28f4050",
"classllzk_1_1function_1_1ReturnOp.html#a85696b5c27cd8e7f431e90dde4b49d4c",
"classllzk_1_1global_1_1GlobalReadOp.html",
"classllzk_1_1global_1_1detail_1_1GlobalWriteOpGenericAdaptorBase.html#a3cbe4a35c70294815df33992ab9821dc",
"classllzk_1_1impl_1_1MemberWriteValidatorPassBase.html#a0d887ca6e576f3cc676e588d56703070",
"classllzk_1_1impl_1_1WhileToForPassBase.html#a9015934829c2a0d0e00fe24603d9159e",
"classllzk_1_1pod_1_1NewPodOpAdaptor.html#a72decce2c1cb91af56215c0f2cbc3b10",
"classllzk_1_1pod_1_1WritePodOp.html#ad5b52f5e311886f08c344600ce223e86",
"classllzk_1_1polymorphic_1_1ApplyMapOpAdaptor.html#a3211ce4b5368b5a0bcd516f7fd5392a6",
"classllzk_1_1polymorphic_1_1TemplateOp.html#ab486a0d8cc2bd09838798c4a970de4ff",
"classllzk_1_1polymorphic_1_1YieldOp.html",
"classllzk_1_1polymorphic_1_1detail_1_1YieldOpGenericAdaptorBase.html#a8ed929675541f9b5622c6aada04fd0cb",
"classllzk_1_1smt_1_1AndOp.html#a70d44b79d6ab929d8e50dbcc4ecbf5c2",
"classllzk_1_1smt_1_1ArraySelectOpGenericAdaptor.html#a95e8658e6b13911928f4dc6aef00039a",
"classllzk_1_1smt_1_1BVAShrOp.html#a9d346a765cc9e6ea89ec39455b9f1d58",
"classllzk_1_1smt_1_1BVCmpOpAdaptor.html#a506b36904e67ca8bc409a763a29a0a10",
"classllzk_1_1smt_1_1BVMulOpGenericAdaptor.html#a96d9f2f0fe8cc35cc24ab3001065899b",
"classllzk_1_1smt_1_1BVSDivOpAdaptor.html#ab4dfba4f452d7aba96764be85cff50eb",
"classllzk_1_1smt_1_1BVUDivOp.html#a547bde529a52cf89075370b4ff442dfd",
"classllzk_1_1smt_1_1BoolConstantOpAdaptor.html#a20d08357100f4e62935755ff9957bc58",
"classllzk_1_1smt_1_1DistinctOp.html#a10640dadb197711e7d3d6789655e3325",
"classllzk_1_1smt_1_1ExtractOp.html#a636ecd85507deb75c7676b6d04d675cc",
"classllzk_1_1smt_1_1Int2BVOp.html#a1670aad8d43660406852fb170ef1bdef",
"classllzk_1_1smt_1_1IntCmpOpGenericAdaptor.html",
"classllzk_1_1smt_1_1IntMulOp.html#a492d85bf0e530e0553616944c2fe4706",
"classllzk_1_1smt_1_1IteOpAdaptor.html#a7423ff1e5de42242612460eb8f0f6a8c",
"classllzk_1_1smt_1_1PushOp.html#a9ae89966de82888d32aeafffdc75a2d5",
"classllzk_1_1smt_1_1SMTOpVisitor.html#a704329a373cc782eb1c9c7db552d5ccb",
"classllzk_1_1smt_1_1XOrOp.html#add42e658edc22eeeb64a63bdf92bb85a",
"classllzk_1_1smt_1_1detail_1_1BVAddOpGenericAdaptorBase.html#ac523af36d4968bebcab4526a91d80e50",
"classllzk_1_1smt_1_1detail_1_1BVXOrOpGenericAdaptorBase.html#a4622335a4854a7c486c6923d9997247c",
"classllzk_1_1smt_1_1detail_1_1Int2BVOpGenericAdaptorBase.html#a5a14bbdd31de8183444162a10c481fae",
"classllzk_1_1smt_1_1detail_1_1PushOpGenericAdaptorBase.html#a136913f6a1dfc4052fd5b3a0a0ffae5c",
"classllzk_1_1verif_1_1ContractOp.html",
"classllzk_1_1verif_1_1EnsureConstrainOpAdaptor.html#aad6d10378da1087d246df1586807818a",
"classllzk_1_1verif_1_1RequireConstrainOp.html#a16331bfcdd6087abf88fed750b21581d",
"classllzk_1_1witgen_1_1FunctionInterpreter.html",
"dialects.html#results-12",
"functions_o.html",
"namespacellzk.html#a1998bc2f299c5283fc5c74a324306f3b",
"namespacellzk.html#ac05a1e5322d050b620a18869c6e310b4",
"namespacellzk_1_1function.html#af06d600ccce5f2a82dcf17c5437f1ac9a0b0a718d4ce7399e6723abe9f1e5f996",
"structAttrOrTypeHeaderGenerator.html#ad65f025d0230510bc743ae199329a6c3",
"structSignedIntDivFeltOpBuildFuncHelper.html#a63ed948121b5f553baa95188df466442",
"structllzk_1_1IntervalAnalysisContext.html#a48ff4cc2ce95d178efbd90f3cb2d682b",
"structllzk_1_1boolean_1_1detail_1_1AssertOpGenericAdaptorBase_1_1Properties.html#ac01045c099be935f6fcd9444865303a7",
"structllzk_1_1function_1_1detail_1_1CallOpGenericAdaptorBase_1_1Properties.html#a12480cf20a884a5fe4f4c26cf6cd2a94",
"structllzk_1_1pod_1_1detail_1_1PodRefOpInterfaceTrait.html",
"structllzk_1_1smt_1_1detail_1_1DeclareFunOpGenericAdaptorBase_1_1Properties.html#a40cd9a06fe2f0b8712c6b62b5247f172",
"structllzk_1_1verif_1_1detail_1_1IncludeOpGenericAdaptorBase_1_1Properties.html#a0128173db2d3f7c7412ceba3f5b089a4"
];

var SYNCONMSG = 'click to disable panel synchronization';
var SYNCOFFMSG = 'click to enable panel synchronization';
var LISTOFALLMEMBERS = 'List of all members';