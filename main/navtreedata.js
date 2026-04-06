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
            ] ]
          ] ],
          [ "'array' Dialect Transformation Passes", "tools.html#array-dialect-transformation-passes", [
            [ "<span class=\"tt\">-llzk-array-to-scalar</span>", "tools.html#autotoc_md-llzk-array-to-scalar", null ]
          ] ],
          [ "'polymorphic' Dialect Transformation Passes", "tools.html#polymorphic-dialect-transformation-passes", [
            [ "<span class=\"tt\">-llzk-drop-empty-params</span>", "tools.html#autotoc_md-llzk-drop-empty-params", null ],
            [ "<span class=\"tt\">-llzk-flatten</span>", "tools.html#autotoc_md-llzk-flatten", [
              [ "Options", "tools.html#options-9", null ]
            ] ]
          ] ],
          [ "Validation Passes", "tools.html#validation-passes", [
            [ "<span class=\"tt\">-llzk-validate-member-writes</span>", "tools.html#autotoc_md-llzk-validate-member-writes", null ]
          ] ]
        ] ]
      ] ],
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
            [ "Operands:", "dialects.html#operands-12", null ],
            [ "Results:", "dialects.html#results-9", null ]
          ] ],
          [ "<span class=\"tt\">cast.toindex</span> (llzk::cast::FeltToIndexOp)", "dialects.html#casttoindex-llzkcastfelttoindexop", [
            [ "Operands:", "dialects.html#operands-13", null ],
            [ "Results:", "dialects.html#results-10", null ]
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
            [ "Attributes:", "dialects.html#attributes-4", null ],
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
        [ "Attributes", "dialects.html#attributes-5", [
          [ "FeltConstAttr", "dialects.html#feltconstattr", [
            [ "Parameters:", "dialects.html#parameters-2", null ]
          ] ],
          [ "FieldSpecAttr", "dialects.html#fieldspecattr", [
            [ "Parameters:", "dialects.html#parameters-3", null ]
          ] ]
        ] ],
        [ "Types", "dialects.html#types-2", [
          [ "FeltType", "dialects.html#felttype", [
            [ "Parameters:", "dialects.html#parameters-4", null ]
          ] ]
        ] ]
      ] ],
      [ "'function' Dialect", "dialects.html#function-dialect", [
        [ "Operations", "dialects.html#operations-5", [
          [ "<span class=\"tt\">function.call</span> (llzk::function::CallOp)", "dialects.html#functioncall-llzkfunctioncallop", [
            [ "Attributes:", "dialects.html#attributes-6", null ],
            [ "Operands:", "dialects.html#operands-33", null ],
            [ "Results:", "dialects.html#results-29", null ]
          ] ],
          [ "<span class=\"tt\">function.def</span> (llzk::function::FuncDefOp)", "dialects.html#functiondef-llzkfunctionfuncdefop", [
            [ "Attributes:", "dialects.html#attributes-7", null ]
          ] ],
          [ "<span class=\"tt\">function.return</span> (llzk::function::ReturnOp)", "dialects.html#functionreturn-llzkfunctionreturnop", [
            [ "Operands:", "dialects.html#operands-34", null ]
          ] ]
        ] ]
      ] ],
      [ "'global' Dialect", "dialects.html#global-dialect", [
        [ "Operations", "dialects.html#operations-6", [
          [ "<span class=\"tt\">global.def</span> (llzk::global::GlobalDefOp)", "dialects.html#globaldef-llzkglobalglobaldefop", [
            [ "Attributes:", "dialects.html#attributes-8", null ]
          ] ],
          [ "<span class=\"tt\">global.read</span> (llzk::global::GlobalReadOp)", "dialects.html#globalread-llzkglobalglobalreadop", [
            [ "Attributes:", "dialects.html#attributes-9", null ],
            [ "Results:", "dialects.html#results-30", null ]
          ] ],
          [ "<span class=\"tt\">global.write</span> (llzk::global::GlobalWriteOp)", "dialects.html#globalwrite-llzkglobalglobalwriteop", [
            [ "Attributes:", "dialects.html#attributes-10", null ],
            [ "Operands:", "dialects.html#operands-35", null ]
          ] ]
        ] ]
      ] ],
      [ "'include' Dialect", "dialects.html#include-dialect", [
        [ "Operations", "dialects.html#operations-7", [
          [ "<span class=\"tt\">include.from</span> (llzk::include::IncludeOp)", "dialects.html#includefrom-llzkincludeincludeop", [
            [ "Attributes:", "dialects.html#attributes-11", null ]
          ] ]
        ] ]
      ] ],
      [ "'llzk' Dialect", "dialects.html#llzk-dialect", [
        [ "Operations", "dialects.html#operations-8", [
          [ "<span class=\"tt\">llzk.nondet</span> (llzk::NonDetOp)", "dialects.html#llzknondet-llzknondetop", [
            [ "Results:", "dialects.html#results-31", null ]
          ] ]
        ] ],
        [ "Attributes", "dialects.html#attributes-12", [
          [ "LoopBoundsAttr", "dialects.html#loopboundsattr", [
            [ "Parameters:", "dialects.html#parameters-5", null ]
          ] ],
          [ "PublicAttr", "dialects.html#publicattr", null ]
        ] ]
      ] ],
      [ "'pod' Dialect", "dialects.html#pod-dialect", [
        [ "Operations", "dialects.html#operations-9", [
          [ "<span class=\"tt\">pod.new</span> (llzk::pod::NewPodOp)", "dialects.html#podnew-llzkpodnewpodop", [
            [ "Attributes:", "dialects.html#attributes-13", null ],
            [ "Operands:", "dialects.html#operands-36", null ],
            [ "Results:", "dialects.html#results-32", null ]
          ] ],
          [ "<span class=\"tt\">pod.read</span> (llzk::pod::ReadPodOp)", "dialects.html#podread-llzkpodreadpodop", [
            [ "Attributes:", "dialects.html#attributes-14", null ],
            [ "Operands:", "dialects.html#operands-37", null ],
            [ "Results:", "dialects.html#results-33", null ]
          ] ],
          [ "<span class=\"tt\">pod.write</span> (llzk::pod::WritePodOp)", "dialects.html#podwrite-llzkpodwritepodop", [
            [ "Attributes:", "dialects.html#attributes-15", null ],
            [ "Operands:", "dialects.html#operands-38", null ]
          ] ]
        ] ],
        [ "Attributes", "dialects.html#attributes-16", [
          [ "RecordAttr", "dialects.html#recordattr", [
            [ "Parameters:", "dialects.html#parameters-6", null ]
          ] ]
        ] ],
        [ "Types", "dialects.html#types-3", [
          [ "PodType", "dialects.html#podtype", [
            [ "Parameters:", "dialects.html#parameters-7", null ]
          ] ]
        ] ]
      ] ],
      [ "'poly' Dialect", "dialects.html#poly-dialect", [
        [ "Operations", "dialects.html#operations-10", [
          [ "<span class=\"tt\">poly.applymap</span> (llzk::polymorphic::ApplyMapOp)", "dialects.html#polyapplymap-llzkpolymorphicapplymapop", [
            [ "Attributes:", "dialects.html#attributes-17", null ],
            [ "Operands:", "dialects.html#operands-39", null ],
            [ "Results:", "dialects.html#results-34", null ]
          ] ],
          [ "<span class=\"tt\">poly.read_const</span> (llzk::polymorphic::ConstReadOp)", "dialects.html#polyread_const-llzkpolymorphicconstreadop", [
            [ "Attributes:", "dialects.html#attributes-18", null ],
            [ "Results:", "dialects.html#results-35", null ]
          ] ],
          [ "<span class=\"tt\">poly.unifiable_cast</span> (llzk::polymorphic::UnifiableCastOp)", "dialects.html#polyunifiable_cast-llzkpolymorphicunifiablecastop", [
            [ "Operands:", "dialects.html#operands-40", null ],
            [ "Results:", "dialects.html#results-36", null ]
          ] ]
        ] ],
        [ "Types", "dialects.html#types-4", [
          [ "TypeVarType", "dialects.html#typevartype", [
            [ "Parameters:", "dialects.html#parameters-8", null ]
          ] ]
        ] ]
      ] ],
      [ "'string' Dialect", "dialects.html#string-dialect", [
        [ "Operations", "dialects.html#operations-11", [
          [ "<span class=\"tt\">string.new</span> (llzk::string::LitStringOp)", "dialects.html#stringnew-llzkstringlitstringop", [
            [ "Attributes:", "dialects.html#attributes-19", null ],
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
            [ "Attributes:", "dialects.html#attributes-20", null ]
          ] ],
          [ "<span class=\"tt\">struct.member</span> (llzk::component::MemberDefOp)", "dialects.html#structmember-llzkcomponentmemberdefop", [
            [ "Attributes:", "dialects.html#attributes-21", null ]
          ] ],
          [ "<span class=\"tt\">struct.new</span> (llzk::component::CreateStructOp)", "dialects.html#structnew-llzkcomponentcreatestructop", [
            [ "Results:", "dialects.html#results-38", null ]
          ] ],
          [ "<span class=\"tt\">struct.readm</span> (llzk::component::MemberReadOp)", "dialects.html#structreadm-llzkcomponentmemberreadop", [
            [ "Attributes:", "dialects.html#attributes-22", null ],
            [ "Operands:", "dialects.html#operands-41", null ],
            [ "Results:", "dialects.html#results-39", null ]
          ] ],
          [ "<span class=\"tt\">struct.writem</span> (llzk::component::MemberWriteOp)", "dialects.html#structwritem-llzkcomponentmemberwriteop", [
            [ "Attributes:", "dialects.html#attributes-23", null ],
            [ "Operands:", "dialects.html#operands-42", null ]
          ] ]
        ] ],
        [ "Types", "dialects.html#types-6", [
          [ "StructType", "dialects.html#structtype", [
            [ "Parameters:", "dialects.html#parameters-9", null ]
          ] ]
        ] ]
      ] ]
    ] ],
    [ "Backend Dialects", "backend-dialects.html", [
      [ "'r1cs' Dialect", "backend-dialects.html#r1cs-dialect", [
        [ "Operations", "backend-dialects.html#operations-13", [
          [ "<span class=\"tt\">r1cs.add</span> (r1cs::AddOp)", "backend-dialects.html#r1csadd-r1csaddop", [
            [ "Operands:", "backend-dialects.html#operands-43", null ],
            [ "Results:", "backend-dialects.html#results-40", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.circuit</span> (r1cs::CircuitDefOp)", "backend-dialects.html#r1cscircuit-r1cscircuitdefop", [
            [ "Attributes:", "backend-dialects.html#attributes-24", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.const</span> (r1cs::ConstOp)", "backend-dialects.html#r1csconst-r1csconstop", [
            [ "Attributes:", "backend-dialects.html#attributes-25", null ],
            [ "Results:", "backend-dialects.html#results-41", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.constrain</span> (r1cs::ConstrainOp)", "backend-dialects.html#r1csconstrain-r1csconstrainop", [
            [ "Operands:", "backend-dialects.html#operands-44", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.def</span> (r1cs::SignalDefOp)", "backend-dialects.html#r1csdef-r1cssignaldefop", [
            [ "Attributes:", "backend-dialects.html#attributes-26", null ],
            [ "Results:", "backend-dialects.html#results-42", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.mul_const</span> (r1cs::MulConstOp)", "backend-dialects.html#r1csmul_const-r1csmulconstop", [
            [ "Attributes:", "backend-dialects.html#attributes-27", null ],
            [ "Operands:", "backend-dialects.html#operands-45", null ],
            [ "Results:", "backend-dialects.html#results-43", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.neg</span> (r1cs::NegOp)", "backend-dialects.html#r1csneg-r1csnegop", [
            [ "Operands:", "backend-dialects.html#operands-46", null ],
            [ "Results:", "backend-dialects.html#results-44", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.to_linear</span> (r1cs::ToLinearOp)", "backend-dialects.html#r1csto_linear-r1cstolinearop", [
            [ "Operands:", "backend-dialects.html#operands-47", null ],
            [ "Results:", "backend-dialects.html#results-45", null ]
          ] ]
        ] ],
        [ "Attributes", "backend-dialects.html#attributes-28", [
          [ "FeltAttr", "backend-dialects.html#feltattr", [
            [ "Parameters:", "backend-dialects.html#parameters-10", null ]
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
"Bool_2IR_2Ops_8capi_8test_8cpp_8inc.html#ac303e7270975b83723ad15990e2186f9",
"Constrain_2IR_2Ops_8capi_8cpp_8inc.html#aa83eabd1cb524d8be7969f92c49a4870",
"Felt_2IR_2Ops_8capi_8cpp_8inc.html#a9fbdd152585648fbbb02440f809b3c4f",
"Felt_2IR_2Ops_8capi_8test_8cpp_8inc.html#a89b9d6ac06e21c6efa4f82310cb75783",
"Function_2IR_2Ops_8capi_8h_8inc.html#a1ad9f2565fce3b901f82db791b796043",
"Global_2IR_2Ops_8h.html",
"LLZK_2IR_2Attrs_8capi_8h_8inc.html#a616ed95f160bb71abd57c869f62d50c4",
"POD_2IR_2Ops_8cpp.html",
"SourceRefLattice_8cpp.html#ad78e062f62e0d6e453941fb4ca843e4d",
"Struct_2IR_2Ops_8capi_8test_8cpp_8inc.html#a63fbdd302109ffb617003b21bf76e3ee",
"backend-dialects.html#signaltype",
"classllzk_1_1ExpressionValue.html#a6b75cc20a9f12488c2af5c1632c0ed65",
"classllzk_1_1Interval.html#ae7090a31ef19bbaaf1787b08fb76a79aae976faae1fbc21b48e962a5024d7dff3",
"classllzk_1_1NonDetOp.html#a7c3f1780bea1b6d4af061f3ce6b56f15",
"classllzk_1_1SourceRefLattice.html#a3aa497adaeabbbf5c32e954249d6d776",
"classllzk_1_1SymbolUseGraph.html#a1d108465a51644111a5b53307679d50b",
"classllzk_1_1array_1_1ArrayLengthOp.html#afb4448e4f992a3c4a4d747135aeaf47a",
"classllzk_1_1array_1_1ExtractArrayOpGenericAdaptor.html#a7c784c03b57a49c3691113de502c4e15",
"classllzk_1_1array_1_1WriteArrayOpAdaptor.html#aac47cb2e36c33d57023a3b8e7fe4478d",
"classllzk_1_1boolean_1_1AndBoolOpGenericAdaptor.html#a677198e9c68ea375bf1c12efeae2f5c2",
"classllzk_1_1boolean_1_1NotBoolOpGenericAdaptor.html#a9d173449f953fa777b36f8298983aeeb",
"classllzk_1_1boolean_1_1detail_1_1XorBoolOpGenericAdaptorBase.html#aa0fa837432b904cd8e5f55c080c9cbb6",
"classllzk_1_1component_1_1MemberDefOp.html#a2994740401a835e4e9a890c4365e539d",
"classllzk_1_1component_1_1MemberWriteOp.html#a21decc943010cd07b49671d03e2969a9",
"classllzk_1_1component_1_1detail_1_1CreateStructOpGenericAdaptorBase.html#af7530d2214c366ebde218eaea979b724",
"classllzk_1_1constrain_1_1EmitEqualityOp.html#ac6359e81696302e0f6332e57c77c95d2",
"classllzk_1_1felt_1_1AddFeltOpAdaptor.html#a0194dc15a27926201c64071039be587f",
"classllzk_1_1felt_1_1FeltConstantOpGenericAdaptor.html#a3482dd4cfd223c28e58143fee3b60af8",
"classllzk_1_1felt_1_1NotFeltOp.html#a25daed2a6f7b9a018832d94081023287",
"classllzk_1_1felt_1_1ShlFeltOpAdaptor.html#ac91a875f414a27b18cb425d376748fb1",
"classllzk_1_1felt_1_1SubFeltOp.html#a5b2754f970273f7f7ae192d82695cd83",
"classllzk_1_1felt_1_1XorFeltOpAdaptor.html#ab2a584dba73ab6a6be8697b008dd1c4a",
"classllzk_1_1felt_1_1detail_1_1SignedIntDivFeltOpGenericAdaptorBase.html#a986b9a0d40bea701a3137b8c90e1f4c2",
"classllzk_1_1function_1_1FuncDefOp.html#a28ba9e4fc63a987e84a9c5c2edebcfc3",
"classllzk_1_1function_1_1detail_1_1CallOpGenericAdaptorBase.html#ac658de4dce09993da93e10230c8d6235",
"classllzk_1_1global_1_1GlobalRefOpInterface.html",
"classllzk_1_1impl_1_1ComputeConstrainToProductPassBase.html#a39e8657ae4f54e75e332b4e0e56dea96",
"classllzk_1_1impl_1_1PredecessorPrinterPassBase.html#ae528f34591a042c7af2cf9e19bc901d3",
"classllzk_1_1include_1_1detail_1_1IncludeOpGenericAdaptorBase.html#a411d1e0bd12c087bfc2cdc0bd6394c60",
"classllzk_1_1pod_1_1ReadPodOpGenericAdaptor.html#a05d10ac647372d2e01a3b53b95572631",
"classllzk_1_1polymorphic_1_1ApplyMapOpAdaptor.html#a26dd6cd183bdd98d6bdc6a7fc32ff5be",
"classllzk_1_1polymorphic_1_1detail_1_1CreateArrayOpClassReplacePattern.html",
"contribution-guide.html#make-changes",
"dir_303a645775d7c586afea5555a77293cf.html",
"llzk-c_2Builder_8h.html#a46e45fc93ddbc76257881ab5b8d5050c",
"namespacellzk.html#a6982559aa99cea04c0cf47ee38166ce2",
"namespacellzk_1_1array.html#a608683adb2db2b27afd796d5539d407c",
"structAttrOrTypeTestGenerator.html#ab2bf6842518e7302cc97842c5a53d940",
"structllvm_1_1DOTGraphTraits_3_01const_01llzk_1_1SymbolDefTreeNode_01_5_01_4.html#a20621aefb1d2c00057fd71d7fcf46cf7",
"structllzk_1_1PredecessorPrinterPassOptions.html#ab90aeb53740b01372d8a5956a6054c08",
"structllzk_1_1component_1_1detail_1_1MemberWriteOpGenericAdaptorBase_1_1Properties.html#a06ded30bc0d42b54ebbafbf861e8346b",
"structllzk_1_1include_1_1detail_1_1IncludeOpGenericAdaptorBase_1_1Properties.html#a0c9dfdb13f750dab7962d90ad2bd6eeb"
];

var SYNCONMSG = 'click to disable panel synchronization';
var SYNCOFFMSG = 'click to enable panel synchronization';
var LISTOFALLMEMBERS = 'List of all members';