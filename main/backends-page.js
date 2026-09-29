var backends_page =
[
    [ "R1CS Backend", "r1cs-backend.html", [
      [ "'r1cs' Dialect", "r1cs-backend.html#r1cs-dialect", [
        [ "Operations", "r1cs-backend.html#operations-13", [
          [ "<span class=\"tt\">r1cs.add</span> (r1cs::AddOp)", "r1cs-backend.html#r1csadd-r1csaddop", [
            [ "Operands:", "r1cs-backend.html#operands-47", null ],
            [ "Results:", "r1cs-backend.html#results-42", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.circuit</span> (r1cs::CircuitDefOp)", "r1cs-backend.html#r1cscircuit-r1cscircuitdefop", [
            [ "Attributes:", "r1cs-backend.html#attributes-30", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.const</span> (r1cs::ConstOp)", "r1cs-backend.html#r1csconst-r1csconstop", [
            [ "Attributes:", "r1cs-backend.html#attributes-31", null ],
            [ "Results:", "r1cs-backend.html#results-43", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.constrain</span> (r1cs::ConstrainOp)", "r1cs-backend.html#r1csconstrain-r1csconstrainop", [
            [ "Operands:", "r1cs-backend.html#operands-48", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.def</span> (r1cs::SignalDefOp)", "r1cs-backend.html#r1csdef-r1cssignaldefop", [
            [ "Attributes:", "r1cs-backend.html#attributes-32", null ],
            [ "Results:", "r1cs-backend.html#results-44", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.mul_const</span> (r1cs::MulConstOp)", "r1cs-backend.html#r1csmul_const-r1csmulconstop", [
            [ "Attributes:", "r1cs-backend.html#attributes-33", null ],
            [ "Operands:", "r1cs-backend.html#operands-49", null ],
            [ "Results:", "r1cs-backend.html#results-45", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.neg</span> (r1cs::NegOp)", "r1cs-backend.html#r1csneg-r1csnegop", [
            [ "Operands:", "r1cs-backend.html#operands-50", null ],
            [ "Results:", "r1cs-backend.html#results-46", null ]
          ] ],
          [ "<span class=\"tt\">r1cs.to_linear</span> (r1cs::ToLinearOp)", "r1cs-backend.html#r1csto_linear-r1cstolinearop", [
            [ "Operands:", "r1cs-backend.html#operands-51", null ],
            [ "Results:", "r1cs-backend.html#results-47", null ]
          ] ]
        ] ],
        [ "Attributes", "r1cs-backend.html#attributes-34", [
          [ "FeltAttr", "r1cs-backend.html#feltattr", [
            [ "Parameters:", "r1cs-backend.html#parameters-11", null ]
          ] ],
          [ "PublicAttr", "r1cs-backend.html#publicattr-1", null ]
        ] ],
        [ "Types", "r1cs-backend.html#types-7", [
          [ "LinearType", "r1cs-backend.html#lineartype", null ],
          [ "SignalType", "r1cs-backend.html#signaltype", null ]
        ] ]
      ] ]
    ] ],
    [ "SMT Backend", "smt-backend.html", [
      [ "'smt_info' Dialect", "smt-backend.html#smt_info-dialect", [
        [ "Operations", "smt-backend.html#operations-14", [
          [ "<span class=\"tt\">smt_info.set</span> (llzk::smt_info::SMTInfoSetOp)", "smt-backend.html#smt_infoset-llzksmt_infosmtinfosetop", [
            [ "Attributes:", "smt-backend.html#attributes-35", null ]
          ] ]
        ] ],
        [ "Attributes", "smt-backend.html#attributes-36", [
          [ "KeywordAttr", "smt-backend.html#keywordattr", [
            [ "Parameters:", "smt-backend.html#parameters-12", null ]
          ] ],
          [ "SymbolAttr", "smt-backend.html#symbolattr", [
            [ "Parameters:", "smt-backend.html#parameters-13", null ]
          ] ]
        ] ]
      ] ]
    ] ],
    [ "PCL Backend", "pcl-backend.html", [
      [ "'pcl' Dialect", "pcl-backend.html#pcl-dialect", [
        [ "Operations", "pcl-backend.html#operations-15", [
          [ "<span class=\"tt\">pcl.add</span> (pcl::AddOp)", "pcl-backend.html#pcladd-pcladdop", [
            [ "Operands:", "pcl-backend.html#operands-52", null ],
            [ "Results:", "pcl-backend.html#results-48", null ]
          ] ],
          [ "<span class=\"tt\">pcl.and</span> (pcl::AndOp)", "pcl-backend.html#pcland-pclandop", [
            [ "Operands:", "pcl-backend.html#operands-53", null ],
            [ "Results:", "pcl-backend.html#results-49", null ]
          ] ],
          [ "<span class=\"tt\">pcl.asfelt</span> (pcl::AsFeltOp)", "pcl-backend.html#pclasfelt-pclasfeltop", [
            [ "Operands:", "pcl-backend.html#operands-54", null ],
            [ "Results:", "pcl-backend.html#results-50", null ]
          ] ],
          [ "<span class=\"tt\">pcl.assert</span> (pcl::AssertOp)", "pcl-backend.html#pclassert-pclassertop", [
            [ "Operands:", "pcl-backend.html#operands-55", null ]
          ] ],
          [ "<span class=\"tt\">pcl.assume.deterministic</span> (pcl::AssumeDeterministicOp)", "pcl-backend.html#pclassumedeterministic-pclassumedeterministicop", [
            [ "Operands:", "pcl-backend.html#operands-56", null ]
          ] ],
          [ "<span class=\"tt\">pcl.const</span> (pcl::ConstOp)", "pcl-backend.html#pclconst-pclconstop", [
            [ "Attributes:", "pcl-backend.html#attributes-37", null ],
            [ "Results:", "pcl-backend.html#results-51", null ]
          ] ],
          [ "<span class=\"tt\">pcl.det</span> (pcl::DetOp)", "pcl-backend.html#pcldet-pcldetop", [
            [ "Operands:", "pcl-backend.html#operands-57", null ],
            [ "Results:", "pcl-backend.html#results-52", null ]
          ] ],
          [ "<span class=\"tt\">pcl.eq</span> (pcl::CmpEqOp)", "pcl-backend.html#pcleq-pclcmpeqop", [
            [ "Operands:", "pcl-backend.html#operands-58", null ],
            [ "Results:", "pcl-backend.html#results-53", null ]
          ] ],
          [ "<span class=\"tt\">pcl.false</span> (pcl::FalseOp)", "pcl-backend.html#pclfalse-pclfalseop", [
            [ "Results:", "pcl-backend.html#results-54", null ]
          ] ],
          [ "<span class=\"tt\">pcl.ge</span> (pcl::CmpGeOp)", "pcl-backend.html#pclge-pclcmpgeop", [
            [ "Operands:", "pcl-backend.html#operands-59", null ],
            [ "Results:", "pcl-backend.html#results-55", null ]
          ] ],
          [ "<span class=\"tt\">pcl.gt</span> (pcl::CmpGtOp)", "pcl-backend.html#pclgt-pclcmpgtop", [
            [ "Operands:", "pcl-backend.html#operands-60", null ],
            [ "Results:", "pcl-backend.html#results-56", null ]
          ] ],
          [ "<span class=\"tt\">pcl.iff</span> (pcl::IffOp)", "pcl-backend.html#pcliff-pcliffop", [
            [ "Operands:", "pcl-backend.html#operands-61", null ],
            [ "Results:", "pcl-backend.html#results-57", null ]
          ] ],
          [ "<span class=\"tt\">pcl.implies</span> (pcl::ImpliesOp)", "pcl-backend.html#pclimplies-pclimpliesop", [
            [ "Operands:", "pcl-backend.html#operands-62", null ],
            [ "Results:", "pcl-backend.html#results-58", null ]
          ] ],
          [ "<span class=\"tt\">pcl.le</span> (pcl::CmpLeOp)", "pcl-backend.html#pclle-pclcmpleop", [
            [ "Operands:", "pcl-backend.html#operands-63", null ],
            [ "Results:", "pcl-backend.html#results-59", null ]
          ] ],
          [ "<span class=\"tt\">pcl.lt</span> (pcl::CmpLtOp)", "pcl-backend.html#pcllt-pclcmpltop", [
            [ "Operands:", "pcl-backend.html#operands-64", null ],
            [ "Results:", "pcl-backend.html#results-60", null ]
          ] ],
          [ "<span class=\"tt\">pcl.mul</span> (pcl::MulOp)", "pcl-backend.html#pclmul-pclmulop", [
            [ "Operands:", "pcl-backend.html#operands-65", null ],
            [ "Results:", "pcl-backend.html#results-61", null ]
          ] ],
          [ "<span class=\"tt\">pcl.neg</span> (pcl::NegOp)", "pcl-backend.html#pclneg-pclnegop", [
            [ "Operands:", "pcl-backend.html#operands-66", null ],
            [ "Results:", "pcl-backend.html#results-62", null ]
          ] ],
          [ "<span class=\"tt\">pcl.not</span> (pcl::NotOp)", "pcl-backend.html#pclnot-pclnotop", [
            [ "Operands:", "pcl-backend.html#operands-67", null ],
            [ "Results:", "pcl-backend.html#results-63", null ]
          ] ],
          [ "<span class=\"tt\">pcl.or</span> (pcl::OrOp)", "pcl-backend.html#pclor-pclorop", [
            [ "Operands:", "pcl-backend.html#operands-68", null ],
            [ "Results:", "pcl-backend.html#results-64", null ]
          ] ],
          [ "<span class=\"tt\">pcl.post_cond</span> (pcl::PostOp)", "pcl-backend.html#pclpost_cond-pclpostop", [
            [ "Operands:", "pcl-backend.html#operands-69", null ]
          ] ],
          [ "<span class=\"tt\">pcl.sub</span> (pcl::SubOp)", "pcl-backend.html#pclsub-pclsubop", [
            [ "Operands:", "pcl-backend.html#operands-70", null ],
            [ "Results:", "pcl-backend.html#results-65", null ]
          ] ],
          [ "<span class=\"tt\">pcl.true</span> (pcl::TrueOp)", "pcl-backend.html#pcltrue-pcltrueop", [
            [ "Results:", "pcl-backend.html#results-66", null ]
          ] ],
          [ "<span class=\"tt\">pcl.var</span> (pcl::VarOp)", "pcl-backend.html#pclvar-pclvarop", [
            [ "Attributes:", "pcl-backend.html#attributes-38", null ],
            [ "Results:", "pcl-backend.html#results-67", null ]
          ] ]
        ] ],
        [ "Attributes", "pcl-backend.html#attributes-39", [
          [ "FeltAttr", "pcl-backend.html#feltattr-1", [
            [ "Parameters:", "pcl-backend.html#parameters-14", null ]
          ] ],
          [ "BoolAttr", "pcl-backend.html#boolattr", [
            [ "Parameters:", "pcl-backend.html#parameters-15", null ]
          ] ],
          [ "PrimeAttr", "pcl-backend.html#primeattr", [
            [ "Parameters:", "pcl-backend.html#parameters-16", null ]
          ] ]
        ] ],
        [ "Types", "pcl-backend.html#types-8", [
          [ "BoolType", "pcl-backend.html#booltype", null ],
          [ "FeltType", "pcl-backend.html#felttype-1", null ]
        ] ]
      ] ]
    ] ],
    [ "ZKLean Backend", "zklean-backend.html", [
      [ "'ZKExpr' Dialect", "zklean-backend.html#zkexpr-dialect", [
        [ "Operations", "zklean-backend.html#operations-16", [
          [ "<span class=\"tt\">ZKExpr.Add</span> (llzk::zkexpr::AddOp)", "zklean-backend.html#zkexpradd-llzkzkexpraddop", [
            [ "Operands:", "zklean-backend.html#operands-71", null ],
            [ "Results:", "zklean-backend.html#results-68", null ]
          ] ],
          [ "<span class=\"tt\">ZKExpr.Literal</span> (llzk::zkexpr::LiteralOp)", "zklean-backend.html#zkexprliteral-llzkzkexprliteralop", [
            [ "Operands:", "zklean-backend.html#operands-72", null ],
            [ "Results:", "zklean-backend.html#results-69", null ]
          ] ],
          [ "<span class=\"tt\">ZKExpr.Mul</span> (llzk::zkexpr::MulOp)", "zklean-backend.html#zkexprmul-llzkzkexprmulop", [
            [ "Operands:", "zklean-backend.html#operands-73", null ],
            [ "Results:", "zklean-backend.html#results-70", null ]
          ] ],
          [ "<span class=\"tt\">ZKExpr.Neg</span> (llzk::zkexpr::NegOp)", "zklean-backend.html#zkexprneg-llzkzkexprnegop", [
            [ "Operands:", "zklean-backend.html#operands-74", null ],
            [ "Results:", "zklean-backend.html#results-71", null ]
          ] ],
          [ "<span class=\"tt\">ZKExpr.Sub</span> (llzk::zkexpr::SubOp)", "zklean-backend.html#zkexprsub-llzkzkexprsubop", [
            [ "Operands:", "zklean-backend.html#operands-75", null ],
            [ "Results:", "zklean-backend.html#results-72", null ]
          ] ]
        ] ],
        [ "Types", "zklean-backend.html#types-9", [
          [ "ComposedLookupTableType", "zklean-backend.html#composedlookuptabletype", [
            [ "Parameters:", "zklean-backend.html#parameters-17", null ]
          ] ],
          [ "WitnessIDType", "zklean-backend.html#witnessidtype", null ],
          [ "ZKExprType", "zklean-backend.html#zkexprtype", null ]
        ] ]
      ] ],
      [ "'ZKBuilder' Dialect", "zklean-backend.html#zkbuilder-dialect", [
        [ "Operations", "zklean-backend.html#operations-17", [
          [ "<span class=\"tt\">ZKBuilder.AllocWitness</span> (llzk::zkbuilder::AllocWitnessOp)", "zklean-backend.html#zkbuilderallocwitness-llzkzkbuilderallocwitnessop", [
            [ "Results:", "zklean-backend.html#results-73", null ]
          ] ],
          [ "<span class=\"tt\">ZKBuilder.ConstrainEq</span> (llzk::zkbuilder::ConstrainEqOp)", "zklean-backend.html#zkbuilderconstraineq-llzkzkbuilderconstraineqop", [
            [ "Operands:", "zklean-backend.html#operands-76", null ],
            [ "Results:", "zklean-backend.html#results-74", null ]
          ] ],
          [ "<span class=\"tt\">ZKBuilder.ConstrainR1CS</span> (llzk::zkbuilder::ConstrainR1CSOp)", "zklean-backend.html#zkbuilderconstrainr1cs-llzkzkbuilderconstrainr1csop", [
            [ "Operands:", "zklean-backend.html#operands-77", null ],
            [ "Results:", "zklean-backend.html#results-75", null ]
          ] ]
        ] ],
        [ "Types", "zklean-backend.html#types-10", [
          [ "ZKBuilderStateType", "zklean-backend.html#zkbuilderstatetype", null ]
        ] ]
      ] ],
      [ "'ZKLeanLean' Dialect", "zklean-backend.html#zkleanlean-dialect", [
        [ "Operations", "zklean-backend.html#operations-18", [
          [ "<span class=\"tt\">ZKLeanLean.accessor</span> (llzk::zkleanlean::AccessorOp)", "zklean-backend.html#zkleanleanaccessor-llzkzkleanleanaccessorop", [
            [ "Attributes:", "zklean-backend.html#attributes-40", null ],
            [ "Operands:", "zklean-backend.html#operands-78", null ],
            [ "Results:", "zklean-backend.html#results-76", null ]
          ] ],
          [ "<span class=\"tt\">ZKLeanLean.call</span> (llzk::zkleanlean::CallOp)", "zklean-backend.html#zkleanleancall-llzkzkleanleancallop", [
            [ "Attributes:", "zklean-backend.html#attributes-41", null ],
            [ "Operands:", "zklean-backend.html#operands-79", null ],
            [ "Results:", "zklean-backend.html#results-77", null ]
          ] ],
          [ "<span class=\"tt\">ZKLeanLean.member</span> (llzk::zkleanlean::MemberDefOp)", "zklean-backend.html#zkleanleanmember-llzkzkleanleanmemberdefop", [
            [ "Attributes:", "zklean-backend.html#attributes-42", null ]
          ] ],
          [ "<span class=\"tt\">ZKLeanLean.structure</span> (llzk::zkleanlean::StructDefOp)", "zklean-backend.html#zkleanleanstructure-llzkzkleanleanstructdefop", [
            [ "Attributes:", "zklean-backend.html#attributes-43", null ]
          ] ]
        ] ],
        [ "Types", "zklean-backend.html#types-11", [
          [ "StructType", "zklean-backend.html#structtype-1", [
            [ "Parameters:", "zklean-backend.html#parameters-18", null ]
          ] ]
        ] ]
      ] ]
    ] ]
];