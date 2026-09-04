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
      [ "'smt' Dialect", "smt-backend.html#smt-dialect", [
        [ "Operations", "smt-backend.html#operations-14", [
          [ "<span class=\"tt\">smt.and</span> (llzk::smt::AndOp)", "smt-backend.html#smtand-llzksmtandop", [
            [ "Operands:", "smt-backend.html#operands-52", null ],
            [ "Results:", "smt-backend.html#results-48", null ]
          ] ],
          [ "<span class=\"tt\">smt.apply_func</span> (llzk::smt::ApplyFuncOp)", "smt-backend.html#smtapply_func-llzksmtapplyfuncop", [
            [ "Operands:", "smt-backend.html#operands-53", null ],
            [ "Results:", "smt-backend.html#results-49", null ]
          ] ],
          [ "<span class=\"tt\">smt.array.broadcast</span> (llzk::smt::ArrayBroadcastOp)", "smt-backend.html#smtarraybroadcast-llzksmtarraybroadcastop", [
            [ "Operands:", "smt-backend.html#operands-54", null ],
            [ "Results:", "smt-backend.html#results-50", null ]
          ] ],
          [ "<span class=\"tt\">smt.array.select</span> (llzk::smt::ArraySelectOp)", "smt-backend.html#smtarrayselect-llzksmtarrayselectop", [
            [ "Operands:", "smt-backend.html#operands-55", null ],
            [ "Results:", "smt-backend.html#results-51", null ]
          ] ],
          [ "<span class=\"tt\">smt.array.store</span> (llzk::smt::ArrayStoreOp)", "smt-backend.html#smtarraystore-llzksmtarraystoreop", [
            [ "Operands:", "smt-backend.html#operands-56", null ],
            [ "Results:", "smt-backend.html#results-52", null ]
          ] ],
          [ "<span class=\"tt\">smt.assert</span> (llzk::smt::AssertOp)", "smt-backend.html#smtassert-llzksmtassertop", [
            [ "Operands:", "smt-backend.html#operands-57", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.add</span> (llzk::smt::BVAddOp)", "smt-backend.html#smtbvadd-llzksmtbvaddop", [
            [ "Operands:", "smt-backend.html#operands-58", null ],
            [ "Results:", "smt-backend.html#results-53", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.and</span> (llzk::smt::BVAndOp)", "smt-backend.html#smtbvand-llzksmtbvandop", [
            [ "Operands:", "smt-backend.html#operands-59", null ],
            [ "Results:", "smt-backend.html#results-54", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.ashr</span> (llzk::smt::BVAShrOp)", "smt-backend.html#smtbvashr-llzksmtbvashrop", [
            [ "Operands:", "smt-backend.html#operands-60", null ],
            [ "Results:", "smt-backend.html#results-55", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.cmp</span> (llzk::smt::BVCmpOp)", "smt-backend.html#smtbvcmp-llzksmtbvcmpop", [
            [ "Attributes:", "smt-backend.html#attributes-35", null ],
            [ "Operands:", "smt-backend.html#operands-61", null ],
            [ "Results:", "smt-backend.html#results-56", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.concat</span> (llzk::smt::ConcatOp)", "smt-backend.html#smtbvconcat-llzksmtconcatop", [
            [ "Operands:", "smt-backend.html#operands-62", null ],
            [ "Results:", "smt-backend.html#results-57", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.constant</span> (llzk::smt::BVConstantOp)", "smt-backend.html#smtbvconstant-llzksmtbvconstantop", [
            [ "Attributes:", "smt-backend.html#attributes-36", null ],
            [ "Results:", "smt-backend.html#results-58", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.extract</span> (llzk::smt::ExtractOp)", "smt-backend.html#smtbvextract-llzksmtextractop", [
            [ "Attributes:", "smt-backend.html#attributes-37", null ],
            [ "Operands:", "smt-backend.html#operands-63", null ],
            [ "Results:", "smt-backend.html#results-59", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.lshr</span> (llzk::smt::BVLShrOp)", "smt-backend.html#smtbvlshr-llzksmtbvlshrop", [
            [ "Operands:", "smt-backend.html#operands-64", null ],
            [ "Results:", "smt-backend.html#results-60", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.mul</span> (llzk::smt::BVMulOp)", "smt-backend.html#smtbvmul-llzksmtbvmulop", [
            [ "Operands:", "smt-backend.html#operands-65", null ],
            [ "Results:", "smt-backend.html#results-61", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.neg</span> (llzk::smt::BVNegOp)", "smt-backend.html#smtbvneg-llzksmtbvnegop", [
            [ "Operands:", "smt-backend.html#operands-66", null ],
            [ "Results:", "smt-backend.html#results-62", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.not</span> (llzk::smt::BVNotOp)", "smt-backend.html#smtbvnot-llzksmtbvnotop", [
            [ "Operands:", "smt-backend.html#operands-67", null ],
            [ "Results:", "smt-backend.html#results-63", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.or</span> (llzk::smt::BVOrOp)", "smt-backend.html#smtbvor-llzksmtbvorop", [
            [ "Operands:", "smt-backend.html#operands-68", null ],
            [ "Results:", "smt-backend.html#results-64", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.repeat</span> (llzk::smt::RepeatOp)", "smt-backend.html#smtbvrepeat-llzksmtrepeatop", [
            [ "Operands:", "smt-backend.html#operands-69", null ],
            [ "Results:", "smt-backend.html#results-65", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.sdiv</span> (llzk::smt::BVSDivOp)", "smt-backend.html#smtbvsdiv-llzksmtbvsdivop", [
            [ "Operands:", "smt-backend.html#operands-70", null ],
            [ "Results:", "smt-backend.html#results-66", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.shl</span> (llzk::smt::BVShlOp)", "smt-backend.html#smtbvshl-llzksmtbvshlop", [
            [ "Operands:", "smt-backend.html#operands-71", null ],
            [ "Results:", "smt-backend.html#results-67", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.smod</span> (llzk::smt::BVSModOp)", "smt-backend.html#smtbvsmod-llzksmtbvsmodop", [
            [ "Operands:", "smt-backend.html#operands-72", null ],
            [ "Results:", "smt-backend.html#results-68", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.srem</span> (llzk::smt::BVSRemOp)", "smt-backend.html#smtbvsrem-llzksmtbvsremop", [
            [ "Operands:", "smt-backend.html#operands-73", null ],
            [ "Results:", "smt-backend.html#results-69", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.udiv</span> (llzk::smt::BVUDivOp)", "smt-backend.html#smtbvudiv-llzksmtbvudivop", [
            [ "Operands:", "smt-backend.html#operands-74", null ],
            [ "Results:", "smt-backend.html#results-70", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.urem</span> (llzk::smt::BVURemOp)", "smt-backend.html#smtbvurem-llzksmtbvuremop", [
            [ "Operands:", "smt-backend.html#operands-75", null ],
            [ "Results:", "smt-backend.html#results-71", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv.xor</span> (llzk::smt::BVXOrOp)", "smt-backend.html#smtbvxor-llzksmtbvxorop", [
            [ "Operands:", "smt-backend.html#operands-76", null ],
            [ "Results:", "smt-backend.html#results-72", null ]
          ] ],
          [ "<span class=\"tt\">smt.bv2int</span> (llzk::smt::BV2IntOp)", "smt-backend.html#smtbv2int-llzksmtbv2intop", [
            [ "Attributes:", "smt-backend.html#attributes-38", null ],
            [ "Operands:", "smt-backend.html#operands-77", null ],
            [ "Results:", "smt-backend.html#results-73", null ]
          ] ],
          [ "<span class=\"tt\">smt.check</span> (llzk::smt::CheckOp)", "smt-backend.html#smtcheck-llzksmtcheckop", [
            [ "Results:", "smt-backend.html#results-74", null ]
          ] ],
          [ "<span class=\"tt\">smt.constant</span> (llzk::smt::BoolConstantOp)", "smt-backend.html#smtconstant-llzksmtboolconstantop", [
            [ "Attributes:", "smt-backend.html#attributes-39", null ],
            [ "Results:", "smt-backend.html#results-75", null ]
          ] ],
          [ "<span class=\"tt\">smt.declare_fun</span> (llzk::smt::DeclareFunOp)", "smt-backend.html#smtdeclare_fun-llzksmtdeclarefunop", [
            [ "Attributes:", "smt-backend.html#attributes-40", null ],
            [ "Results:", "smt-backend.html#results-76", null ]
          ] ],
          [ "<span class=\"tt\">smt.distinct</span> (llzk::smt::DistinctOp)", "smt-backend.html#smtdistinct-llzksmtdistinctop", [
            [ "Operands:", "smt-backend.html#operands-78", null ],
            [ "Results:", "smt-backend.html#results-77", null ]
          ] ],
          [ "<span class=\"tt\">smt.eq</span> (llzk::smt::EqOp)", "smt-backend.html#smteq-llzksmteqop", [
            [ "Operands:", "smt-backend.html#operands-79", null ],
            [ "Results:", "smt-backend.html#results-78", null ]
          ] ],
          [ "<span class=\"tt\">smt.exists</span> (llzk::smt::ExistsOp)", "smt-backend.html#smtexists-llzksmtexistsop", [
            [ "Attributes:", "smt-backend.html#attributes-41", null ],
            [ "Results:", "smt-backend.html#results-79", null ]
          ] ],
          [ "<span class=\"tt\">smt.forall</span> (llzk::smt::ForallOp)", "smt-backend.html#smtforall-llzksmtforallop", [
            [ "Attributes:", "smt-backend.html#attributes-42", null ],
            [ "Results:", "smt-backend.html#results-80", null ]
          ] ],
          [ "<span class=\"tt\">smt.implies</span> (llzk::smt::ImpliesOp)", "smt-backend.html#smtimplies-llzksmtimpliesop", [
            [ "Operands:", "smt-backend.html#operands-80", null ],
            [ "Results:", "smt-backend.html#results-81", null ]
          ] ],
          [ "<span class=\"tt\">smt.int.abs</span> (llzk::smt::IntAbsOp)", "smt-backend.html#smtintabs-llzksmtintabsop", [
            [ "Operands:", "smt-backend.html#operands-81", null ],
            [ "Results:", "smt-backend.html#results-82", null ]
          ] ],
          [ "<span class=\"tt\">smt.int.add</span> (llzk::smt::IntAddOp)", "smt-backend.html#smtintadd-llzksmtintaddop", [
            [ "Operands:", "smt-backend.html#operands-82", null ],
            [ "Results:", "smt-backend.html#results-83", null ]
          ] ],
          [ "<span class=\"tt\">smt.int.cmp</span> (llzk::smt::IntCmpOp)", "smt-backend.html#smtintcmp-llzksmtintcmpop", [
            [ "Attributes:", "smt-backend.html#attributes-43", null ],
            [ "Operands:", "smt-backend.html#operands-83", null ],
            [ "Results:", "smt-backend.html#results-84", null ]
          ] ],
          [ "<span class=\"tt\">smt.int.constant</span> (llzk::smt::IntConstantOp)", "smt-backend.html#smtintconstant-llzksmtintconstantop", [
            [ "Attributes:", "smt-backend.html#attributes-44", null ],
            [ "Results:", "smt-backend.html#results-85", null ]
          ] ],
          [ "<span class=\"tt\">smt.int.div</span> (llzk::smt::IntDivOp)", "smt-backend.html#smtintdiv-llzksmtintdivop", [
            [ "Operands:", "smt-backend.html#operands-84", null ],
            [ "Results:", "smt-backend.html#results-86", null ]
          ] ],
          [ "<span class=\"tt\">smt.int.mod</span> (llzk::smt::IntModOp)", "smt-backend.html#smtintmod-llzksmtintmodop", [
            [ "Operands:", "smt-backend.html#operands-85", null ],
            [ "Results:", "smt-backend.html#results-87", null ]
          ] ],
          [ "<span class=\"tt\">smt.int.mul</span> (llzk::smt::IntMulOp)", "smt-backend.html#smtintmul-llzksmtintmulop", [
            [ "Operands:", "smt-backend.html#operands-86", null ],
            [ "Results:", "smt-backend.html#results-88", null ]
          ] ],
          [ "<span class=\"tt\">smt.int.neg</span> (llzk::smt::IntNegOp)", "smt-backend.html#smtintneg-llzksmtintnegop", [
            [ "Operands:", "smt-backend.html#operands-87", null ],
            [ "Results:", "smt-backend.html#results-89", null ]
          ] ],
          [ "<span class=\"tt\">smt.int.sub</span> (llzk::smt::IntSubOp)", "smt-backend.html#smtintsub-llzksmtintsubop", [
            [ "Operands:", "smt-backend.html#operands-88", null ],
            [ "Results:", "smt-backend.html#results-90", null ]
          ] ],
          [ "<span class=\"tt\">smt.int2bv</span> (llzk::smt::Int2BVOp)", "smt-backend.html#smtint2bv-llzksmtint2bvop", [
            [ "Operands:", "smt-backend.html#operands-89", null ],
            [ "Results:", "smt-backend.html#results-91", null ]
          ] ],
          [ "<span class=\"tt\">smt.ite</span> (llzk::smt::IteOp)", "smt-backend.html#smtite-llzksmtiteop", [
            [ "Operands:", "smt-backend.html#operands-90", null ],
            [ "Results:", "smt-backend.html#results-92", null ]
          ] ],
          [ "<span class=\"tt\">smt.not</span> (llzk::smt::NotOp)", "smt-backend.html#smtnot-llzksmtnotop", [
            [ "Operands:", "smt-backend.html#operands-91", null ],
            [ "Results:", "smt-backend.html#results-93", null ]
          ] ],
          [ "<span class=\"tt\">smt.or</span> (llzk::smt::OrOp)", "smt-backend.html#smtor-llzksmtorop", [
            [ "Operands:", "smt-backend.html#operands-92", null ],
            [ "Results:", "smt-backend.html#results-94", null ]
          ] ],
          [ "<span class=\"tt\">smt.pop</span> (llzk::smt::PopOp)", "smt-backend.html#smtpop-llzksmtpopop", [
            [ "Attributes:", "smt-backend.html#attributes-45", null ]
          ] ],
          [ "<span class=\"tt\">smt.push</span> (llzk::smt::PushOp)", "smt-backend.html#smtpush-llzksmtpushop", [
            [ "Attributes:", "smt-backend.html#attributes-46", null ]
          ] ],
          [ "<span class=\"tt\">smt.reset</span> (llzk::smt::ResetOp)", "smt-backend.html#smtreset-llzksmtresetop", null ],
          [ "<span class=\"tt\">smt.set_info</span> (llzk::smt::SetInfoOp)", "smt-backend.html#smtset_info-llzksmtsetinfoop", [
            [ "Attributes:", "smt-backend.html#attributes-47", null ]
          ] ],
          [ "<span class=\"tt\">smt.set_logic</span> (llzk::smt::SetLogicOp)", "smt-backend.html#smtset_logic-llzksmtsetlogicop", [
            [ "Attributes:", "smt-backend.html#attributes-48", null ]
          ] ],
          [ "<span class=\"tt\">smt.solver</span> (llzk::smt::SolverOp)", "smt-backend.html#smtsolver-llzksmtsolverop", [
            [ "Operands:", "smt-backend.html#operands-93", null ],
            [ "Results:", "smt-backend.html#results-95", null ]
          ] ],
          [ "<span class=\"tt\">smt.xor</span> (llzk::smt::XOrOp)", "smt-backend.html#smtxor-llzksmtxorop", [
            [ "Operands:", "smt-backend.html#operands-94", null ],
            [ "Results:", "smt-backend.html#results-96", null ]
          ] ],
          [ "<span class=\"tt\">smt.yield</span> (llzk::smt::YieldOp)", "smt-backend.html#smtyield-llzksmtyieldop", [
            [ "Operands:", "smt-backend.html#operands-95", null ]
          ] ]
        ] ],
        [ "Attributes", "smt-backend.html#attributes-49", [
          [ "BitVectorAttr", "smt-backend.html#bitvectorattr", [
            [ "Parameters:", "smt-backend.html#parameters-12", null ]
          ] ],
          [ "KeywordAttr", "smt-backend.html#keywordattr", [
            [ "Parameters:", "smt-backend.html#parameters-13", null ]
          ] ],
          [ "SymbolAttr", "smt-backend.html#symbolattr", [
            [ "Parameters:", "smt-backend.html#parameters-14", null ]
          ] ]
        ] ],
        [ "Types", "smt-backend.html#types-8", [
          [ "ArrayType", "smt-backend.html#arraytype-1", [
            [ "Parameters:", "smt-backend.html#parameters-15", null ]
          ] ],
          [ "BitVectorType", "smt-backend.html#bitvectortype", [
            [ "Parameters:", "smt-backend.html#parameters-16", null ]
          ] ],
          [ "BoolType", "smt-backend.html#booltype", null ],
          [ "IntType", "smt-backend.html#inttype", null ],
          [ "SMTFuncType", "smt-backend.html#smtfunctype", [
            [ "Parameters:", "smt-backend.html#parameters-17", null ]
          ] ],
          [ "SortType", "smt-backend.html#sorttype", [
            [ "Parameters:", "smt-backend.html#parameters-18", null ]
          ] ]
        ] ],
        [ "Enums", "smt-backend.html#enums-2", [
          [ "BVCmpPredicate", "smt-backend.html#bvcmppredicate", [
            [ "Cases:", "smt-backend.html#cases-2", null ]
          ] ],
          [ "IntPredicate", "smt-backend.html#intpredicate", [
            [ "Cases:", "smt-backend.html#cases-3", null ]
          ] ]
        ] ]
      ] ]
    ] ],
    [ "PCL Backend", "pcl-backend.html", [
      [ "'pcl' Dialect", "pcl-backend.html#pcl-dialect", [
        [ "Operations", "pcl-backend.html#operations-15", [
          [ "<span class=\"tt\">pcl.add</span> (pcl::AddOp)", "pcl-backend.html#pcladd-pcladdop", [
            [ "Operands:", "pcl-backend.html#operands-96", null ],
            [ "Results:", "pcl-backend.html#results-97", null ]
          ] ],
          [ "<span class=\"tt\">pcl.and</span> (pcl::AndOp)", "pcl-backend.html#pcland-pclandop", [
            [ "Operands:", "pcl-backend.html#operands-97", null ],
            [ "Results:", "pcl-backend.html#results-98", null ]
          ] ],
          [ "<span class=\"tt\">pcl.asfelt</span> (pcl::AsFeltOp)", "pcl-backend.html#pclasfelt-pclasfeltop", [
            [ "Operands:", "pcl-backend.html#operands-98", null ],
            [ "Results:", "pcl-backend.html#results-99", null ]
          ] ],
          [ "<span class=\"tt\">pcl.assert</span> (pcl::AssertOp)", "pcl-backend.html#pclassert-pclassertop", [
            [ "Operands:", "pcl-backend.html#operands-99", null ]
          ] ],
          [ "<span class=\"tt\">pcl.assume.deterministic</span> (pcl::AssumeDeterministicOp)", "pcl-backend.html#pclassumedeterministic-pclassumedeterministicop", [
            [ "Operands:", "pcl-backend.html#operands-100", null ]
          ] ],
          [ "<span class=\"tt\">pcl.const</span> (pcl::ConstOp)", "pcl-backend.html#pclconst-pclconstop", [
            [ "Attributes:", "pcl-backend.html#attributes-50", null ],
            [ "Results:", "pcl-backend.html#results-100", null ]
          ] ],
          [ "<span class=\"tt\">pcl.det</span> (pcl::DetOp)", "pcl-backend.html#pcldet-pcldetop", [
            [ "Operands:", "pcl-backend.html#operands-101", null ],
            [ "Results:", "pcl-backend.html#results-101", null ]
          ] ],
          [ "<span class=\"tt\">pcl.eq</span> (pcl::CmpEqOp)", "pcl-backend.html#pcleq-pclcmpeqop", [
            [ "Operands:", "pcl-backend.html#operands-102", null ],
            [ "Results:", "pcl-backend.html#results-102", null ]
          ] ],
          [ "<span class=\"tt\">pcl.false</span> (pcl::FalseOp)", "pcl-backend.html#pclfalse-pclfalseop", [
            [ "Results:", "pcl-backend.html#results-103", null ]
          ] ],
          [ "<span class=\"tt\">pcl.ge</span> (pcl::CmpGeOp)", "pcl-backend.html#pclge-pclcmpgeop", [
            [ "Operands:", "pcl-backend.html#operands-103", null ],
            [ "Results:", "pcl-backend.html#results-104", null ]
          ] ],
          [ "<span class=\"tt\">pcl.gt</span> (pcl::CmpGtOp)", "pcl-backend.html#pclgt-pclcmpgtop", [
            [ "Operands:", "pcl-backend.html#operands-104", null ],
            [ "Results:", "pcl-backend.html#results-105", null ]
          ] ],
          [ "<span class=\"tt\">pcl.iff</span> (pcl::IffOp)", "pcl-backend.html#pcliff-pcliffop", [
            [ "Operands:", "pcl-backend.html#operands-105", null ],
            [ "Results:", "pcl-backend.html#results-106", null ]
          ] ],
          [ "<span class=\"tt\">pcl.implies</span> (pcl::ImpliesOp)", "pcl-backend.html#pclimplies-pclimpliesop", [
            [ "Operands:", "pcl-backend.html#operands-106", null ],
            [ "Results:", "pcl-backend.html#results-107", null ]
          ] ],
          [ "<span class=\"tt\">pcl.le</span> (pcl::CmpLeOp)", "pcl-backend.html#pclle-pclcmpleop", [
            [ "Operands:", "pcl-backend.html#operands-107", null ],
            [ "Results:", "pcl-backend.html#results-108", null ]
          ] ],
          [ "<span class=\"tt\">pcl.lt</span> (pcl::CmpLtOp)", "pcl-backend.html#pcllt-pclcmpltop", [
            [ "Operands:", "pcl-backend.html#operands-108", null ],
            [ "Results:", "pcl-backend.html#results-109", null ]
          ] ],
          [ "<span class=\"tt\">pcl.mul</span> (pcl::MulOp)", "pcl-backend.html#pclmul-pclmulop", [
            [ "Operands:", "pcl-backend.html#operands-109", null ],
            [ "Results:", "pcl-backend.html#results-110", null ]
          ] ],
          [ "<span class=\"tt\">pcl.neg</span> (pcl::NegOp)", "pcl-backend.html#pclneg-pclnegop", [
            [ "Operands:", "pcl-backend.html#operands-110", null ],
            [ "Results:", "pcl-backend.html#results-111", null ]
          ] ],
          [ "<span class=\"tt\">pcl.not</span> (pcl::NotOp)", "pcl-backend.html#pclnot-pclnotop", [
            [ "Operands:", "pcl-backend.html#operands-111", null ],
            [ "Results:", "pcl-backend.html#results-112", null ]
          ] ],
          [ "<span class=\"tt\">pcl.or</span> (pcl::OrOp)", "pcl-backend.html#pclor-pclorop", [
            [ "Operands:", "pcl-backend.html#operands-112", null ],
            [ "Results:", "pcl-backend.html#results-113", null ]
          ] ],
          [ "<span class=\"tt\">pcl.post_cond</span> (pcl::PostOp)", "pcl-backend.html#pclpost_cond-pclpostop", [
            [ "Operands:", "pcl-backend.html#operands-113", null ]
          ] ],
          [ "<span class=\"tt\">pcl.sub</span> (pcl::SubOp)", "pcl-backend.html#pclsub-pclsubop", [
            [ "Operands:", "pcl-backend.html#operands-114", null ],
            [ "Results:", "pcl-backend.html#results-114", null ]
          ] ],
          [ "<span class=\"tt\">pcl.true</span> (pcl::TrueOp)", "pcl-backend.html#pcltrue-pcltrueop", [
            [ "Results:", "pcl-backend.html#results-115", null ]
          ] ],
          [ "<span class=\"tt\">pcl.var</span> (pcl::VarOp)", "pcl-backend.html#pclvar-pclvarop", [
            [ "Attributes:", "pcl-backend.html#attributes-51", null ],
            [ "Results:", "pcl-backend.html#results-116", null ]
          ] ]
        ] ],
        [ "Attributes", "pcl-backend.html#attributes-52", [
          [ "FeltAttr", "pcl-backend.html#feltattr-1", [
            [ "Parameters:", "pcl-backend.html#parameters-19", null ]
          ] ],
          [ "BoolAttr", "pcl-backend.html#boolattr", [
            [ "Parameters:", "pcl-backend.html#parameters-20", null ]
          ] ],
          [ "PrimeAttr", "pcl-backend.html#primeattr", [
            [ "Parameters:", "pcl-backend.html#parameters-21", null ]
          ] ]
        ] ],
        [ "Types", "pcl-backend.html#types-9", [
          [ "BoolType", "pcl-backend.html#booltype-1", null ],
          [ "FeltType", "pcl-backend.html#felttype-1", null ]
        ] ]
      ] ]
    ] ],
    [ "ZKLean Backend", "zklean-backend.html", [
      [ "'ZKExpr' Dialect", "zklean-backend.html#zkexpr-dialect", [
        [ "Operations", "zklean-backend.html#operations-16", [
          [ "<span class=\"tt\">ZKExpr.Add</span> (llzk::zkexpr::AddOp)", "zklean-backend.html#zkexpradd-llzkzkexpraddop", [
            [ "Operands:", "zklean-backend.html#operands-115", null ],
            [ "Results:", "zklean-backend.html#results-117", null ]
          ] ],
          [ "<span class=\"tt\">ZKExpr.Literal</span> (llzk::zkexpr::LiteralOp)", "zklean-backend.html#zkexprliteral-llzkzkexprliteralop", [
            [ "Operands:", "zklean-backend.html#operands-116", null ],
            [ "Results:", "zklean-backend.html#results-118", null ]
          ] ],
          [ "<span class=\"tt\">ZKExpr.Mul</span> (llzk::zkexpr::MulOp)", "zklean-backend.html#zkexprmul-llzkzkexprmulop", [
            [ "Operands:", "zklean-backend.html#operands-117", null ],
            [ "Results:", "zklean-backend.html#results-119", null ]
          ] ],
          [ "<span class=\"tt\">ZKExpr.Neg</span> (llzk::zkexpr::NegOp)", "zklean-backend.html#zkexprneg-llzkzkexprnegop", [
            [ "Operands:", "zklean-backend.html#operands-118", null ],
            [ "Results:", "zklean-backend.html#results-120", null ]
          ] ],
          [ "<span class=\"tt\">ZKExpr.Sub</span> (llzk::zkexpr::SubOp)", "zklean-backend.html#zkexprsub-llzkzkexprsubop", [
            [ "Operands:", "zklean-backend.html#operands-119", null ],
            [ "Results:", "zklean-backend.html#results-121", null ]
          ] ]
        ] ],
        [ "Types", "zklean-backend.html#types-10", [
          [ "ComposedLookupTableType", "zklean-backend.html#composedlookuptabletype", [
            [ "Parameters:", "zklean-backend.html#parameters-22", null ]
          ] ],
          [ "WitnessIDType", "zklean-backend.html#witnessidtype", null ],
          [ "ZKExprType", "zklean-backend.html#zkexprtype", null ]
        ] ]
      ] ],
      [ "'ZKBuilder' Dialect", "zklean-backend.html#zkbuilder-dialect", [
        [ "Operations", "zklean-backend.html#operations-17", [
          [ "<span class=\"tt\">ZKBuilder.AllocWitness</span> (llzk::zkbuilder::AllocWitnessOp)", "zklean-backend.html#zkbuilderallocwitness-llzkzkbuilderallocwitnessop", [
            [ "Results:", "zklean-backend.html#results-122", null ]
          ] ],
          [ "<span class=\"tt\">ZKBuilder.ConstrainEq</span> (llzk::zkbuilder::ConstrainEqOp)", "zklean-backend.html#zkbuilderconstraineq-llzkzkbuilderconstraineqop", [
            [ "Operands:", "zklean-backend.html#operands-120", null ],
            [ "Results:", "zklean-backend.html#results-123", null ]
          ] ],
          [ "<span class=\"tt\">ZKBuilder.ConstrainR1CS</span> (llzk::zkbuilder::ConstrainR1CSOp)", "zklean-backend.html#zkbuilderconstrainr1cs-llzkzkbuilderconstrainr1csop", [
            [ "Operands:", "zklean-backend.html#operands-121", null ],
            [ "Results:", "zklean-backend.html#results-124", null ]
          ] ]
        ] ],
        [ "Types", "zklean-backend.html#types-11", [
          [ "ZKBuilderStateType", "zklean-backend.html#zkbuilderstatetype", null ]
        ] ]
      ] ],
      [ "'ZKLeanLean' Dialect", "zklean-backend.html#zkleanlean-dialect", [
        [ "Operations", "zklean-backend.html#operations-18", [
          [ "<span class=\"tt\">ZKLeanLean.accessor</span> (llzk::zkleanlean::AccessorOp)", "zklean-backend.html#zkleanleanaccessor-llzkzkleanleanaccessorop", [
            [ "Attributes:", "zklean-backend.html#attributes-53", null ],
            [ "Operands:", "zklean-backend.html#operands-122", null ],
            [ "Results:", "zklean-backend.html#results-125", null ]
          ] ],
          [ "<span class=\"tt\">ZKLeanLean.call</span> (llzk::zkleanlean::CallOp)", "zklean-backend.html#zkleanleancall-llzkzkleanleancallop", [
            [ "Attributes:", "zklean-backend.html#attributes-54", null ],
            [ "Operands:", "zklean-backend.html#operands-123", null ],
            [ "Results:", "zklean-backend.html#results-126", null ]
          ] ],
          [ "<span class=\"tt\">ZKLeanLean.member</span> (llzk::zkleanlean::MemberDefOp)", "zklean-backend.html#zkleanleanmember-llzkzkleanleanmemberdefop", [
            [ "Attributes:", "zklean-backend.html#attributes-55", null ]
          ] ],
          [ "<span class=\"tt\">ZKLeanLean.structure</span> (llzk::zkleanlean::StructDefOp)", "zklean-backend.html#zkleanleanstructure-llzkzkleanleanstructdefop", [
            [ "Attributes:", "zklean-backend.html#attributes-56", null ]
          ] ]
        ] ],
        [ "Types", "zklean-backend.html#types-12", [
          [ "StructType", "zklean-backend.html#structtype-1", [
            [ "Parameters:", "zklean-backend.html#parameters-23", null ]
          ] ]
        ] ]
      ] ]
    ] ]
];