
BB.newBackendSpecificInstr("SophonLoad")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In);

BB.newBackendSpecificInstr("SophonStore")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .setType("Src->getType()")
    .dataParallel();
