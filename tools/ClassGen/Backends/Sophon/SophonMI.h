/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// BM General
BB.newBackendSpecificInstr("SophonMIGDMAGlobalToLocal")
    .addOperand("Local", OperandKind::Out)
    .addOperand("Global", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "ShapeNCHW")
    .addMember(MemberType::VectorUnsigned, "GlobalStrideNCH")
    .addMember(MemberType::Boolean, "IsGlobalWeightSpace")
    .addMember(MemberType::Boolean, "IsLocalAligned")
    .autoVerify(VerifyKind::SameElementType, {"Local", "Global"});

BB.newBackendSpecificInstr("SophonMIGDMALocalToGlobal")
    .addOperand("Global", OperandKind::Out)
    .addOperand("Local", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "ShapeNCHW")
    .addMember(MemberType::VectorUnsigned, "GlobalStrideNCH")
    .addMember(MemberType::Boolean, "IsGlobalWeightSpace")
    .addMember(MemberType::Boolean, "IsLocalAligned")
    .autoVerify(VerifyKind::SameElementType, {"Local", "Global"});

// BM INT8 INST
BB.newBackendSpecificInstr("SophonMIReluQ8")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonMIMacConstQ8")
    .addOperand("DestLow", OperandKind::Out)
    .addOperand("DestHigh", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .addMember(MemberType::Unsigned, "LShiftWidth")
    .addMember(MemberType::Unsigned, "Multiplier")
    .addMember(MemberType::Unsigned, "IsMultiplierSigned")
    .addMember(MemberType::Boolean, "IsResultI8")
    .autoVerify(VerifyKind::SameElementType, {"DestLow", "DestHigh", "Src"});

BB.newBackendSpecificInstr("SophonMIMulConstQ8")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Multiplier")
    .addMember(MemberType::Unsigned, "IsMultiplierSigned")
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonMIMulConstQ16")
    .addOperand("DestLow", OperandKind::Out)
    .addOperand("DestHigh", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Multiplier")
    .addMember(MemberType::Unsigned, "IsMultiplierSigned")
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .autoVerify(VerifyKind::SameElementType, {"DestLow", "DestHigh", "Src"});

BB.newBackendSpecificInstr("SophonMIAvgPoolingQ8")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "KernelHW")
    .addMember(MemberType::VectorUnsigned, "StrideHW")
    .addMember(MemberType::VectorUnsigned, "PadTLBR")
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonMIMaxPoolingQ8")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "KernelHW")
    .addMember(MemberType::VectorUnsigned, "StrideHW")
    .addMember(MemberType::VectorUnsigned, "PadTLBR")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonMIConvolutionQ8")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "StrideHW")
    .addMember(MemberType::VectorUnsigned, "PadTLBR")
    .addMember(MemberType::VectorUnsigned, "DilationHW")
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .addMember(MemberType::Boolean, "EnableRelu")
    .addMember(MemberType::Unsigned, "StreamID")
    .addMember(MemberType::Unsigned, "InstID")
    .addMember(MemberType::VectorUnsigned, "Depends")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter", "Bias"});

BB.newBackendSpecificInstr("SophonMIDepthwiseConvolutionQ8")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "StrideHW")
    .addMember(MemberType::VectorUnsigned, "PadTLBR")
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .addMember(MemberType::Unsigned, "StreamID")
    .addMember(MemberType::Unsigned, "InstID")
    .addMember(MemberType::VectorUnsigned, "Depends")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter", "Bias"});

BB.newBackendSpecificInstr("SophonMIFCQ8")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .addMember(MemberType::Unsigned, "LShiftWidth")
    .addMember(MemberType::Boolean, "ResultAdd")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter", "Bias"});

BB.newBackendSpecificInstr("SophonMIFCQ16")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .addMember(MemberType::Unsigned, "LShiftWidth")
    .addMember(MemberType::Boolean, "ResultAdd")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter", "Bias"});
