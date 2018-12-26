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

/// BM1682
/// TG in old bmnet

//===--------------------------------------------------------------------===//
//                   Convolution / Pool / FC
//===--------------------------------------------------------------------===//
BB.newBackendSpecificInstr("SophonConvolution")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::Unsigned, "Group")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter", "Bias"});

BB.newBackendSpecificInstr("SophonConvolutionWithoutBias")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::Unsigned, "Group")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter"});

BB.newBackendSpecificInstr("SophonConvolutionQ8")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "StrideHW")
    .addMember(MemberType::VectorUnsigned, "PadTLBR")
    .addMember(MemberType::VectorUnsigned, "DilationHW")
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .addMember(MemberType::Boolean, "EnableRelu")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter"});

BB.newBackendSpecificInstr("SophonAvgPool")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonAvgPoolQ8")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "KernelHW")
    .addMember(MemberType::VectorUnsigned, "StrideHW")
    .addMember(MemberType::VectorUnsigned, "PadTLBR")
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .addMember(MemberType::Unsigned, "Multiplier")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonMaxPool")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonMaxPoolQ8")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "KernelHW")
    .addMember(MemberType::VectorUnsigned, "StrideHW")
    .addMember(MemberType::VectorUnsigned, "PadTLBR")
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .addMember(MemberType::Unsigned, "Multiplier")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

#if 1 // use SophonMatMul
BB.newBackendSpecificInstr("SophonFullyConnected")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Weights", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::Boolean, "Relu")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType,
                {"Dest", "Src", "Weights", "Bias"});
#endif

BB.newBackendSpecificInstr("SophonFullyConnectedQ8")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Weights", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::Boolean, "Relu")
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .addMember(MemberType::Unsigned, "LShiftWidth")
    .addMember(MemberType::Boolean, "ResultAdd")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Weights"});

//===--------------------------------------------------------------------===//
//                     Normalization
//===--------------------------------------------------------------------===//
BB.newBackendSpecificInstr("SophonNormalize")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    //    .addInput("Scale") //!< delete
    .addMember(MemberType::Boolean, "AcrossSpatial")
    .addMember(MemberType::Boolean, "ChannelShared")
    .addMember(MemberType::Float, "Epsilon")
    .addMember(MemberType::Float, "Scale") //!< add this
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonBatchNormalization")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    //    .addInput("Bias")   //!< delete
    .addOperand("Mean", OperandKind::In)
    .addOperand("Variance", OperandKind::In)
    .addMember(MemberType::Float, "Scale")
    .addMember(MemberType::Float, "Epsilon")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType,
                {"Dest", "Src", "Mean", "Variance"});

BB.newBackendSpecificInstr("SophonBatchNormalizationOpt")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Mean", OperandKind::In)
    .addOperand("Variance", OperandKind::In)
    .addMember(MemberType::Float, "Scale")
    .addMember(MemberType::Float, "Epsilon")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType,
                {"Dest", "Src", "Mean", "Variance"});

BB.newBackendSpecificInstr("SophonLocalResponseNormalization")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Float, "Alpha")
    .addMember(MemberType::Float, "Beta")
    //.addMember(MemberType::Float, "bias")
    .addMember(MemberType::Unsigned, "NormRegion")
    .addMember(MemberType::Unsigned, "Size")
    .addMember(MemberType::Float, "K")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

//===--------------------------------------------------------------------===//
//                     Activation
//===--------------------------------------------------------------------===//
BB.newBackendSpecificInstr("SophonRelu")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Float, "NegativeSlope")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonReluQ8")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonSigmoid")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonTanh")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonPrelu")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Slope", OperandKind::In)
    .addMember(MemberType::Boolean, "ChannelShared")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Slope"});

//===--------------------------------------------------------------------===//
//                     Other NN operations
//===--------------------------------------------------------------------===//

BB.newBackendSpecificInstr("SophonSoftMax")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Axis")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonPriorbox")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Weight", OperandKind::In)
    //    .addMember(MemberType::VectorFloat, "MinSize")
    //    .addMember(MemberType::VectorFloat, "MaxSize")
    //    .addMember(MemberType::VectorFloat, "AspectRatio")
    //    .addMember(MemberType::VectorFloat, "Variance")
    .addMember(MemberType::Unsigned, "NumPriors")
    .addMember(MemberType::Unsigned, "ImgH")
    .addMember(MemberType::Unsigned, "ImgW")
    .addMember(MemberType::Float, "StepH")
    .addMember(MemberType::Float, "StepW")
    .addMember(MemberType::Unsigned, "Clip")
    .addMember(MemberType::Unsigned, "Offset")
    .addMember(MemberType::Unsigned, "ReducebBoxes")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Weight"});

BB.newBackendSpecificInstr("SophonUpsample")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Size")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonDeconvolution")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::Unsigned, "Group")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter", "Bias"});

BB.newBackendSpecificInstr("SophonDeconvolutionWithoutBias")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::Unsigned, "Group")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter"});

BB.newBackendSpecificInstr("SophonDeconvolutionOpt")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::Unsigned, "Group")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter", "Bias"});

BB.newBackendSpecificInstr("SophonDeconvolutionWithoutBiasOpt")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::Unsigned, "Group")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter"});

BB.newBackendSpecificInstr("SophonROIPool")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Rois", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "PoolShape")
    .addMember(MemberType::Float, "SpatialScale")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Rois"});

BB.newBackendSpecificInstr("SophonPSROIPool")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Float, "SpatialScale")
    .addMember(MemberType::Unsigned, "Group")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonMultiRegion")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Classes")
    .addMember(MemberType::Unsigned, "Coords")
    .addMember(MemberType::Unsigned, "Nums")
    .addMember(MemberType::VectorUnsigned, "ActivateParameters")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonLSTM")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("W", OperandKind::In)
    .addOperand("R", OperandKind::In)
    .addOperand("B", OperandKind::In)
    .addOperand("P", OperandKind::In)
    .addMember(MemberType::Unsigned, "time_num")
    .addMember(MemberType::Unsigned, "with_x_static")
    .addMember(MemberType::Unsigned, "expose_hidden")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType,
                {"Dest", "Src", "W", "R", "B", "P"});

BB.newBackendSpecificInstr("SophonShuffleChannel")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Group")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

//===--------------------------------------------------------------------===//
//                     Shape transformations
//===--------------------------------------------------------------------===//
//
BB.newBackendSpecificInstr("SophonConcat")
    .addOperand("Dest", OperandKind::Out)
    //.addMember(MemberType::VectorNodeValue, "Inputs") //delete this, use LHS
    // and RHS if concat number>3, use concat 2 times
    .addMember(MemberType::Unsigned, "Dim")
    //.autoIRGen()
    .autoVerify(VerifyKind::NoVerify);

// FIXME
BB.newBackendSpecificInstr("SophonConcat2")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Input0", OperandKind::In)
    .addOperand("Input1", OperandKind::In)
    .addMember(MemberType::Unsigned, "Dim")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Input0", "Input1"});

BB.newBackendSpecificInstr("SophonConcat3")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Input0", OperandKind::In)
    .addOperand("Input1", OperandKind::In)
    .addOperand("Input2", OperandKind::In)
    .addMember(MemberType::Unsigned, "Dim")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType,
                {"Dest", "Input0", "Input1", "Input2"});

BB.newBackendSpecificInstr("SophonConcat4")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Input0", OperandKind::In)
    .addOperand("Input1", OperandKind::In)
    .addOperand("Input2", OperandKind::In)
    .addOperand("Input3", OperandKind::In)
    .addMember(MemberType::Unsigned, "Dim")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType,
                {"Dest", "Input0", "Input1", "Input2", "Input3"});

BB.newBackendSpecificInstr("SophonConcat5")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Input0", OperandKind::In)
    .addOperand("Input1", OperandKind::In)
    .addOperand("Input2", OperandKind::In)
    .addOperand("Input3", OperandKind::In)
    .addOperand("Input4", OperandKind::In)
    .addMember(MemberType::Unsigned, "Dim")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType,
                {"Dest", "Input0", "Input1", "Input2", "Input3", "Input4"});

BB.newBackendSpecificInstr("SophonConcat6")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Input0", OperandKind::In)
    .addOperand("Input1", OperandKind::In)
    .addOperand("Input2", OperandKind::In)
    .addOperand("Input3", OperandKind::In)
    .addOperand("Input4", OperandKind::In)
    .addOperand("Input5", OperandKind::In)
    .addMember(MemberType::Unsigned, "Dim")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType,
                {"Dest", "Input0", "Input1", "Input2", "Input3", "Input4",
                 "Input5"});

BB.newBackendSpecificInstr("SophonReshape")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorSizeT, "Dims")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonTranspose")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Shuffle")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonFlatten")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonReorg")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Stride")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonCrop")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Offsets")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonPermute")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Order")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonDummyData")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonEltwise")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    //.addMember(MemberType::VectorFloat, "Coeff")
    .addMember(MemberType::Unsigned, "Operation")
    .addMember(MemberType::Boolean, "StableProdGrad")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS"});

BB.newBackendSpecificInstr("SophonTile")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    //    .addMember(MemberType::VectorFloat, "coeff")
    .addMember(MemberType::Unsigned, "Axis")
    .addMember(MemberType::Unsigned, "Tiles")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

//===--------------------------------------------------------------------===//
//                     Arithmetic
//===--------------------------------------------------------------------===//

BB.newBackendSpecificInstr("SophonMatMul")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .addOperand("Slice", OperandKind::In)
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS", "Slice"});

BB.newBackendSpecificInstr("SophonScale")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Scale", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::Unsigned, "Axis")
    .addMember(MemberType::Unsigned, "NumAxes")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Scale", "Bias"});

BB.newBackendSpecificInstr("SophonScaleWithoutBias")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Scale", OperandKind::In)
    .addMember(MemberType::Unsigned, "Axis")
    .addMember(MemberType::Unsigned, "NumAxes")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Scale"});

BB.newBackendSpecificInstr("SophonScale1")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::Unsigned, "Axis")
    .addMember(MemberType::Unsigned, "NumAxes")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Bias"});

BB.newBackendSpecificInstr("SophonMul")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS"});

BB.newBackendSpecificInstr("SophonAdd")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS"});

BB.newBackendSpecificInstr("SophonMax")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS"});

BB.newBackendSpecificInstr("SophonPow")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Float, "Power")
    .addMember(MemberType::Float, "Scale")
    .addMember(MemberType::Float, "Shift")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonSub")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS"});

BB.newBackendSpecificInstr("SophonDiv")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS"});

//===--------------------------------------------------------------------===//
//                     Others
//===--------------------------------------------------------------------===//
BB.newBackendSpecificInstr("SophonAbs")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonYolo")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(
        MemberType::Unsigned,
        "classes") // by ycs: in bmnet-caffe.proto, it's int32? not usigned?
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonRegion")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(
        MemberType::Unsigned,
        "classes") // by ycs: in bmnet-caffe.proto, it's int32? not usigned?
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonProposal")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "feat_stride")
    .addMember(MemberType::Unsigned, "pre_nms_topN")
    .addMember(MemberType::Unsigned, "post_nms_topN")
    .addMember(MemberType::Float, "nms_thresh")
    .addMember(MemberType::Unsigned, "min_size")
    .addMember(MemberType::Unsigned, "base_size")
    .addMember(MemberType::Unsigned, "version")
    //    .addMember(MemberType::VectorFloat, "scale") by ycs: cann't use
    //    vectorfloat yet, need fix .addMember(MemberType::VectorFloat, "ratio")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

/******************
BB.newBackendSpecificInstr("SophonSlice")
    .addOperand("Dest", OperandKind::Out)
    //.addOperand("Dest1", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Axis")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Dest1","Src"});
******************/

BB.newBackendSpecificInstr("SophonReduction")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Axis")
    .addMember(MemberType::Unsigned, "Operation")
    .addMember(MemberType::Float, "Coeff")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("SophonInterp")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Weights", OperandKind::In)
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Weights"});

/// TL for old bmnet
BB.newBackendSpecificInstr("SophonTLActivation")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Input")
    .addMember(MemberType::Unsigned, "Output")
    .addMember(MemberType::Unsigned, "Activation")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("SophonTLPrelu")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Input")
    .addMember(MemberType::Unsigned, "Output")
    .addMember(MemberType::Unsigned, "Weight")
    .addMember(MemberType::Boolean, "ChannelShared")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("SophonTLConvolution")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Input")
    .addMember(MemberType::Unsigned, "Output")
    .addMember(MemberType::Unsigned, "Weight")
    .addMember(MemberType::Unsigned, "Bias")
    .addMember(MemberType::Unsigned, "Working")
    .addMember(MemberType::Unsigned, "Group")
    .addMember(MemberType::Boolean, "DoBias")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::Boolean, "ResultAdd")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("SophonTLConvolutionWithoutBias")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Input")
    .addMember(MemberType::Unsigned, "Output")
    .addMember(MemberType::Unsigned, "Weight")
    .addMember(MemberType::Unsigned, "Working")
    .addMember(MemberType::Unsigned, "Group")
    .addMember(MemberType::Boolean, "DoBias")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::Boolean, "ResultAdd")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("SophonTLMaxPooling")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Input")
    .addMember(MemberType::Unsigned, "Output")
    .addMember(MemberType::Unsigned, "Working")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("SophonTLAveragePooling")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Input")
    .addMember(MemberType::Unsigned, "Output")
    .addMember(MemberType::Unsigned, "Working")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("SophonTLTransportLoad")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Laddr")
    .addMember(MemberType::Boolean, "Transpose")
    .addMember(MemberType::Boolean, "Aligned")
    .addMember(MemberType::Boolean, "IsWeight")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("SophonTLTransportStore")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Laddr")
    .addMember(MemberType::Boolean, "Transpose")
    .addMember(MemberType::Boolean, "Aligned")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("SophonTLUpsample")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Input")
    .addMember(MemberType::Unsigned, "Output")
    .addMember(MemberType::Unsigned, "Size")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("SophonTLLrn")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Input")
    .addMember(MemberType::Unsigned, "Output")
    .addMember(MemberType::Unsigned, "Working")
    .addMember(MemberType::Unsigned, "Local_Size")
    .addMember(MemberType::Float, "Alpha")
    .addMember(MemberType::Float, "Beta")
    .addMember(MemberType::Float, "K")
    .addMember(MemberType::Unsigned, "Norm_Region")
    .addMember(MemberType::Unsigned, "sqr_lut_weight")
    .addMember(MemberType::Unsigned, "power_lut_weight")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("SophonTLScale")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Input")
    .addMember(MemberType::Unsigned, "Output")
    .addMember(MemberType::Unsigned, "Scale")
    .addMember(MemberType::Unsigned, "Bias")
    .addMember(MemberType::Unsigned, "Scale_Dim")
    .addMember(MemberType::Boolean, "Bias_Term")
    .addMember(MemberType::Boolean, "If_Relu")
    .addMember(MemberType::Float, "Relu_Slope")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("SophonTLEltwise")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Input")
    .addMember(MemberType::Unsigned, "Output")
    .addMember(MemberType::Unsigned, "Working")
    .addMember(MemberType::Unsigned, "Op_Code")
    .addMember(MemberType::VectorFloat, "Coeff")
    .addMember(MemberType::Boolean, "If_Relu")
    .addMember(MemberType::Float, "Relu_Slope")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("SophonTLBatchNorm")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Input")
    .addMember(MemberType::Unsigned, "Output")
    .addMember(MemberType::Unsigned, "Mean")
    .addMember(MemberType::Unsigned, "Variance")
    .addMember(MemberType::Unsigned, "Scale_ma")
    .addMember(MemberType::Float, "Eps")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("SophonTLShuffleChannel")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Input")
    .addMember(MemberType::Unsigned, "Output")
    .addMember(MemberType::Unsigned, "Group")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("SophonTLResizeBilinear")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Input")
    .addMember(MemberType::Unsigned, "Output")
    .addMember(MemberType::Unsigned, "Weight")
    .addMember(MemberType::Unsigned, "Resize_h")
    .addMember(MemberType::Unsigned, "Resize_w")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("ParallelEnable").autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("ParallelDisable").autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Alloc_Const")
    .addMember(MemberType::Float, "fvalue")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Alloc")
    .addMember(MemberType::TypeRef, "Ty")
    .addMember(MemberType::Unsigned, "Ctrls")
    .setType("Ty")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Alloc_Bank")
    .addMember(MemberType::Unsigned, "BankId")
    .addMember(MemberType::TypeRef, "Ty")
    .addMember(MemberType::Unsigned, "Ctrls")
    .setType("Ty")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Prealloc")
    .addMember(MemberType::Unsigned, "LAddress")
    .addMember(MemberType::TypeRef, "Ty")
    .setType("Ty")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Prealloc_Align")
    .addMember(MemberType::Unsigned, "LAddress")
    .addMember(MemberType::TypeRef, "Ty")
    .setType("Ty")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Free")
    .addOperand("Src", OperandKind::In)
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Load_Stride")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorSizeT, "Strides")
    .addMember(MemberType::Unsigned, "Ctrls")
    .addMember(MemberType::Unsigned, "GAddress") // FIXME
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Store_Stride")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorSizeT, "Strides")
    .addMember(MemberType::Unsigned, "Ctrls")
    .addMember(MemberType::Unsigned, "GAddress") // FIXME
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Load")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Ctrls")
    .addMember(MemberType::Unsigned, "GAddress") // FIXME
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Store")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Ctrls")
    .addMember(MemberType::Unsigned, "GAddress") // FIXME
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Slice")
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "N_Start")
    .addMember(MemberType::Unsigned, "N_End")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_CW_Transpose")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Copy_Gdma")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Copy_Stride")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorSizeT, "D_Strides")
    .addMember(MemberType::VectorSizeT, "S_Strides")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("G_Copy")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "D_GAddr")
    .addMember(MemberType::Unsigned, "S_GAddr")
    .addMember(MemberType::VectorUnsigned, "Shape")
    .addMember(MemberType::Unsigned, "Ctrls")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("G_Copy_Stride")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "D_GAddr")
    .addMember(MemberType::Unsigned, "S_GAddr")
    .addMember(MemberType::VectorUnsigned, "D_Shape")
    .addMember(MemberType::VectorSizeT, "D_Strides")
    .addMember(MemberType::VectorUnsigned, "S_Shape")
    .addMember(MemberType::VectorSizeT, "S_Strides")
    .addMember(MemberType::Unsigned, "Ctrls")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("G_Copy_Stride_Transpose")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "D_GAddr")
    .addMember(MemberType::Unsigned, "S_GAddr")
    .addMember(MemberType::VectorUnsigned, "D_Shape")
    .addMember(MemberType::VectorSizeT, "D_Strides")
    .addMember(MemberType::VectorUnsigned, "S_Shape")
    .addMember(MemberType::VectorSizeT, "S_Strides")
    .addMember(MemberType::Boolean, "Transpose")
    .addMember(MemberType::Unsigned, "Ctrls")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Conv")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Inserts")
    .addMember(MemberType::VectorUnsigned, "InsertsLast")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorSizeT, "KStrides")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::Boolean, "KernelFlip")
    .addMember(MemberType::Boolean, "ResultAdd")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter", "Bias"});

BB.newBackendSpecificInstr("TL_Conv_Without_Bias")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Inserts")
    .addMember(MemberType::VectorUnsigned, "InsertsLast")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorSizeT, "KStrides")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::Boolean, "KernelFlip")
    .addMember(MemberType::Boolean, "ResultAdd")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter"});

BB.newBackendSpecificInstr("TL_MaxPool")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Inserts")
    .addMember(MemberType::VectorUnsigned, "InsertsLast")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("TL_AvgPool")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Inserts")
    .addMember(MemberType::VectorUnsigned, "InsertsLast")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::Float, "AvgConst")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("TL_MaxPool_Bwd")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Index", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Inserts")
    .addMember(MemberType::VectorUnsigned, "InsertsLast")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("TL_AvgPool_Bwd")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Inserts")
    .addMember(MemberType::VectorUnsigned, "InsertsLast")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::Float, "AvgConst")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("TL_Matrix_Mac")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .addMember(MemberType::Unsigned, "Ctrls")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS"});

BB.newBackendSpecificInstr("TL_Matrix_Mac_Bias")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::Unsigned, "Ctrls")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS", "Bias"});

BB.newBackendSpecificInstr("TL_Mac")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .addMember(MemberType::Unsigned, "Ctrls")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Mac_Bias")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::Unsigned, "Ctrls")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Max")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .autoVerify(VerifyKind::NoVerify);

// FIXME
BB.newBackendSpecificInstr("TL_Max1")
    .addOperand("LHS", OperandKind::InOut) // Result is same with LHS
    .addOperand("RHS", OperandKind::In)
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Cmp")
    .addOperand("R_AB", OperandKind::Out)
    .addOperand("R_CD", OperandKind::Out)
    .addOperand("SRC_A", OperandKind::In)
    .addOperand("SRC_B", OperandKind::In)
    .addOperand("SRC_C", OperandKind::In)
    .addOperand("SRC_D", OperandKind::In)
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Mul")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .addMember(MemberType::Unsigned, "Ctrls")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Mul_Stride")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .addMember(MemberType::VectorSizeT, "D_Strides")
    .addMember(MemberType::VectorSizeT, "L_Strides")
    .addMember(MemberType::VectorSizeT, "R_Strides")
    .addMember(MemberType::Unsigned, "Ctrls")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS"});

BB.newBackendSpecificInstr("TL_Reshape")
    .addOperand("Src", OperandKind::InOut)
    .addMember(MemberType::VectorSizeT, "Shape")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Lrn_Shift")
    .addOperand("Src", OperandKind::In)
    .addOperand("Dest", OperandKind::Out)
    .addMember(MemberType::Boolean, "Right_shift")
    .addMember(MemberType::Unsigned, "Lrn_step")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Div")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .addMember(MemberType::Unsigned, "Ctrls")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Div_Stride")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .addMember(MemberType::VectorSizeT, "D_Strides")
    .addMember(MemberType::VectorSizeT, "L_Strides")
    .addMember(MemberType::VectorSizeT, "R_Strides")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Add")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .addMember(MemberType::Unsigned, "Ctrls")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS"});

BB.newBackendSpecificInstr("TL_Add_Stride")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .addMember(MemberType::VectorSizeT, "D_Strides")
    .addMember(MemberType::VectorSizeT, "L_Strides")
    .addMember(MemberType::VectorSizeT, "R_Strides")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS"});

BB.newBackendSpecificInstr("TL_Sub")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .addMember(MemberType::Unsigned, "Ctrls")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS"});

BB.newBackendSpecificInstr("TL_Sub_Stride")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .addMember(MemberType::VectorSizeT, "D_Strides")
    .addMember(MemberType::VectorSizeT, "L_Strides")
    .addMember(MemberType::VectorSizeT, "R_Strides")
    .autoVerify(VerifyKind::SameElementType, {"Dest", "LHS", "RHS"});

BB.newBackendSpecificInstr("TL_Sum")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Ctrls")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Img_Sum")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Unsigned, "Ctrls")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Xa")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Float, "Const_a")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Ex")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Lnx")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Rsq")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Xn")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::Float, "Const_n") // This argument should be int.
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Arithmetic")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("LHS", OperandKind::In)
    .addOperand("RHS", OperandKind::In)
    .addMember(MemberType::Unsigned, "Op")
    .addMember(MemberType::Unsigned, "Ctrls")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Copy")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Fill")
    .addOperand("Dest", OperandKind::Out)
    .addMember(MemberType::Float, "Val")
    .autoVerify(VerifyKind::NoVerify);

BB.newBackendSpecificInstr("TL_Cpu_Op")
    .addMember(MemberType::String, "Op_Name")
    .addMember(MemberType::String, "Payload")
    .addMember(MemberType::Unsigned, "Size")
    .autoVerify(VerifyKind::NoVerify);

#include "SophonMI.h"
#include "SophonOpInstrs.h"

/// verification
BB.includeBackendSpecificVerification("SophonSpecificInstrsVerification.h");
