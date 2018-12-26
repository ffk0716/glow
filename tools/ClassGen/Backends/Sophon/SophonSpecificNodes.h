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

//===--------------------------------------------------------------------===//
//                   Convolution / Pool / FC
//===--------------------------------------------------------------------===//

BB.newNode("SophonConvolution")
    .addInput("Input")
    .addInput("Filter")
    .addInput("Bias")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::Unsigned, "Group")
    .addResultFromCtorArg()
    .setDocstring("Performs Convolution using a given Input, Filter, and "
                  "Bias tensors, as well as provided Kernels, Strides, Pads, "
                  "and Group.");

BB.newNode("SophonConvolutionWithoutBias")
    .addInput("Input")
    .addInput("Filter")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::Unsigned, "Group")
    .addResultFromCtorArg()
    .setDocstring("Performs Convolution using a given Input, Filter, "
                  "as well as provided Kernels, Strides, Pads, "
                  "and Group.");

BB.newNode("SophonConvolutionQ8")
    .addInput("Input")
    .addInput("Filter")
    .addInput("Bias")
    .addMember(MemberType::VectorUnsigned, "StrideHW")
    .addMember(MemberType::VectorUnsigned, "PadTLBR")
    .addMember(MemberType::VectorUnsigned, "DilationHW")
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .addMember(MemberType::Boolean, "EnableRelu")
    .addResultFromCtorArg()
    .setDocstring("Performs Convolution int8 using a given Input, Filter, and "
                  "Bias tensors, as well as provided Kernels, Strides, Pads, "
                  "and Group.");

BB.newNode("SophonMaxPool")
    .addInput("Input")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::Boolean, "RoundMode")
    .addResultFromCtorArg()
    .setDocstring("Performs a Max Pool operation on the Input given provided "
                  "Kernels, Strides, and Pads.");

BB.newNode("SophonMaxPoolQ8")
    .addInput("Input")
    .addMember(MemberType::VectorUnsigned, "KernelHW")
    .addMember(MemberType::VectorUnsigned, "StrideHW")
    .addMember(MemberType::VectorUnsigned, "PadTLBR")
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .addMember(MemberType::Unsigned, "Multiplier")
    .addMember(MemberType::Boolean, "RoundMode")
    .addResultFromCtorArg()
    .setDocstring(
        "Performs a Max Pool int8 operation on the Input given provided "
        "Kernels, Strides, and Pads.");

BB.newNode("SophonAvgPool")
    .addInput("Input")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::Boolean, "RoundMode")
    .addResultFromCtorArg()
    .setDocstring("Performs a Avg Pool operation on the Input given provided "
                  "Kernels, Strides, and Pads.");

BB.newNode("SophonAvgPoolQ8")
    .addInput("Input")
    .addMember(MemberType::VectorUnsigned, "KernelHW")
    .addMember(MemberType::VectorUnsigned, "StrideHW")
    .addMember(MemberType::VectorUnsigned, "PadTLBR")
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .addMember(MemberType::Unsigned, "Multiplier")
    .addMember(MemberType::Boolean, "RoundMode")
    .addResultFromCtorArg()
    .setDocstring(
        "Performs a Avg Pool int8 operation on the Input given provided "
        "Kernels, Strides, and Pads.");

#if 1 // use SophonMatMul instead
BB.newNode("SophonFullyConnected")
    .addInput("Input")
    .addInput("Weights")
    .addInput("Bias")
    .addMember(MemberType::Boolean, "Relu") // TODO(wwcai)
    .addResultFromCtorArg()
    .setDocstring("Creates a FullyConnected node where the Input tensor and "
                  "Weights tensor are multiplied, and then the Bias tensor "
                  "is added to it, producing the Output.");
#endif
BB.newNode("SophonFullyConnectedQ8")
    .addInput("Input")
    .addInput("Weights")
    .addInput("Bias")
    .addMember(MemberType::Boolean, "Relu")
    .addMember(MemberType::Unsigned, "RShiftWidth")
    .addMember(MemberType::Unsigned, "LShiftWidth")
    .addMember(MemberType::Boolean, "ResultAdd")
    .addResultFromCtorArg()
    .setDocstring(
        "Creates a FullyConnected int8 node where the Input tensor and "
        "Weights tensor are multiplied, and then the Bias tensor "
        "is added to it, producing the Output.");

BB.newNode("SophonMatMul")
    .addInput("LHS")
    .addInput("RHS")
    .addInput("Slice")
    .addResultFromCtorArg()
    .setDocstring(
        "Performs matrix multiplication between the LHS RHS, and "
        "Adds the 'Slice' operand to each one of the slices in the batch."
        "Example: (A, Z) x (Z, B) + (B) => (A, B)");

//===--------------------------------------------------------------------===//
//                     Normalization
//===--------------------------------------------------------------------===//

BB.newNode("SophonNormalize")
    .addInput("Input")
    //    .addInput("Scale") //!< delete
    .addMember(MemberType::Boolean, "AcrossSpatial")
    .addMember(MemberType::Boolean, "ChannelShared")
    .addMember(MemberType::Float, "Epsilon")
    .addMember(MemberType::Float, "Scale") //!< add this
    .addResultFromCtorArg()
    .setDocstring("Performs batch normalization on the Input tensor with the "
                  "provided Scale, Bias, Mean, Var, ChannelIdx, Epsilon, and "
                  "Momentum. Similar to Caffe2 SpatialBN, and ONNX "
                  "BatchNormalization operator.");

BB.newNode("SophonBatchNormalization")
    .addInput("Input")
    //    .addInput("Bias")   //!< delete
    .addInput("Mean")
    .addInput("Variance")
    .addMember(MemberType::Float, "Scale")
    .addMember(MemberType::Float, "Epsilon")
    .addResultFromCtorArg()
    .setDocstring("Performs batch normalization on the Input tensor with the "
                  "provided Scale, Bias, Mean, Var, ChannelIdx, Epsilon, and "
                  "Momentum. Similar to Caffe2 SpatialBN, and ONNX "
                  "BatchNormalization operator.");

BB.newNode("SophonBatchNormalizationOpt")
    .addInput("Input")
    .addInput("Mean")
    .addInput("Variance")
    .addMember(MemberType::Float, "Scale")
    .addMember(MemberType::Float, "Epsilon")
    .addResultFromCtorArg()
    .setDocstring("Performs batch normalization on the Input tensor with the "
                  "provided Scale, Bias, Mean, Var, ChannelIdx, Epsilon, and "
                  "Momentum. Similar to Caffe2 SpatialBN, and ONNX "
                  "BatchNormalization operator.");

BB.newNode("SophonLocalResponseNormalization")
    .addInput("Input")
    .addMember(MemberType::Float, "Alpha")
    .addMember(MemberType::Float, "Beta")
    //.addMember(MemberType::Float, "bias") caffe doesn't have this parameter,
    // comment it
    .addMember(MemberType::Unsigned, "NormRegion")
    .addMember(MemberType::Unsigned, "Size")
    .addMember(MemberType::Float, "K")
    .addResultFromCtorArg()
    .setDocstring("Performs local response normalization on the Input tensor "
                  "with the provided Scale, Bias, Mean, Var, ChannelIdx, "
                  "Epsilon, and Momentum. Similar to Caffe2 and ONNX LRN.");

//===--------------------------------------------------------------------===//
//                     Activation
//===--------------------------------------------------------------------===//
BB.newNode("SophonRelu")
    .addInput("Input")
    .addMember(MemberType::Float, "NegativeSlope")
    .addResultFromCtorArg()
    .setDocstring(
        "Applies ReLU, max(0, x), to each element in the Input tensor.");

BB.newNode("SophonReluQ8")
    .addInput("Input")
    .addResultFromCtorArg()
    .setDocstring(
        "Applies ReLU, max(0, x), to each element in the Input int8 tensor.");

BB.newNode("SophonSigmoid")
    .addInput("Input")
    .addResultFromCtorArg()
    .setDocstring("Applies Sigmoid, 1 / (1 + exp(-x)), to each element in "
                  "the Input tensor.");

BB.newNode("SophonTanh")
    .addInput("Input")
    .addResultFromCtorArg()
    .setDocstring("Applies hyperbolic tangent to each element in the Input "
                  "tensor.");

BB.newNode("SophonPrelu")
    .addInput("Input")
    .addInput("Slope")
    .addMember(MemberType::Boolean, "ChannelShared")
    .addResultFromCtorArg()
    .setDocstring("PRelu takes input data (Tensor) and slope tensor as input,"
                  "and produces one output data (Tensor) where the function"
                  "f(x) = slope * x for x < 0, f(x) = x for x >= 0., "
                  "is applied to the data tensor elementwise.");

//===--------------------------------------------------------------------===//
//                     Other NN operations
//===--------------------------------------------------------------------===//

BB.newNode("SophonSoftMax")
    .addInput("Input")
    .addMember(MemberType::Unsigned, "Axis")
    .addResultFromCtorArg()
    .setDocstring("Performs SoftMax normalization on the Input tensor.");

BB.newNode("SophonPriorbox")
    .addInput("Input")
    .addInput("Weight")
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
    .addResultFromCtorArg()
    .setDocstring("PriorBox takes input data, and produces prior box of "
                  "featuremap. Similar to SSD PriorBox.");

BB.newNode("SophonUpsample")
    .addInput("Input")
    .addMember(MemberType::Unsigned, "Size")
    .addResultFromCtorArg()
    .setDocstring("Upsample the input tensor. Each dimension value of the "
                  "output tensor is: output_dimension = size.");

BB.newNode("SophonDeconvolution")
    .addInput("Input")
    .addInput("Filter")
    .addInput("Bias")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::Unsigned, "Group")
    .addResultFromCtorArg()
    .setDocstring(
        "Performs Deconvolution using a given Input, Filter, and "
        "Bias tensors, as well as provided Kernels, Strides, Pads, and Group.");

BB.newNode("SophonDeconvolutionWithoutBias")
    .addInput("Input")
    .addInput("Filter")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::Unsigned, "Group")
    .addResultFromCtorArg()
    .setDocstring(
        "Performs Deconvolution using a given Input, Filter, and "
        "Bias tensors, as well as provided Kernels, Strides, Pads, and Group.");

BB.newNode("SophonDeconvolutionOpt")
    .addInput("Input")
    .addInput("Filter")
    .addInput("Bias")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::Unsigned, "Group")
    .addResultFromCtorArg()
    .setDocstring("Performs Deconvolution using a given Input and Filter, "
                  "as well as provided Kernels, Strides, Pads, and Group.");

BB.newNode("SophonDeconvolutionWithoutBiasOpt")
    .addInput("Input")
    .addInput("Filter")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::VectorUnsigned, "Dilations")
    .addMember(MemberType::Unsigned, "Group")
    .addResultFromCtorArg()
    .setDocstring("Performs Deconvolution using a given Input and Filter,"
                  "as well as provided Kernels, Strides, Pads, and Group.");

BB.newNode("SophonROIPool")
    .addInput("Input")
    .addInput("Rois")
    .addMember(MemberType::VectorUnsigned, "PoolShape")
    .addMember(MemberType::Float, "SpatialScale")
    .addResultFromCtorArg()
    .setDocstring(
        "ROI pool consumes an input tensor X and region of interests "
        "(RoIs) to apply pooling across each RoI, to produce output 4-D tensor "
        "of shape (num_rois, channels, pooled_shape[0], pooled_shape[1]). "
        "Similar "
        "to Faster-RCNN ROIPooling.");

BB.newNode("SophonPSROIPool")
    .addInput("Input")
    .addMember(MemberType::Float, "SpatialScale")
    .addMember(MemberType::Unsigned, "Group")
    .addResultFromCtorArg()
    .setDocstring(
        "Position sensitive ROI pooling. Similar to RFCN PSROIPooling.");

BB.newNode("SophonMultiRegion")
    .addInput("Input")
    .addMember(MemberType::Unsigned, "Classes")
    .addMember(MemberType::Unsigned, "Coords")
    .addMember(MemberType::Unsigned, "Nums")
    .addMember(MemberType::VectorUnsigned, "ActivateParameters")
    .addResultFromCtorArg()
    .setDocstring("Similar to darknet multiregion.");

BB.newNode("SophonLSTM")
    .addInput("Input")
    .addInput("W")
    .addInput("R")
    .addInput("B")
    .addInput("P")
    .addMember(MemberType::Unsigned, "time_num")
    .addMember(MemberType::Unsigned, "with_x_static")
    .addMember(MemberType::Unsigned, "expose_hidden")
    .addResultFromCtorArg()
    .setDocstring("Computes an one-layer LSTM.");

BB.newNode("SophonShuffleChannel")
    .addInput("Input")
    .addMember(MemberType::Unsigned, "Group")
    .addResultFromCtorArg()
    .setDocstring("Shuffle the channels of input tensor based on the group.");

BB.newNode("SophonSlice")
    .addInput("Input")
    .addMember(MemberType::Unsigned, "axis")
    .addResultFromCtorArg()
    .setDocstring("Slice.");

//===--------------------------------------------------------------------===//
//                     Shape transformations
//===--------------------------------------------------------------------===//

BB.newNode("SophonConcat")
    .addMember(MemberType::VectorNodeValue, "Inputs")
    .addMember(MemberType::Unsigned, "Dim")
    .addResultFromCtorArg()
    .setDocstring("The concat operator adds two tensors together.\nThe "
                  "parameter 'dim' specifies the dimension to use when "
                  "joining the tensors.");

// FIXME
BB.newNode("SophonConcat2")
    .addInput("Input0")
    .addInput("Input1")
    .addMember(MemberType::Unsigned, "Dim")
    .addResultFromCtorArg()
    .setDocstring("The concat operator adds two tensors together.\nThe "
                  "parameter 'dim' specifies the dimension to use when "
                  "joining the tensors.");

BB.newNode("SophonConcat3")
    .addInput("Input0")
    .addInput("Input1")
    .addInput("Input2")
    .addMember(MemberType::Unsigned, "Dim")
    .addResultFromCtorArg()
    .setDocstring("The concat operator adds two tensors together.\nThe "
                  "parameter 'dim' specifies the dimension to use when "
                  "joining the tensors.");

BB.newNode("SophonConcat4")
    .addInput("Input0")
    .addInput("Input1")
    .addInput("Input2")
    .addInput("Input3")
    .addMember(MemberType::Unsigned, "Dim")
    .addResultFromCtorArg()
    .setDocstring("The concat operator adds two tensors together.\nThe "
                  "parameter 'dim' specifies the dimension to use when "
                  "joining the tensors.");

BB.newNode("SophonConcat5")
    .addInput("Input0")
    .addInput("Input1")
    .addInput("Input2")
    .addInput("Input3")
    .addInput("Input4")
    .addMember(MemberType::Unsigned, "Dim")
    .addResultFromCtorArg()
    .setDocstring("The concat operator adds two tensors together.\nThe "
                  "parameter 'dim' specifies the dimension to use when "
                  "joining the tensors.");

BB.newNode("SophonConcat6")
    .addInput("Input0")
    .addInput("Input1")
    .addInput("Input2")
    .addInput("Input3")
    .addInput("Input4")
    .addInput("Input5")
    .addMember(MemberType::Unsigned, "Dim")
    .addResultFromCtorArg()
    .setDocstring("The concat operator adds two tensors together.\nThe "
                  "parameter 'dim' specifies the dimension to use when "
                  "joining the tensors.");

BB.newNode("SophonReshape")
    .addInput("Input")
    .addMember(MemberType::VectorSizeT, "Dims")
    .addResultFromCtorArg()
    .setDocstring("Reshape the Input tensor to shape Dims.");

BB.newNode("SophonTranspose")
    .addInput("Input")
    .addMember(MemberType::VectorUnsigned, "Shuffle")
    .addResultFromCtorArg()
    .setDocstring("Transpose the Input tensor based on the vector Shuffle, "
                  "which assigns a new axis for each dimension in Input.");

BB.newNode("SophonFlatten")
    .addInput("Input")
    .addResultFromCtorArg()
    .setDocstring("Flattens the input tensor into a 2D matrix.");

BB.newNode("SophonReorg")
    .addInput("Input")
    .addMember(MemberType::Unsigned, "Stride")
    .addResultFromCtorArg()
    .setDocstring("The reorganization layer takes every alternate pixel and "
                  "puts that into a different channel.");

BB.newNode("SophonCrop")
    .addInput("Input")
    .addMember(MemberType::VectorUnsigned, "Offsets")
    .addResultFromCtorArg()
    .setDocstring("Crop the Input tensor by Offsets.");

BB.newNode("SophonPermute")
    .addInput("Input")
    .addMember(MemberType::VectorUnsigned, "Order")
    .addResultFromCtorArg()
    .setDocstring("Replace index axis order.");

BB.newNode("SophonDummyData")
    .addInput("Input")
    .addResultFromCtorArg()
    .setDocstring("Dummy Data.");

BB.newNode("SophonEltwise")
    .addInput("LHS")
    .addInput("RHS")
    //.addMember(MemberType::VectorFloat, "Coeff")
    .addMember(MemberType::Unsigned, "Operation")
    .addMember(MemberType::Boolean, "StableProdGrad")
    .addResultFromCtorArg()
    .setDocstring(
        "The eltwise operator adds/product/max two tensors together.");

BB.newNode("SophonTile")
    .addInput("Input")
    //    .addMember(MemberType::VectorFloat, "coeff")
    .addMember(MemberType::Unsigned, "Axis")
    .addMember(MemberType::Unsigned, "Tiles")
    .addResultFromCtorArg()
    .setDocstring(
        "The eltwise operator adds/product/max two tensors together.");

//===--------------------------------------------------------------------===//
//                     Arithmetic
//===--------------------------------------------------------------------===//

BB.newNode("SophonScale")
    .addInput("Input")
    .addInput("Scale")
    .addInput("Bias")
    .addMember(MemberType::Unsigned, "Axis")
    .addMember(MemberType::Unsigned, "NumAxes")
    .addResultFromCtorArg()
    .setDocstring("Scale input tensor.");

BB.newNode("SophonScaleWithoutBias")
    .addInput("Input")
    .addInput("Scale")
    .addMember(MemberType::Unsigned, "Axis")
    .addMember(MemberType::Unsigned, "NumAxes")
    .addResultFromCtorArg()
    .setDocstring("Scale input tensor.");

BB.newNode("SophonScale1")
    .addInput("Input")
    .addInput("Bias")
    .addMember(MemberType::Unsigned, "Axis")
    .addMember(MemberType::Unsigned, "NumAxes")
    .addResultFromCtorArg()
    .setDocstring("Scale input tensor.");

BB.newNode("SophonMul")
    .addInput("LHS")
    .addInput("RHS")
    .addResultFromCtorArg()
    .setDocstring("Performs Mul on the LHS and RHS operands.");

BB.newNode("SophonAdd")
    .addInput("LHS")
    .addInput("RHS")
    .addResultFromCtorArg()
    .setDocstring("Performs Add on the LHS and RHS operands.");

BB.newNode("SophonMax")
    .addInput("LHS")
    .addInput("RHS")
    .addResultFromCtorArg()
    .setDocstring("Performs Max on the LHS and RHS operands.");

BB.newNode("SophonPow")
    .addInput("Input")
    .addMember(MemberType::Float, "Power")
    .addMember(MemberType::Float, "Scale")
    .addMember(MemberType::Float, "Shift")
    .addResultFromCtorArg()
    .setDocstring("Performs elementwise pow(LHS, RHS).");

BB.newNode("SophonSub")
    .addInput("LHS")
    .addInput("RHS")
    .addResultFromCtorArg()
    .setDocstring("Performs Sub on the LHS and RHS operands.");

BB.newNode("SophonDiv")
    .addInput("LHS")
    .addInput("RHS")
    .addResultFromCtorArg()
    .setDocstring("Performs Div on the LHS and RHS operands.");

//===--------------------------------------------------------------------===//
//                     Others
//===--------------------------------------------------------------------===//
BB.newNode("SophonAbs")
    .addInput("Input")
    .addResultFromCtorArg()
    .setDocstring("Applies Abs.");

BB.newNode("SophonYolo")
    .addInput("Input")
    .addMember(
        MemberType::Unsigned,
        "classes") // by ycs: in bmnet-caffe.proto, it's int32? not usigned?
    .addResultFromCtorArg()
    .setDocstring("yolo.");

BB.newNode("SophonRegion")
    .addInput("Input")
    .addMember(
        MemberType::Unsigned,
        "classes") // by ycs: in bmnet-caffe.proto, it's int32? not usigned?
    .addResultFromCtorArg()
    .setDocstring("Region.");

BB.newNode("SophonProposal")
    .addInput("Input")
    .addMember(MemberType::Unsigned, "feat_stride")
    .addMember(MemberType::Unsigned, "pre_nms_topN")
    .addMember(MemberType::Unsigned, "post_nms_topN")
    .addMember(MemberType::Float, "nms_thresh")
    .addMember(MemberType::Unsigned, "min_size")
    .addMember(MemberType::Unsigned, "base_size")
    .addMember(MemberType::Unsigned, "version")
    //    .addMember(MemberType::VectorFloat, "scale") by ycs: cann't use
    //    vectorfloat yet, need fix .addMember(MemberType::VectorFloat, "ratio")
    .addResultFromCtorArg()
    .setDocstring("Prosal.");

/******************
BB.newNode("SophonSlice")
    .addInput("Input")
    .addMember(MemberType::Unsigned, "Axis")
    .addResultFromCtorArg()
    .addResultFromCtorArg()
    .setDocstring(
      "Applies slice.");
******************/

BB.newNode("SophonReduction")
    .addInput("Input")
    .addMember(MemberType::Unsigned, "Axis")
    .addMember(MemberType::Unsigned, "Operation")
    .addMember(MemberType::Float, "Coeff")
    .addResultFromCtorArg()
    .setDocstring("Applies slice.");
BB.newNode("SophonInterp")
    .addInput("Input")
    .addInput("Weights")
    .addResultFromCtorArg()
    .setDocstring("Interp.");

BB.includeBackendSpecificVerification("SophonSpecificNodesVerification.h");
