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

void SophonConvolutionNode::verify() const {
  // TODO: just demo
  ShapeNCHW idim(getInput().getType()->dims());
  ShapeNCHW odim(getResult().getType()->dims());
  auto outSz = calculateConvPoolOutputDims(idim.h, idim.w, getKernels(),
                                           getStrides(), getPads());
  ShapeNCHW exp(idim.n, getBias().dims()[0], outSz.first, outSz.second);
  (void)exp;
  assert(exp == odim && "Invalid output dimensions");
}

void SophonConvolutionWithoutBiasNode::verify() const {}

void SophonMaxPoolNode::verify() const {}
void SophonAvgPoolNode::verify() const {}
void SophonFullyConnectedNode::verify() const {}
void SophonMatMulNode::verify() const {}


void SophonNormalizeNode::verify() const {}
void SophonBatchNormalizationNode::verify() const {}
void SophonBatchNormalizationOptNode::verify() const {}
void SophonLocalResponseNormalizationNode::verify() const {}


void SophonReluNode::verify() const {}
void SophonTanhNode::verify() const {}
void SophonSigmoidNode::verify() const {}
void SophonPreluNode::verify() const {}


void SophonSoftMaxNode::verify() const {}
void SophonPriorboxNode::verify() const {assert(0);}
void SophonUpsampleNode::verify() const {}
void SophonDeconvolutionNode::verify() const {}
void SophonDeconvolutionWithoutBiasNode::verify() const {}
void SophonDeconvolutionOptNode::verify() const {}
void SophonDeconvolutionWithoutBiasOptNode::verify() const {}
void SophonROIPoolNode::verify() const {}
void SophonPSROIPoolNode::verify() const {}
void SophonMultiRegionNode::verify() const {}
void SophonLSTMNode::verify() const {}
void SophonShuffleChannelNode::verify() const {}
void SophonSliceNode::verify() const {}


void SophonConcatNode::verify() const {}
void SophonConcat2Node::verify() const {}
void SophonConcat3Node::verify() const {}
void SophonConcat4Node::verify() const {}
void SophonConcat5Node::verify() const {}
void SophonConcat6Node::verify() const {}
void SophonReshapeNode::verify() const {}
void SophonTransposeNode::verify() const {}
void SophonFlattenNode::verify() const {}
void SophonCropNode::verify() const {}
void SophonReorgNode::verify() const {}
void SophonPermuteNode::verify() const {}
void SophonDummyDataNode::verify() const {}
void SophonEltwiseNode::verify() const {}
void SophonTileNode::verify() const {}


void SophonScaleNode::verify() const {}
void SophonScaleWithoutBiasNode::verify() const {}
void SophonScale1Node::verify() const {}
void SophonMulNode::verify() const {}
void SophonAddNode::verify() const {}
void SophonMaxNode::verify() const {}
void SophonPowNode::verify() const {}
void SophonAbsNode::verify() const {}
void SophonSubNode::verify() const {}
void SophonDivNode::verify() const {}

void SophonProposalNode::verify() const {assert(0);}
void SophonRegionNode::verify() const {}
void SophonYoloNode::verify() const {}
void SophonInterpNode::verify() const {}

//void SophonSliceNode::verify() const {}
void SophonReductionNode::verify() const {}







