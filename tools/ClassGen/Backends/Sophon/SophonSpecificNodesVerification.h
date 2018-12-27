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
#ifdef GLOW_WITH_SOPHON

bool SophonConvolutionNode::verify() const {
  // TODO: just demo
  ShapeNCHW idim(getInput().getType()->dims());
  ShapeNCHW odim(getResult().getType()->dims());
  auto outSz = calculateConvPoolOutputDims(idim.h, idim.w, getKernels(),
                                           getStrides(), getPads());
  ShapeNCHW exp(idim.n, getBias().dims()[0], outSz.first, outSz.second);
  (void)exp;
  assert(exp == odim && "Invalid output dimensions");
  return true;
}

bool SophonConvolutionWithoutBiasNode::verify() const { return true; }

bool SophonMaxPoolNode::verify() const { return true; }
bool SophonAvgPoolNode::verify() const { return true; }
bool SophonFullyConnectedNode::verify() const { return true; }
bool SophonMatMulNode::verify() const { return true; }

bool SophonNormalizeNode::verify() const { return true; }
bool SophonBatchNormalizationNode::verify() const { return true; }
bool SophonBatchNormalizationOptNode::verify() const { return true; }
bool SophonLocalResponseNormalizationNode::verify() const { return true; }

bool SophonReluNode::verify() const { return true; }
bool SophonTanhNode::verify() const { return true; }
bool SophonSigmoidNode::verify() const { return true; }
bool SophonPreluNode::verify() const { return true; }

bool SophonSoftMaxNode::verify() const { return true; }
bool SophonPriorboxNode::verify() const { assert(0); }
bool SophonUpsampleNode::verify() const { return true; }
bool SophonDeconvolutionNode::verify() const { return true; }
bool SophonDeconvolutionWithoutBiasNode::verify() const { return true; }
bool SophonDeconvolutionOptNode::verify() const { return true; }
bool SophonDeconvolutionWithoutBiasOptNode::verify() const { return true; }
bool SophonROIPoolNode::verify() const { return true; }
bool SophonPSROIPoolNode::verify() const { return true; }
bool SophonMultiRegionNode::verify() const { return true; }
bool SophonLSTMNode::verify() const { return true; }
bool SophonShuffleChannelNode::verify() const { return true; }
bool SophonSliceNode::verify() const { return true; }

bool SophonConcatNode::verify() const { return true; }
bool SophonConcat2Node::verify() const { return true; }
bool SophonConcat3Node::verify() const { return true; }
bool SophonConcat4Node::verify() const { return true; }
bool SophonConcat5Node::verify() const { return true; }
bool SophonConcat6Node::verify() const { return true; }
bool SophonReshapeNode::verify() const { return true; }
bool SophonTransposeNode::verify() const { return true; }
bool SophonFlattenNode::verify() const { return true; }
bool SophonCropNode::verify() const { return true; }
bool SophonReorgNode::verify() const { return true; }
bool SophonPermuteNode::verify() const { return true; }
bool SophonDummyDataNode::verify() const { return true; }
bool SophonEltwiseNode::verify() const { return true; }
bool SophonTileNode::verify() const { return true; }

bool SophonScaleNode::verify() const { return true; }
bool SophonScaleWithoutBiasNode::verify() const { return true; }
bool SophonScale1Node::verify() const { return true; }
bool SophonMulNode::verify() const { return true; }
bool SophonAddNode::verify() const { return true; }
bool SophonMaxNode::verify() const { return true; }
bool SophonPowNode::verify() const { return true; }
bool SophonAbsNode::verify() const { return true; }
bool SophonSubNode::verify() const { return true; }
bool SophonDivNode::verify() const { return true; }

bool SophonProposalNode::verify() const { assert(0); }
bool SophonRegionNode::verify() const { return true; }
bool SophonYoloNode::verify() const { return true; }
bool SophonInterpNode::verify() const { return true; }

// bool SophonSliceNode::verify() const {return true;}
bool SophonReductionNode::verify() const { return true; }

#endif
