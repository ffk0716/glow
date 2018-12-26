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

#define DEBUG_TYPE "sophon_quantizer"

#include "glow/Support/Debug.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <bmnet/utils/io.hpp>

#include "SophonQuantizer.h"

using llvm::dyn_cast;

namespace glow {

const LayerCalibrationParameter *
SophonQuantizer::getLayerCalibrationParameter(const std::string &name) {
  for (int i = 0; i < netCalibrationParameter_.layer_size(); i++) {
    const LayerCalibrationParameter &layer = netCalibrationParameter_.layer(i);
    if (layer.name() == name) {
      return &layer;
    }
    for (int j = 0; j < layer.blob_param_size(); j++) {
      if (layer.blob_param(j).name() == name) {
        return &layer;
      }
    }
  }
  return nullptr;
}

TypeRef SophonQuantizer::getTargetTypeForOutput(const NodeValue &out) const {
  if (out.getElementType() != ElemKind::FloatTy) {
    return nullptr;
  }
  // TODO: Deal with Int16QTy output case in FC.
  return mod_.uniqueType(ElemKind::Int8QTy, out.dims(), 1, 0);
}

TypeRef SophonQuantizer::getTargetTypeForInput(const Node &use,
                                               unsigned idx) const {
  NodeValue val = use.getNthInput(idx);

  // Do not quantize non floating point type, e.g., Index type.
  if (val.getElementType() != ElemKind::FloatTy) {
    return nullptr;
  }

  // For bias of a conv/fc op, it is quantized to int16.
  if (use.getKind() == glow::Kinded::Kind::SophonConvolutionNodeKind &&
      idx == 2) {
    auto convN = llvm::dyn_cast<SophonConvolutionNode>(&use);
    return mod_.uniqueType(ElemKind::Int16QTy, val.dims(), 1, 0);
  } else if (use.getKind() == glow::Kinded::Kind::FullyConnectedNodeKind &&
             idx == 2) {
    return mod_.uniqueType(ElemKind::Int16QTy, val.dims(), 1, 0);
  } else {
    return mod_.uniqueType(ElemKind::Int8QTy, val.dims(), 1, 0);
  }
}

bool SophonQuantizer::canConvert(const Node &node) const {
  auto kind = node.getKind();

  if (!EE_.isOpSupported(kind, ElemKind::Int8QTy)) {
    return false;
  }

  // Make sure that all inputs are floats.
  for (unsigned i = 0, e = node.getNumInputs(); i < e; ++i) {
    if (node.getNthInput(i).getElementType() != ElemKind::FloatTy) {
      return false;
    }
  }

  return true;
}

Node *SophonQuantizer::createConversion(Function &function, NodeValue &val,
                                        TypeRef destTy) {
  if (destTy->isQuantizedType()) {
    assert((destTy->getElementType() == ElemKind::Int8QTy ||
            destTy->getElementType() == ElemKind::Int16QTy ||
            destTy->getElementType() == ElemKind::Int32QTy) &&
           "We only support int8_t int16_t and int32_t quantization now");
    return function_.createQuantize("quantize", val, destTy);
  }

  assert(destTy->getElementType() == ElemKind::FloatTy && "");
  return function.createDequantize("dequantize", val);
}

/// Replace nodes with SophonQ8 nodes.
Node &SophonQuantizer::morphNode(Node &node) {
  Node *quantizedNode{};
  // Some nodes are glow-HIR and some are sophon-HIR. Depends on loader.
  if (auto *Conv = dyn_cast<SophonConvolutionNode>(&node)) {
    auto QT =
        mod_.uniqueType(ElemKind::Int8QTy, Conv->getResult().dims(), 1, 0);
    const auto *calibration_parameter =
        getLayerCalibrationParameter(Conv->getName());
    GLOW_ASSERT(calibration_parameter);
    int right_shift_width = calibration_parameter->right_shift_width();

    quantizedNode = function_.addNode(new SophonConvolutionQ8Node(
        Conv->getName(), QT, node.getNthInput(0), node.getNthInput(1),
        node.getNthInput(2), {Conv->getStrides()[0], Conv->getStrides()[1]},
        {Conv->getPads()[0], Conv->getPads()[1], Conv->getPads()[2],
         Conv->getPads()[3]},
        {1, 1}, // defalut DilationHW is 1,1
        right_shift_width,
        false // EnableRelu
        ));
  } else if (auto *FC = dyn_cast<FullyConnectedNode>(&node)) {
    auto QT = mod_.uniqueType(ElemKind::Int8QTy, FC->getResult().dims(), 1, 0);
    const auto *calibration_parameter =
        getLayerCalibrationParameter(FC->getName());
    GLOW_ASSERT(calibration_parameter);
    int right_shift_width = calibration_parameter->right_shift_width();
    quantizedNode = function_.addNode(
        new SophonFullyConnectedQ8Node(FC->getName(), QT, node.getNthInput(0),
                                       node.getNthInput(1), node.getNthInput(2),
                                       false, // Relu
                                       right_shift_width, 0, false));
  } else if (auto *Relu = dyn_cast<SophonReluNode>(&node)) {
    auto QT =
        mod_.uniqueType(ElemKind::Int8QTy, Relu->getResult().dims(), 1, 0);
    quantizedNode = function_.addNode(
        new SophonReluQ8Node(Relu->getName(), QT, node.getNthInput(0)));
  } else if (auto *Maxpool = dyn_cast<SophonMaxPoolNode>(&node)) {
    auto QT =
        mod_.uniqueType(ElemKind::Int8QTy, Maxpool->getResult().dims(), 1, 0);
    const auto *calibration_parameter =
        getLayerCalibrationParameter(Maxpool->getName());
    GLOW_ASSERT(calibration_parameter);
    int right_shift_width = calibration_parameter->right_shift_width();
    const int *threshold_x_quantized =
        calibration_parameter->threshold_x_quantized().data();
    quantizedNode = function_.addNode(new SophonMaxPoolQ8Node(
        Maxpool->getName(), QT, node.getNthInput(0),
        {Maxpool->getKernels()[0], Maxpool->getKernels()[1]},
        {Maxpool->getStrides()[0], Maxpool->getStrides()[1]},
        {Maxpool->getPads()[0], Maxpool->getPads()[1], Maxpool->getPads()[2],
         Maxpool->getPads()[3]},
        right_shift_width, threshold_x_quantized[0], Maxpool->getRoundMode()));
  }

  if (quantizedNode != nullptr) {
    NodeValue(&node, 0).replaceAllUsesOfWith(quantizedNode);
    return *quantizedNode;
  } else {
    return node;
  }
}

void SophonQuantizer::convertTensor(Tensor &tensor, TypeRef destTy) {
  assert(tensor.getElementType() == ElemKind::FloatTy &&
         destTy->getElementType() == ElemKind::Int8QTy &&
         "Dequantization not implemented");
  // Do nothing now.
}

} // namespace glow

namespace glow {
namespace quantizesophon {

const NetCalibrationParameter loadCtableFile(const std::string &filename) {
  NetCalibrationParameter netCalibrationParameter;
  bmnet::ReadProtoFromBinaryFile(filename, &netCalibrationParameter);
  std::string ctableNameString = netCalibrationParameter.DebugString();
  DEBUG_GLOW(llvm::dbgs() << "ImportCalibrationTable: "
                          << "\n"
                          << ctableNameString << "\n");
  return netCalibrationParameter;
}

void quantizeSophonGraph(const ExecutionEngine &EE, Function *F,
                         const std::string &filename) {
  const NetCalibrationParameter netCalibrationParameter =
      loadCtableFile(filename);
  SophonQuantizer sq(EE, *F, netCalibrationParameter);
  sq.convert();
}

} // namespace quantizesophon
} // namespace glow
