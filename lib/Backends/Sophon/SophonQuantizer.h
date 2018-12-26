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

#ifndef GLOW_SOPHON_QUANTIZATION_H
#define GLOW_SOPHON_QUANTIZATION_H

#include <bmnet/calibration.pb.h>

#include "glow/Converter/FunctionConverter.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"

namespace glow {

/// This class produces a quantized function based on a provided ctable.
class SophonQuantizer : public FunctionConverter {
protected:
  TypeRef getTargetTypeForOutput(const NodeValue &out) const override;

  TypeRef getTargetTypeForInput(const Node &use, unsigned idx) const override;

  bool canConvert(const Node &node) const override;

  Node *createConversion(Function &function, NodeValue &val,
                         TypeRef destTy) override;

  Node &morphNode(Node &node) override;

  void convertTensor(Tensor &tensor, TypeRef destTy) override;

private:
  Module &mod_;

  const ExecutionEngine &EE_;

  /// Calibration table for low precision.
  const NetCalibrationParameter &netCalibrationParameter_;

public:
  SophonQuantizer(const ExecutionEngine &EE, Function &F,
                  const NetCalibrationParameter &netCalibrationParameter)
      : FunctionConverter(F), mod_(*F.getParent()), EE_(EE),
        netCalibrationParameter_(netCalibrationParameter) {}

  const LayerCalibrationParameter *
  getLayerCalibrationParameter(const std::string &name);
};

namespace quantizesophon {

const NetCalibrationParameter loadCtableFile(const std::string &filename);
void quantizeSophonGraph(const ExecutionEngine &EE, Function *F,
                         const std::string &filename);

} // namespace quantizesophon
} // namespace glow

#endif // GLOW_SOPHON_QUANTIZATION_H
