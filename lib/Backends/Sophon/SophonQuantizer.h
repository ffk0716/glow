/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
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
