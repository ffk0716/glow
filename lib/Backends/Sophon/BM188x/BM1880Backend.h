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

#pragma once

#include "Backends/Sophon/AllocationsInfo.h"
#include "Backends/Sophon/SophonBackend.h"

#include "glow/Backends/CompiledFunction.h"
#include "glow/Graph/Node.h"

namespace glow {

class BM1880AllocationsInfo;

/// This is the Sophon backend.
class BM1880Backend final : public SophonBackend {
public:
  BM1880Backend() = default;

  ~BM1880Backend() override = default;

  std::unique_ptr<CompiledFunction>
  compileIR(std::unique_ptr<IRFunction> IR) const override;
  std::unique_ptr<CompiledFunction>
  codegen(std::unique_ptr<IRFunction> IR,
          AllocationsInfo *allocationsInfo) const;

  void runOptimizationPasses(IRFunction *IR,
                             BM1880AllocationsInfo *allocationsInfo) const;

  /// JIT Mode: compile to FunctionCompiled
  std::unique_ptr<CompiledFunction> compile(Function *F) const override;

  /// AOT Mode: save to bmodel
  void save(Function *F, llvm::StringRef outputDir,
            llvm::StringRef networkName) const override;

  bool transformPreLowering(Function *F, CompilationMode mode) const override;

  bool transformPostLowering(Function *F, CompilationMode mode) const override {
    return false;
  };

  bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const override;

  bool shouldLower(const Node *N) const override;

  uint32_t getTarget() const override { return 1880; }

  void reorderWeights(IRFunction *F) const;

  void generateWeights(IRFunction *F, AllocationsInfo &allocationsInfo,
                       std::vector<uint8_t> &weights) const override;

  void codeGenCmdbuf(IRFunction *F, AllocationsInfo &allocationsInfo,
                     SophonCmdBuf &cmdbuf) const override;

  // delete quantize/dequantize nodes
  bool deleteQuantizeNodes(Function *F) const;

  // feature
  bool hasFP32Inst() const override { return false; }
  bool hasInt8Inst() const override { return true; }
  virtual sophon::SophonTargetTransformInfo *getTTI() const override;
};

} // namespace glow
