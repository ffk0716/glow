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
#ifndef GLOW_BACKENDS_Sophon_SophonBackend_H
#define GLOW_BACKENDS_Sophon_SophonBackend_H

#include "AllocationsInfo.h"
#include "SophonTargetTransformInfo.h"

#include "glow/Backends/Backend.h"
#include "glow/Backends/CompiledFunction.h"
#include "glow/Graph/Node.h"

namespace glow {

namespace Sophon {
enum Target {
  BM1682,
  BM1880,
  BM1882,
};
} // namespace Sophon

using SophonCmdBuf = std::vector<uint8_t>;

// Base class for all Sophon platform
class SophonBackend : public BackendUsingGlowIR {
public:
  SophonBackend() = default;

  ~SophonBackend() override = default;

  static Backend *createBackend();

  virtual uint32_t getTarget() const = 0; // such as 1682/1880/...

  virtual void generateWeights(IRFunction *F, AllocationsInfo &allocationsInfo,
                               std::vector<uint8_t> &weights) const = 0;
  virtual void codeGenCmdbuf(IRFunction *F, AllocationsInfo &allocationsInfo,
                             SophonCmdBuf &cmdbuf) const = 0;

  // feature
  virtual bool hasFP32Inst() const = 0;
  virtual bool hasInt8Inst() const = 0;
  virtual sophon::SophonTargetTransformInfo *getTTI() const { return nullptr; }
};

} // namespace glow

#endif // GLOW_BACKENDS_Sophon_SophonBackend_H
