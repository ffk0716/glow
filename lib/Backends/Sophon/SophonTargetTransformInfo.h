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
#ifndef SOPHON_TARGET_TRANSFORM_INFO_H
#define SOPHON_TARGET_TRANSFORM_INFO_H

#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include <vector>

namespace glow {
namespace sophon {

class SophonTargetTransformInfo {
public:
  virtual size_t getLMemSizeFromValue(glow::Value *value) const { return 0; }
  virtual bool isEUAligned(const glow::AllocActivationInst *Inst) const { return true; }
  virtual std::vector<size_t> getLMemSizeFromInst(glow::Instruction *Inst) const { return std::vector<size_t>(); }
  virtual int getLocalMemSizeInBytes() const { return 0; }
  virtual int getTPUNum() const { return 0; }
  virtual int getNPUNum() const { return 0; }
  virtual int getEUNum() const { return 0; }
};
} // namespace sophon

} // namespace glow
#endif // SOPHON_TARGET_TRANSFORM_INFO_H
