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
#ifndef BM1880_TARGET_TRANSFORM_INFO_H
#define BM1880_TARGET_TRANSFORM_INFO_H

#include "Backends/Sophon/SophonTargetTransformInfo.h"
namespace glow {
namespace sophon {

class BM1880TargetTransformInfo : public SophonTargetTransformInfo {
public:
  static BM1880TargetTransformInfo *getInstance();
  BM1880TargetTransformInfo(BM1880TargetTransformInfo const &) = delete;
  void operator=(BM1880TargetTransformInfo const &) = delete;

public:
  size_t getLMemSizeFromValue(glow::Value *value) const override;
  bool isEUAligned(const glow::AllocActivationInst *Inst) const override;

  std::vector<size_t>
  getLMemSizeFromInst(glow::Instruction *Inst) const override;

  int getLocalMemSizeInBytes() const override;
  int getTPUNum() const override;
  int getNPUNum() const override;
  int getEUNum() const override;

private:
  BM1880TargetTransformInfo();
};

} // namespace sophon
} // namespace glow

#endif // BM1880_TARGET_TRANSFORM_INFO_H
