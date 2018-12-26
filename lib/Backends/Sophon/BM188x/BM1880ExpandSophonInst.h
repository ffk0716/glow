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

#include "Backends/Sophon/BM188x/BM1880AllocationsInfo.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"

namespace glow {
class BM1880ExpandSophonInst {

public:
  BM1880ExpandSophonInst(IRFunction *F,
                         const BM1880AllocationsInfo &allocationsInfo)
      : F_(F), allocationsInfo_(allocationsInfo) {}
  void run();

private:
  IRFunction *F_;
  const BM1880AllocationsInfo &allocationsInfo_;
};
} // namespace glow
