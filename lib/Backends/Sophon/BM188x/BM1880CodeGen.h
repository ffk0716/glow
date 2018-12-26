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
#ifndef BM1880_CODEGEN_H
#define BM1880_CODEGEN_H

#include "Backends/Sophon/AllocationsInfo.h"
#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Debug.h"
#include <memory>

namespace glow {

class BM1880CodeGen {
public:
  virtual void performCodeGen() = 0;
  virtual std::vector<uint8_t> getCmdbuf() = 0;
  static std::unique_ptr<BM1880CodeGen>
  createCodeGen(IRFunction *F, AllocationsInfo &allocInfo);
};

} // namespace glow

#endif // BM1880_CODEGEN_H
