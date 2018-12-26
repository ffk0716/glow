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
#include "BM1880MemoryAllocPass.h"
#include "BM1880AllocationsInfo.h"
#include "Backends/Sophon/AllocationsInfo.h"
#include "glow/IR/IRBuilder.h"
#include "glow/Support/Debug.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <memory>
#include <string>

using namespace glow;

#define DEBUG_TYPE "BM1880_mem_alloc"

class MemoryAllocPass {
public:
  MemoryAllocPass(BM1880AllocationsInfo *allocInfo) : _allocInfo(allocInfo) {}

  void runOnIRFunction(IRFunction *F);

private:
  BM1880AllocationsInfo *_allocInfo;
};

void MemoryAllocPass::runOnIRFunction(IRFunction *F) {
  DEBUG_GLOW(llvm::dbgs() << "MemoryAllocPass::runOnIRFunction\n");
  _allocInfo->allocateWeightVars(F);
  _allocInfo->allocateActivations(F);
}

namespace glow {
namespace sophon {
void runMemoryAllocPass(IRFunction *F, BM1880AllocationsInfo *allocInfo) {
  auto p = llvm::make_unique<MemoryAllocPass>(allocInfo);
  p->runOnIRFunction(F);
}

} // namespace sophon
} // namespace glow
