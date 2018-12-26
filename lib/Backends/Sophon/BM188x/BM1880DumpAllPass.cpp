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
#include "BM1880DumpAllPass.h"
#include "Backends/Sophon/AllocationsInfo.h"
#include "Backends/Sophon/GlowLIRVisitor.h"
#include "glow/IR/IRBuilder.h"
#include "glow/Support/Debug.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <memory>
#include <string>

using namespace glow;

#define DEBUG_TYPE "GenLS"

class DumpAllPass : public GlowLIRVisitor {
public:
  void runOnIRFunction(IRFunction *F);

private:
  void visit(AllocActivationInst *Inst) override {}
  void visit(DeallocActivationInst *Inst) override {}
  void visit(SophonStoreInst *Inst) override {}
  void visit(SophonLoadInst *Inst) override {}

  void default_method(glow::Instruction *Inst) override {
    std::cout << "debug: " << Inst->getName().str() << std::endl;
    IRBuilder builder(F_);
    for (unsigned i = 0; i < Inst->getNumOperands(); i++) {
      const auto &op = Inst->getOperand(i);
      if (op.second != OperandKind::Out)
        continue;
      auto *dest = op.first;
      if (not llvm::isa<AllocActivationInst>(dest))
        continue;
      // add weightVar
      auto *W = builder.createWeightVar(dest->getType(),
                                        dest->getName().str() + ".spill",
                                        WeightVar::MutabilityKind::Mutable);
      // add placeholder
      auto *PH = F_->getGraph()->getParent()->createPlaceholder(
          dest->getType(),
          std::string("save_") + Inst->getName().str() + "_dump", false);
      // update variable map
      F_->getVariableMap()[PH] = W;
      // add load inst
      auto *S = builder.createSophonStoreInst(Inst->getName().str() + "_dump",
                                              W, dest);
      F_->moveInstruction(Inst, S);
      F_->moveInstruction(S, Inst);
    }
  }

  IRFunction *F_;
};

void DumpAllPass::runOnIRFunction(IRFunction *F) {
  F_ = F;
  auto &instrs = F->getInstrs();
  for (auto it = instrs.begin(), e = instrs.end(); it != e;) {
    auto cur = it;
    auto &I = *it;
    it++;
    accept_helper(&I);
  }
}

namespace glow {
namespace sophon {
void runDumpAllPass(IRFunction *F) {
  auto p = llvm::make_unique<DumpAllPass>();
  p->runOnIRFunction(F);
}

} // namespace sophon
} // namespace glow
