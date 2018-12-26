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
#include "BM1880InsertLoadStorePass.h"
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

class InsertLoadStorePass : public GlowLIRVisitor {
public:
  void runOnIRFunction(IRFunction *F);

private:
  void visit(SophonLoadInst *Inst) override;
  void visit(SophonStoreInst *Inst) override;

  void default_method(glow::Instruction *Inst) override {
    IRBuilder builder(F_);
    for (unsigned i = 0; i < Inst->getNumOperands(); i++) {
      const auto &op = Inst->getOperand(i);
      if (llvm::isa<AllocActivationInst>(op.first))
        continue;
      std::string allocate_name =
          std::string(Inst->getName()) + "_" + std::string(op.first->getName());
      if (op.second == OperandKind::In) {
        auto *A = builder.createAllocActivationInst(allocate_name,
                                                    op.first->getType());
        auto *L =
            builder.createSophonLoadInst(allocate_name + "_load", A, op.first);
        Inst->setOperand(i, A);
        F_->moveInstruction(Inst, L);
        F_->moveInstruction(L, A);
      } else if (op.second == OperandKind::Out) {
        auto *A = builder.createAllocActivationInst(allocate_name,
                                                    op.first->getType());
        auto *S = builder.createSophonStoreInst(allocate_name + "_store",
                                                op.first, A);
        Inst->setOperand(i, A);
        F_->moveInstruction(Inst, S);
        F_->moveInstruction(S, Inst);
        F_->moveInstruction(Inst, A);
      }
    }
  }

  IRFunction *F_;
};

// avoid adding load for load
void InsertLoadStorePass::visit(SophonLoadInst *Inst) { (void)Inst; }
// avoid adding load for store
void InsertLoadStorePass::visit(SophonStoreInst *Inst) { (void)Inst; }

void InsertLoadStorePass::runOnIRFunction(IRFunction *F) {
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
void runInsertLoadStorePass(IRFunction *F) {
  auto p = llvm::make_unique<InsertLoadStorePass>();
  p->runOnIRFunction(F);
}

} // namespace sophon
} // namespace glow
