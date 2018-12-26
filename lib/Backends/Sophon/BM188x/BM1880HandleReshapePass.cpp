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
#include "BM1880HandleReshapePass.h"
#include "Backends/Sophon/AllocationsInfo.h"
#include "Backends/Sophon/GlowLIRVisitor.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/IRUtils.h"
#include "glow/Support/Debug.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <memory>
#include <string>

using namespace glow;

#define DEBUG_TYPE "GenLS"

class HandleReshape : public GlowLIRVisitor {
public:
  void runOnIRFunction(IRFunction *F);

private:
  void visit(TensorViewInst *tv) override;

  void default_method(glow::Instruction *Inst) override {}

  IRFunction *F_;
};

void HandleReshape::visit(TensorViewInst *tv) {
  IRBuilder builder(F_);
  // auto get_use = [&](glow::Instruction *Inst)
  auto origin = getOrigin(tv);
  // we only handle local reshape only
  if (not llvm::isa<AllocActivationInst>(origin))
    return;

  std::string name = std::string(tv->getName());
  // gen tmp weight
  auto *W = builder.createWeightVar(origin->getType(), name + ".spill",
                                    WeightVar::MutabilityKind::Mutable);
  auto *PH = F_->getGraph()->getParent()->createPlaceholder(
      origin->getType(),
      llvm::StringRef(std::string("save_") + name + "_sophon_spill"), false);
  // update variable map
  F_->getVariableMap()[PH] = W;

  // insert store
  auto *S = builder.createSophonStoreInst(name, W, origin);
  F_->moveInstruction(tv, S);
  // insert local tensor
  auto *A = builder.createAllocActivationInst(name, tv->getType());
  F_->moveInstruction(tv, A);
  // insert load
  auto *L = builder.createSophonLoadInst(name, A, W);
  F_->moveInstruction(tv, L);
  {
    auto users = tv->getUsers();
    for (auto it = users.begin(), e = users.end(); it != e;) {
      auto &user = *it;
      it++;
      if (user.getOperand().second == OperandKind::In)
        user.setOperand(A);
    }
  }
  // delete tensorView
  F_->eraseInstruction(tv);
}

void HandleReshape::runOnIRFunction(IRFunction *F) {
  F_ = F;

  auto &instrs = F->getInstrs();
  for (auto it = instrs.begin(), e = instrs.end(); it != e;) {
    auto &I = *it;
    it++;
    accept_helper(&I);
  }
}

namespace glow {
namespace sophon {
void runHandleReshape(IRFunction *F) {
  auto p = llvm::make_unique<HandleReshape>();
  p->runOnIRFunction(F);
}

} // namespace sophon
} // namespace glow
