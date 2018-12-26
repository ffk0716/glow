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
#define DEBUG_TYPE "visitor"
#include "GlowLIRVisitor.h"
#include "glow/Support/Debug.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

using namespace glow;

#define VISIT_INSTR(CLASS)                                                      \
  case glow::Kinded::Kind::CLASS##Kind: {                                       \
    this->visit(llvm::cast<glow::CLASS>(Inst));                                 \
  } break;

void GlowLIRVisitor::accept_helper(glow::Instruction *Inst) {
  switch (Inst->getKind()) {
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME) VISIT_INSTR(CLASS)
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) VISIT_INSTR(CLASS)
#include "glow/AutoGenInstr.def"
  default:
    llvm_unreachable("Unknown value kind");
    break;
  }
}

#define CONST_VISIT_INSTR(CLASS)                                               \
  case glow::Kinded::Kind::CLASS##Kind: {                                      \
    this->visit(llvm::cast<const glow::CLASS>(Inst));                          \
  } break;

void GlowLIRVisitor::accept_helper(const glow::Instruction *Inst) {
  switch (Inst->getKind()) {
#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME) CONST_VISIT_INSTR(CLASS)
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) CONST_VISIT_INSTR(CLASS)
#include "glow/AutoGenInstr.def"
  default:
    llvm_unreachable("Unknown value kind");
    break;
  }
}
