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

#include "glow/IR/Instrs.h"

namespace glow {

class GlowLIRVisitor {
public:
  GlowLIRVisitor() = default;
  virtual ~GlowLIRVisitor() = default;

  void accept_helper(glow::Instruction *Inst);
  void accept_helper(const glow::Instruction *Inst);

  virtual void default_method(const glow::Instruction *Inst) {}
  virtual void default_method(glow::Instruction *Inst) {}

#define DEF_METHOD(CLASS)                                                      \
  virtual void visit(const CLASS *Inst) { default_method(Inst); }              \
  virtual void visit(CLASS *Inst) { default_method(Inst); }

#define DEF_VALUE(CLASS, NAME)
#define DEF_INSTR(CLASS, NAME) DEF_METHOD(CLASS)
#define DEF_BACKEND_SPECIFIC_INSTR(CLASS, NAME) DEF_METHOD(CLASS)
#include "glow/AutoGenInstr.def"

#undef DEF_METHOD
};

} // namespace glow
