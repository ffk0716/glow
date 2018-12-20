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
