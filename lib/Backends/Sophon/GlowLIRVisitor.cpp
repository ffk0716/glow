#define DEBUG_TYPE "visitor"
#include "GlowLIRVisitor.h"
#include "glow/Support/Debug.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

using namespace glow;

#define VISIT_INSTR(CLASS)                                                     \
  case glow::Kinded::Kind::CLASS##Kind: {                                      \
    this->visit(llvm::cast<glow::CLASS>(Inst));                                \
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
