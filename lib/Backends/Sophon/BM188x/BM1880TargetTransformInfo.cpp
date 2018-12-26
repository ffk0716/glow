#include "BM1880TargetTransformInfo.h"
#include "BM188xLMemSizeVisitor.h"
#include <iostream>

namespace glow {
namespace sophon {
BM1880TargetTransformInfo *BM1880TargetTransformInfo::getInstance() {
  static BM1880TargetTransformInfo instance;
  return &instance;
}

BM1880TargetTransformInfo::BM1880TargetTransformInfo() {}

int BM1880TargetTransformInfo::getLocalMemSizeInBytes() const {
  return 64 * 1024;
}

int BM1880TargetTransformInfo::getTPUNum() const { return 1; }

int BM1880TargetTransformInfo::getNPUNum() const { return 32; }

int BM1880TargetTransformInfo::getEUNum() const { return 16; }

size_t
BM1880TargetTransformInfo::getLMemSizeFromValue(glow::Value *value) const {
  BM188xLMemSizeVisitor visitor;
  size_t lmem_size = 0;

  for (auto &use : value->getUsers()) {

    auto *instr = use.get();

    visitor.accept_helper(instr);
    auto opnd_size = visitor.getResult();
    auto opnd_idx = use.idx_;

    if (opnd_idx < opnd_size.size()) {
      lmem_size = opnd_size.at(opnd_idx);
    }

    if (lmem_size > 0) {
      break;
    }
  }
  return lmem_size;
}

// only conv.weight, conv.bias, fc.bias can be eu-unaligned
bool BM1880TargetTransformInfo::isEUAligned(
    const glow::AllocActivationInst *Inst) const {

  for (const auto &use : Inst->getUsers()) {
    const auto *instr = use.get();
    // try to find input user, not output user
    if (use.getOperand().second == OperandKind::Out)
      continue;

    if (auto *conv =
            llvm::dyn_cast<const glow::SophonConvolutionQ8Inst>(instr)) {
      // conv.weight or conv.bias
      if (use.idx_ == 2 || use.idx_ == 3)
        return false;
      return true;
    } else if (auto *fc =
                   llvm::dyn_cast<const glow::SophonFullyConnectedQ8Inst>(
                       instr)) {
      // fc.bias
      if (use.idx_ == 3)
        return false;
      return true;
    }
  }
  return true;
}

std::vector<size_t>
BM1880TargetTransformInfo::getLMemSizeFromInst(glow::Instruction *Inst) const {
  BM188xLMemSizeVisitor visitor;
  visitor.accept_helper(Inst);
  return visitor.getResult();
}

} // namespace sophon
} // namespace glow
