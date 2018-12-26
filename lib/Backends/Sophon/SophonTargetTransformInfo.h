#ifndef SOPHON_TARGET_TRANSFORM_INFO_H
#define SOPHON_TARGET_TRANSFORM_INFO_H

#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include <vector>

namespace glow {
namespace sophon {

class SophonTargetTransformInfo {
public:
  virtual size_t getLMemSizeFromValue(glow::Value *value) const { return 0; }
  virtual bool isEUAligned(const glow::AllocActivationInst *Inst) const {
    return true;
  }
  virtual std::vector<size_t>
  getLMemSizeFromInst(glow::Instruction *Inst) const {
    return std::vector<size_t>();
  }
  virtual int getLocalMemSizeInBytes() const { return 0; }
  virtual int getTPUNum() const { return 0; }
  virtual int getNPUNum() const { return 0; }
  virtual int getEUNum() const { return 0; }
};
} // namespace sophon

} // namespace glow
#endif // SOPHON_TARGET_TRANSFORM_INFO_H
