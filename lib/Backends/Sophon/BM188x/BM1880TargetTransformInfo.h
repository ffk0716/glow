#ifndef BM1880_TARGET_TRANSFORM_INFO_H
#define BM1880_TARGET_TRANSFORM_INFO_H

#include "Backends/Sophon/SophonTargetTransformInfo.h"
namespace glow {
namespace sophon {

class BM1880TargetTransformInfo : public SophonTargetTransformInfo {
public:
  static BM1880TargetTransformInfo *getInstance();
  BM1880TargetTransformInfo(BM1880TargetTransformInfo const &) = delete;
  void operator=(BM1880TargetTransformInfo const &) = delete;

public:
  size_t getLMemSizeFromValue(glow::Value *value) const override;
  bool isEUAligned(const glow::AllocActivationInst *Inst) const override;

  std::vector<size_t>
  getLMemSizeFromInst(glow::Instruction *Inst) const override;

  int getLocalMemSizeInBytes() const override;
  int getTPUNum() const override;
  int getNPUNum() const override;
  int getEUNum() const override;

private:
  BM1880TargetTransformInfo();
};

} // namespace sophon
} // namespace glow

#endif // BM1880_TARGET_TRANSFORM_INFO_H
