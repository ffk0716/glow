#pragma once

#include "Backends/Sophon/BM188x/BM1880AllocationsInfo.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"

namespace glow {
class BM1880ExpandSophonInst {

public:
  BM1880ExpandSophonInst(IRFunction *F,
                         const BM1880AllocationsInfo &allocationsInfo)
      : F_(F), allocationsInfo_(allocationsInfo) {}
  void run();

private:
  IRFunction *F_;
  const BM1880AllocationsInfo &allocationsInfo_;
};
} // namespace glow
