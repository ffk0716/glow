#pragma once

#include "Backends/Sophon/BM188x/BM1880AllocationsInfo.h"
#include "glow/IR/IR.h"

namespace glow {
namespace sophon {
void runMemoryAllocPass(IRFunction *F, BM1880AllocationsInfo *allocInfo);
}
} // namespace glow
