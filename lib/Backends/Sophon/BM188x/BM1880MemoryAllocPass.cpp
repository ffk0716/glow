#include "BM1880MemoryAllocPass.h"
#include "BM1880AllocationsInfo.h"
#include "Backends/Sophon/AllocationsInfo.h"
#include "glow/IR/IRBuilder.h"
#include "glow/Support/Debug.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <memory>
#include <string>

using namespace glow;

#define DEBUG_TYPE "BM1880_mem_alloc"

class MemoryAllocPass {
public:
  MemoryAllocPass(BM1880AllocationsInfo *allocInfo) : _allocInfo(allocInfo) {}

  void runOnIRFunction(IRFunction *F);

private:
  BM1880AllocationsInfo *_allocInfo;
};

void MemoryAllocPass::runOnIRFunction(IRFunction *F) {
  DEBUG_GLOW(llvm::dbgs() << "MemoryAllocPass::runOnIRFunction\n");
  _allocInfo->allocateWeightVars(F);
  _allocInfo->allocateActivations(F);
}

namespace glow {
namespace sophon {
void runMemoryAllocPass(IRFunction *F, BM1880AllocationsInfo *allocInfo) {
  auto p = llvm::make_unique<MemoryAllocPass>(allocInfo);
  p->runOnIRFunction(F);
}

} // namespace sophon
} // namespace glow
