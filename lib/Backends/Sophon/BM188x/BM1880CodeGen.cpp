#include "BM1880CodeGenBMK.h"

namespace glow {
std::unique_ptr<BM1880CodeGen>
BM1880CodeGen::createCodeGen(IRFunction *F, AllocationsInfo &allocInfo) {
  return llvm::make_unique<BM1880CodeGenBMK>(F, allocInfo);
}
} // namespace glow
