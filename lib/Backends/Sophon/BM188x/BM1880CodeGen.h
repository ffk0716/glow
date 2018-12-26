#ifndef BM1880_CODEGEN_H
#define BM1880_CODEGEN_H

#include "Backends/Sophon/AllocationsInfo.h"
#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Debug.h"
#include <memory>

namespace glow {

class BM1880CodeGen {
public:
  virtual void performCodeGen() = 0;
  virtual std::vector<uint8_t> getCmdbuf() = 0;
  static std::unique_ptr<BM1880CodeGen>
  createCodeGen(IRFunction *F, AllocationsInfo &allocInfo);
};

} // namespace glow

#endif // BM1880_CODEGEN_H
