#ifndef BM1880_CODEGEN_BMK_H
#define BM1880_CODEGEN_BMK_H

#include "BM1880CodeGen.h"
#include "Backends/Sophon/GlowLIRVisitor.h"
#include <bmkernel/bm1880/bmkernel_1880.h>

namespace glow {

class BM1880CodeGenBMK : public BM1880CodeGen, public GlowLIRVisitor {
public:
  BM1880CodeGenBMK(IRFunction *F, AllocationsInfo &allocInfo);
  void visit(const SophonMIMulConstQ8Inst *inst) override;
  void visit(const SophonMIMulConstQ16Inst *inst) override;
  void visit(const SophonMIMacConstQ8Inst *inst) override;
  void visit(const SophonMIReluQ8Inst *inst) override;
  void visit(const SophonMIAvgPoolingQ8Inst *inst) override;
  void visit(const SophonMIMaxPoolingQ8Inst *inst) override;
  void visit(const SophonMIConvolutionQ8Inst *inst) override;
  void visit(const SophonMIDepthwiseConvolutionQ8Inst *inst) override;
  void visit(const SophonMIFCQ16Inst *inst) override;
  void visit(const SophonMIFCQ8Inst *inst) override;
  void visit(const SophonMIGDMAGlobalToLocalInst *inst) override;
  void visit(const SophonMIGDMALocalToGlobalInst *inst) override;

  void visit(const AllocActivationInst *inst) { index++; }
  void visit(const DeallocActivationInst *inst) { index++; }
  void default_method(glow::Instruction *Inst) {
    llvm_unreachable("Unknown value kind");
  }
  void performCodeGen() override;
  std::vector<uint8_t> getCmdbuf() override;

private:
  template <class T> void bmk_matrix_mac(const T *inst, bool res_is_int8);
  void bmk_init();
  void bmk_deinit();
  uint64_t emitValueAddress(const glow::Value *val);

private:
  bmk1880_context_t *bmk_ctx_;
  bmk_info_t bmk_info_;
  AllocationsInfo &allocInfo_;
  const IRFunction *F_;
  std::vector<uint8_t> cmdbuf_;
  int cmdbuf_size_;
  int index{0};
};

} // namespace glow

#endif // BM1880_CODEGEN_BMK_H
