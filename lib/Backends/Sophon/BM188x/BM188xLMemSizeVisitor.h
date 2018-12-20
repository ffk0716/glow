#ifndef BM188X_LMEMSIZE_VISITOR_H
#define BM188X_LMEMSIZE_VISITOR_H

#include "Backends/Sophon/GlowLIRVisitor.h"
#include "glow/IR/IR.h"

namespace glow {
namespace sophon {
class BM188xLMemSizeVisitor : public glow::GlowLIRVisitor {
public:
  BM188xLMemSizeVisitor();

  template <typename T> std::vector<size_t> calGeneralAligned(T *inst);
  size_t calAligned(glow::Value *value);
  size_t calNonAligned(glow::Value *value);
  size_t calFCBias(glow::Value *value);

  // GlowOp

  // SophonOp
  void visit(glow::SophonReluQ8Inst *Inst) override;
  void visit(glow::SophonAvgPoolQ8Inst *Inst) override;
  void visit(glow::SophonMaxPoolQ8Inst *Inst) override;
  void visit(glow::SophonConvolutionQ8Inst *Inst) override;
  void visit(glow::SophonFullyConnectedQ8Inst *Inst) override;
  void visit(glow::SophonLoadInst *Inst) override;
  void visit(glow::SophonStoreInst *Inst) override;

  // SophonMI
  void visit(glow::SophonMIGDMAGlobalToLocalInst *Inst) override;
  void visit(glow::SophonMIGDMALocalToGlobalInst *Inst) override;
  void visit(glow::SophonMIMulConstQ16Inst *Inst) override;
  void visit(glow::SophonMIMulConstQ8Inst *Inst) override;
  void visit(glow::SophonMIMacConstQ8Inst *Inst) override;
  void visit(glow::SophonMIReluQ8Inst *Inst) override;
  void visit(glow::SophonMIAvgPoolingQ8Inst *Inst) override;
  void visit(glow::SophonMIMaxPoolingQ8Inst *Inst) override;
  void visit(glow::SophonMIConvolutionQ8Inst *Inst) override;
  void visit(glow::SophonMIDepthwiseConvolutionQ8Inst *Inst) override;
  void visit(glow::SophonMIFCQ8Inst *Inst) override;
  void visit(glow::SophonMIFCQ16Inst *Inst) override;
  void default_method(glow::Instruction *Inst) override;
  std::vector<size_t> getResult();

private:
  template <typename T, typename Type, typename... Types>
  std::vector<size_t> calOperandList(T *inst, const Type &oprnd,
                                     const Types &... oprnds);
  template <typename T>
  void calOperand(T *inst, std::vector<size_t> *result, const int opnd_id,
                  bool aligned);
  template <int ID, typename T, typename Type, typename... Types>
  void calEach(T *inst, std::vector<size_t> *result, const Type &oprnd,
               const Types &... oprnds);
  template <int ID, typename T, typename Type>
  void calEach(T *inst, std::vector<size_t> *result, const Type &oprnd);

  // use input or in/outut user to decide which value to calculate size.
  size_t calValueSize(glow::Value *Value, bool calByOutUser = false);

private:
  const bool NONALIGN = false;
  const bool ALIGN = true;
  unsigned npu_num_;
  unsigned eu_num_;
  std::vector<size_t> oprnd_size_;
};

} // namespace sophon
} // namespace glow

#endif // BM188X_LMEMSIZE_VISITOR_H
