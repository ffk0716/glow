#include "BM1880ExpandSophonInst.h"
#include "Backends/Sophon/GlowLIRVisitor.h"
#include "Backends/Sophon/Utility/memory.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/IRUtils.h"
#include "glow/Support/Debug.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "expandSophonInst"

using namespace glow;

class ExpandVisitor : public GlowLIRVisitor {
public:
  ExpandVisitor(IRFunction *F, const BM1880AllocationsInfo &allocationsInfo)
      : F_(F), allocationsInfo_(allocationsInfo), Builder_(F) {}
  ~ExpandVisitor() = default;
  void visit(SophonLoadInst *Inst) override;
  void visit(SophonStoreInst *Inst) override;
  void visit(SophonConvolutionQ8Inst *Inst) override;
  void visit(SophonMaxPoolQ8Inst *Inst) override;
  void visit(SophonFullyConnectedQ8Inst *Inst) override;
  void visit(SophonReluQ8Inst *Inst) override;
  void visit(AllocActivationInst *Inst) override {}
  void visit(DeallocActivationInst *Inst) override {}

  void default_method(glow::Instruction *Inst) override;

private:
  // load constnat or placeholder
  bool isConstant(glow::SophonLoadInst *Inst);
  bool isLocalAligned(glow::SophonLoadInst *Inst);

private:
  IRFunction *F_;
  const BM1880AllocationsInfo &allocationsInfo_;
  IRBuilder Builder_;
  std::vector<glow::Instruction *> oldInsts_;
};

void ExpandVisitor::default_method(glow::Instruction *Inst) {
  DEBUG_GLOW(llvm::dbgs() << "unsupport ExpandVisitor " << Inst->getName()
                          << "\n");
  llvm_unreachable("TODO!");
}

// load constant weight or not
bool ExpandVisitor::isConstant(SophonLoadInst *Inst) {
  auto *W = llvm::cast<WeightVar>(getOrigin(Inst->getSrc()));
  if (W->getMutability() == WeightVar::MutabilityKind::Constant)
    return true;
  return false;
}

static std::vector<unsigned> getStride(const std::vector<unsigned> &Dims) {
  assert(Dims.size() == 4);
  return {Dims[1] * Dims[2] * Dims[3], Dims[2] * Dims[3], Dims[3]};
}

static void setGlobalStride(const std::vector<unsigned> &GlobalDims,
                            const std::vector<unsigned> &LocalDims,
                            std::vector<unsigned> &GlobalStrideNCH) {
  auto sg = getStride(GlobalDims);
  auto sl = getStride(LocalDims);
  GlobalStrideNCH = {std::min(sg[0], sl[0]), std::min(sg[1], sl[1]),
                     std::min(sg[2], sl[2])};
}

static void setDefautByDims(const std::vector<unsigned> &Dims,
                            std::vector<unsigned> &ShapeNCHW,
                            std::vector<unsigned> &GlobalStrideNCH) {
  assert(Dims.size() == 4);
  ShapeNCHW = {Dims[0], Dims[1], Dims[2], Dims[3]};
  GlobalStrideNCH = {Dims[1] * Dims[2] * Dims[3], Dims[2] * Dims[3], Dims[3]};
}

static void setDefaultChannelWise(unsigned Channel,
                                  std::vector<unsigned> &ShapeNCHW,
                                  std::vector<unsigned> &GlobalStrideNCH) {
  ShapeNCHW = {2, Channel, 1, 1};
  GlobalStrideNCH = {Channel, 1, 1};
}

static bool initConvLoadInfo(unsigned Idx, const std::vector<unsigned> &Dims,
                             bool &IsLocalAligned,
                             std::vector<unsigned> &ShapeNCHW,
                             std::vector<unsigned> &GlobalStrideNCH) {
  bool is_constant;
  switch (Idx) {
  case 1:
    // input
    IsLocalAligned = true;
    is_constant = false;
    setDefautByDims(Dims, ShapeNCHW, GlobalStrideNCH);
    break;
  case 2: {
    // weight
    IsLocalAligned = false;
    is_constant = true;
    // see plat-bm188x/bmkernel/conv_parallel_bmkernel.cpp +238
    // (1, oc, kh*kw, ic)
    unsigned oc = Dims[0];
    unsigned ic = Dims[1];
    ShapeNCHW = {1, oc, Dims[2] * Dims[3], ic};
    // see plat-bm188x/bmkernel/conv_parallel_bmkernel.cpp +298
    // (oc*kh*kw*ic, kh*kw*ic, ic)
    GlobalStrideNCH = {oc * Dims[2] * Dims[3] * ic, Dims[2] * Dims[3] * ic, ic};
    break;
  }
  case 3:
    // bias
    IsLocalAligned = false;
    is_constant = true;
    setDefaultChannelWise(Dims[0], ShapeNCHW, GlobalStrideNCH);
    break;
  default:
    llvm_unreachable("TODO!");
  }
  return is_constant;
}

// FIXME

// Implicit W parameter Rule:
//   This is based on bmkernel implementation.
//   When W is not provided with 2D tensor, we use the algorithm to decide W.
template <typename T>
static bool get_hw_dim(const T &vec, unsigned *n, unsigned *c, unsigned *h,
                       unsigned *w) {
  bool ret = false;
  size_t dim = vec.size();
  switch (dim) {
  case 4:
    *n = vec[0];
    *c = vec[1];
    *h = vec[2];
    *w = vec[3];
    ret = true;
    break;
  case 3:
    assert(false && "Not support Dimension = 3");
    ret = false;
    break;
  case 2: {
    unsigned M = vec[0];
    unsigned N = vec[1];
    if (N > 32) {
      *w = 32;
    } else {
      *w = 16;
    }
    *n = M;
    *h = 1;
    *c = sophon::idiv_round(N, *w);
    ret = true;
  } break;
  case 1:
    *n = 1;
    *c = vec[0];
    *h = 1;
    *w = 1;
    ret = true;
    break;
  default:
    assert(false && "Dimension is not between 1~4");
    ret = false;
    break;
  }
  return ret;
}

static bool initFCLoadInfo(unsigned Idx, const std::vector<unsigned> &Dims,
                           std::vector<unsigned> &ShapeNCHW,
                           std::vector<unsigned> &GlobalStrideNCH,
                           bool &is_local_aligned) {
  bool is_constant;
  if (Idx == 0 or Idx == 1 or Idx == 2) {
    // Idx 0/1 is input
    is_constant = Idx <= 1 ? false : true;
    unsigned n, c, h, w;
    get_hw_dim<std::vector<unsigned>>(Dims, &n, &c, &h, &w);
    ShapeNCHW = {n, c, h, w};
    assert(Dims.size() == 2);
    setGlobalStride({Dims[0], 1, 1, Dims[1]}, ShapeNCHW, GlobalStrideNCH);
    is_local_aligned = true;
  } else if (Idx == 3) {
    // bias
    is_constant = true;
    // Implicit W parameter Rule:
    //   This is based on bmkernel implementation.
    //   When W is not provided with 2D tensor, we use the algorithm to decide
    //   W.
    unsigned w;
    assert(Dims.size() == 1);
    unsigned dim = Dims[0];
    if (dim > 32) {
      w = 32;
    } else {
      w = 16;
    }
    unsigned c = sophon::idiv_round(dim, w);
    ShapeNCHW = {2, c, 1, w};
    setGlobalStride({1, 1, 1, dim}, ShapeNCHW, GlobalStrideNCH);
    is_local_aligned = false;
  } else
    llvm_unreachable("TODO!");
  return is_constant;
}

void ExpandVisitor::visit(SophonLoadInst *Inst) {
  // operands of SophonMIGDMAGlobalToLocalInst
  std::vector<unsigned> shape_NCHW;
  std::vector<unsigned> global_stride_NCH;
  bool is_local_aligned;
  bool is_constant;

  auto type = Inst->getDest()->getType();
  std::vector<unsigned> dims{type->dims().begin(), type->dims().end()};
  auto users = Inst->getDest()->getUsers();
  // check user InstKind to init above operands
  for (auto &user : users) {
    if (user.getOperand().second == OperandKind::Out)
      continue;
    switch (user.get()->getKind()) {
    case glow::Kinded::Kind::SophonConvolutionQ8InstKind:
      is_constant = initConvLoadInfo(user.idx_, dims, is_local_aligned,
                                     shape_NCHW, global_stride_NCH);
      break;
    case glow::Kinded::Kind::SophonFullyConnectedQ8InstKind:
      is_constant = initFCLoadInfo(user.idx_, dims, shape_NCHW,
                                   global_stride_NCH, is_local_aligned);
      break;
#if 0
  case glow::Kinded::Kind::SophonWinograndInstKind:
  case glow::Kinded::Kind::SophonDepthwiseInstKind:
    is_local_aligned = user.idx_ == 1? false : true;
    is_constant = user.idx_ ==1 ? true : false;
    if (user.idx_ == 1) {
      setDefautByDims(dims, shape_NCHW, global_stride_NCH);
    } else {
      setDefaultChannelWise(dims[0], shape_NCHW, global_stride_NCH);
    }
  case glow::Kinded::Kind::SophonArithmeticInstKind:
    is_local_aligned = true;
    is_constant = isConstant(Inst);
    setDefautByDims(dims, shape_NCHW, global_stride_NCH);
#endif
    default: // input/output neruon
      setDefautByDims(dims, shape_NCHW, global_stride_NCH);
      is_constant = false;
      is_local_aligned = true;
      DEBUG_GLOW(llvm::dbgs() << "ExpandSophonInst for "
                              << user.get()->getName() << " Inst\n");
    }
  }

  auto newInst = Builder_.createSophonMIGDMAGlobalToLocalInst(
      Inst->getName(), Inst->getDest(), Inst->getSrc(), shape_NCHW,
      global_stride_NCH, is_constant, is_local_aligned);

  F_->moveInstruction(Inst, newInst);
  DEBUG_GLOW(llvm::dbgs() << "visit " << Inst->getName() << "\n");
  F_->eraseInstruction(Inst);
}

void ExpandVisitor::visit(SophonStoreInst *Inst) {
  auto type = Inst->getSrc()->getType();
  std::vector<unsigned> dims{type->dims().begin(), type->dims().end()};
  bool is_local_aligned;
  std::vector<unsigned> shape_NCHW;
  std::vector<unsigned> global_stride_NCH;

  auto users = Inst->getSrc()->getUsers();
  for (auto &user : users) {
    // store's user is out MI user
    if (user.getOperand().second != OperandKind::Out)
      continue;
    switch (user.get()->getKind()) {
    case glow::Kinded::Kind::SophonMIFCQ8InstKind:
      initFCLoadInfo(user.idx_, dims, shape_NCHW, global_stride_NCH,
                     is_local_aligned);
      break;
    case glow::Kinded::Kind::DeallocActivationInstKind:
      // ignore user deallocInst
      break;
    default:
      is_local_aligned = true;
      shape_NCHW = {dims.begin(), dims.end()};
      global_stride_NCH = {shape_NCHW[1] * shape_NCHW[2] * shape_NCHW[3],
                           shape_NCHW[2] * shape_NCHW[3], shape_NCHW[3]};
      break;
    }
  }
  auto *new_inst = Builder_.createSophonMIGDMALocalToGlobalInst(
      Inst->getName(), Inst->getDest(), Inst->getSrc(), shape_NCHW,
      global_stride_NCH, false /*IsGlobalWeightSpace*/, is_local_aligned);
  F_->moveInstruction(Inst, new_inst);
  DEBUG_GLOW(llvm::dbgs() << "visit " << Inst->getName() << "\n");
  F_->eraseInstruction(Inst);
}

void ExpandVisitor::visit(SophonConvolutionQ8Inst *Inst) {
  std::vector<unsigned_t> depends;
  auto *newInst = Builder_.createSophonMIConvolutionQ8Inst(
      Inst->getName(), Inst->getDest(), Inst->getSrc(), Inst->getFilter(),
      Inst->getBias(), Inst->getStrideHW(), Inst->getPadTLBR(),
      Inst->getDilationHW(), Inst->getRShiftWidth(), Inst->getEnableRelu(), 0,
      0, depends);
  F_->moveInstruction(Inst, newInst);
  DEBUG_GLOW(llvm::dbgs() << "visit " << Inst->getName() << "\n");
  F_->eraseInstruction(Inst);
}

void ExpandVisitor::visit(SophonMaxPoolQ8Inst *Inst) {
  auto *newPool = Builder_.createSophonMIMaxPoolingQ8Inst(
      Inst->getName(), Inst->getDest(), Inst->getSrc(), Inst->getKernelHW(),
      Inst->getStrideHW(), Inst->getPadTLBR());

  auto *newMul = Builder_.createSophonMIMulConstQ8Inst(
      Inst->getName(), Inst->getDest(), Inst->getDest(), Inst->getMultiplier(),
      0, Inst->getRShiftWidth());
  F_->moveInstruction(Inst, newPool);
  F_->moveInstruction(Inst, newMul);
  DEBUG_GLOW(llvm::dbgs() << "visit " << Inst->getName() << "\n");
  F_->eraseInstruction(Inst);
}

void ExpandVisitor::visit(SophonFullyConnectedQ8Inst *Inst) {
  // hard code: lshift default is 3
  const int default_lshift = 3;
  auto *new_inst = Builder_.createSophonMIFCQ8Inst(
      Inst->getName(), Inst->getDest(), Inst->getSrc(), Inst->getWeights(),
      Inst->getBias(), Inst->getRShiftWidth(), default_lshift,
      Inst->getResultAdd());
  F_->moveInstruction(Inst, new_inst);
  if (Inst->getRelu()) {
    auto *relu = Builder_.createSophonMIReluQ8Inst(
        Inst->getName(), Inst->getDest(), Inst->getDest());
    F_->moveInstruction(Inst, relu);
  }
  DEBUG_GLOW(llvm::dbgs() << "visit " << Inst->getName() << "\n");
  F_->eraseInstruction(Inst);
}

void ExpandVisitor::visit(SophonReluQ8Inst *Inst) {
  auto *new_inst = Builder_.createSophonMIReluQ8Inst(
      Inst->getName(), Inst->getDest(), Inst->getSrc());
  F_->moveInstruction(Inst, new_inst);
  DEBUG_GLOW(llvm::dbgs() << "visit " << Inst->getName() << "\n");
  F_->eraseInstruction(Inst);
}

void BM1880ExpandSophonInst::run() {
  std::unique_ptr<GlowLIRVisitor> visitor(
      new ExpandVisitor(F_, allocationsInfo_));
  auto &instrs = F_->getInstrs();
  for (auto it = instrs.begin(), e = instrs.end(); it != e;) {
    auto &I = *it;
    it++;
    visitor->accept_helper(&I);
  }
}
