#include "Backends/Sophon/BM188x/BM1880AllocationsInfo.h"
#include "Backends/Sophon/BM188x/BM1880Backend.h"
#include "Backends/Sophon/BM188x/BM1880ExpandSophonInst.h"
#include "Backends/Sophon/BM188x/BM1880InsertLoadStorePass.h"
#include "glow/Base/Type.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"
#include "gtest/gtest.h"

#include "llvm/Support/Casting.h"

using namespace glow;

TEST(BM1880ExpandTest, ExpandConvQ8) {
  Module mod;
  Function *F = mod.createFunction("TestLIRConv");
  auto IR = llvm::make_unique<IRFunction>(F);
  IRBuilder bb(IR.get());

  // N, C, H, W
  auto *input = bb.createWeightVar(glow::ElemKind::Int8QTy, {1, 1, 3, 3}, 1, 0,
                                   "input", WeightVar::MutabilityKind::Mutable);

  // IC, OC, KH, KW
  auto *conv_w =
      bb.createWeightVar(glow::ElemKind::Int8QTy, {3, 1, 3, 3}, 1, 0, "conv1.w",
                         WeightVar::MutabilityKind::Constant);

  // OC
  auto *conv_b =
      bb.createWeightVar(glow::ElemKind::Int16QTy, {3}, 1, 0, "conv1.b",
                         WeightVar::MutabilityKind::Constant);

  // N, C, H, W
  auto *output =
      bb.createWeightVar(glow::ElemKind::Int8QTy, {1, 3, 1, 1}, 1, 0, "output1",
                         WeightVar::MutabilityKind::Mutable);

  auto *lmem_in = bb.createAllocActivationInst("lmem.in", input->getType());
  auto *lmem_weight =
      bb.createAllocActivationInst("lmem.weight", conv_w->getType());
  auto *lmem_bias =
      bb.createAllocActivationInst("lmem.bias", conv_b->getType());
  auto *lmem_out = bb.createAllocActivationInst("lmem.out", output->getType());

  unsigned int pad_top, pad_left, pad_bottom, pad_right;
  unsigned int dilation_h, dilation_w;
  unsigned int stride_h, stride_w;
  int rshift_width;
  bool enable_relu = false;
  std::vector<unsigned_t> depends;

  pad_top = pad_left = pad_bottom = pad_right = 0;
  dilation_h = dilation_w = 1;
  stride_h = stride_w = 1;
  rshift_width = 0;

  bb.createSophonConvolutionQ8Inst(
      "conv1", lmem_out, lmem_in, lmem_weight, lmem_bias, {stride_h, stride_w},
      {pad_top, pad_left, pad_bottom, pad_right}, {dilation_h, dilation_w},
      rshift_width, enable_relu);

  bb.createDeallocActivationInst("dealloc1", lmem_in);
  bb.createDeallocActivationInst("dealloc2", lmem_weight);
  bb.createDeallocActivationInst("dealloc3", lmem_bias);
  bb.createDeallocActivationInst("dealloc4", lmem_out);

  BM1880AllocationsInfo tmp;
  BM1880ExpandSophonInst(IR.get(), tmp).run();

  auto cur_inst = IR->getInstrs().begin();
  std::advance(cur_inst, 4);
  EXPECT_EQ(cur_inst->getKind(),
            glow::Kinded::Kind::SophonMIConvolutionQ8InstKind);
  auto *MI_inst = llvm::cast<SophonMIConvolutionQ8Inst>(&*cur_inst);
  EXPECT_EQ(MI_inst->getDest()->getName(), lmem_out->getName());
  EXPECT_EQ(MI_inst->getSrc()->getName(), lmem_in->getName());
  EXPECT_EQ(MI_inst->getFilter()->getName(), lmem_weight->getName());
  EXPECT_EQ(MI_inst->getBias()->getName(), lmem_bias->getName());
  EXPECT_EQ(MI_inst->getStrideHW(),
            (llvm::ArrayRef<unsigned>{stride_h, stride_w}));
  EXPECT_EQ(
      MI_inst->getPadTLBR(),
      (llvm::ArrayRef<unsigned>{pad_top, pad_left, pad_bottom, pad_right}));
  EXPECT_EQ(MI_inst->getDilationHW(),
            (llvm::ArrayRef<unsigned>{dilation_h, dilation_w}));
  EXPECT_EQ(MI_inst->getRShiftWidth(), rshift_width);
  EXPECT_EQ(MI_inst->getEnableRelu(), enable_relu);
}

TEST(BM1880ExpandTest, ExpandMaxPoolingQ8) {
  Module mod;
  Function *F = mod.createFunction("TestMaxPool");
  auto IR = llvm::make_unique<IRFunction>(F);
  IRBuilder bb(IR.get());

  // N, C, H, W
  auto *input = bb.createWeightVar(glow::ElemKind::Int8QTy, {1, 1, 4, 4}, 1, 0,
                                   "input", WeightVar::MutabilityKind::Mutable);

  // N, C, H, W
  auto *output =
      bb.createWeightVar(glow::ElemKind::Int8QTy, {1, 1, 2, 2}, 1, 0, "output1",
                         WeightVar::MutabilityKind::Mutable);

  auto *lmem_in = bb.createAllocActivationInst("lmem.in", input->getType());
  auto *lmem_out = bb.createAllocActivationInst("lmem.out", output->getType());

  uint32_t kernel_h, kernel_w;
  uint32_t stride_h, stride_w;
  uint32_t pad_top, pad_left, pad_bottom, pad_right;
  uint32_t rshift_width;
  uint32_t multiplier;

  kernel_h = kernel_w = 2;
  stride_h = stride_w = 2;
  pad_top = pad_left = pad_bottom = pad_right = 0;
  rshift_width = 0;
  multiplier = 3;

  bb.createSophonMaxPoolQ8Inst(
      "pool", lmem_out, lmem_in, {kernel_h, kernel_w}, {stride_h, stride_w},
      {pad_top, pad_left, pad_bottom, pad_right}, rshift_width, multiplier);

  bb.createDeallocActivationInst("dealloc1", lmem_in);
  bb.createDeallocActivationInst("dealloc4", lmem_out);

  BM1880AllocationsInfo tmp;
  BM1880ExpandSophonInst(IR.get(), tmp).run();

  auto cur_inst = IR->getInstrs().begin();
  std::advance(cur_inst, 2);
  EXPECT_EQ(cur_inst->getKind(),
            glow::Kinded::Kind::SophonMIMaxPoolingQ8InstKind);
  auto *pool_inst = llvm::cast<SophonMIMaxPoolingQ8Inst>(&*cur_inst);
  EXPECT_EQ(pool_inst->getDest()->getName(), lmem_out->getName());
  EXPECT_EQ(pool_inst->getSrc()->getName(), lmem_in->getName());
  EXPECT_EQ(pool_inst->getKernelHW(),
            (llvm::ArrayRef<unsigned>{kernel_h, kernel_w}));
  EXPECT_EQ(pool_inst->getStrideHW(),
            (llvm::ArrayRef<unsigned>{stride_h, stride_w}));
  EXPECT_EQ(
      pool_inst->getPadTLBR(),
      (llvm::ArrayRef<unsigned>{pad_top, pad_left, pad_bottom, pad_right}));
  std::advance(cur_inst, 1);
  EXPECT_EQ(cur_inst->getKind(),
            glow::Kinded::Kind::SophonMIMulConstQ8InstKind);
  auto *mul_inst = llvm::cast<SophonMIMulConstQ8Inst>(&*cur_inst);
  EXPECT_EQ(mul_inst->getDest()->getName(), lmem_out->getName());
  EXPECT_EQ(mul_inst->getSrc()->getName(), lmem_out->getName());
  EXPECT_EQ(mul_inst->getMultiplier(), multiplier);
  EXPECT_EQ(mul_inst->getIsMultiplierSigned(), 0);
  EXPECT_EQ(mul_inst->getRShiftWidth(), rshift_width);
}

TEST(BM1880ExpandTest, ExpandReluQ8) {
  Module mod;
  Function *F = mod.createFunction("TestRelu");
  auto IR = llvm::make_unique<IRFunction>(F);
  IRBuilder bb(IR.get());

  // N, C, H, W
  auto *input = bb.createWeightVar(glow::ElemKind::Int8QTy, {1, 1, 4, 4}, 1, 0,
                                   "input", WeightVar::MutabilityKind::Mutable);

  // N, C, H, W
  auto *output =
      bb.createWeightVar(glow::ElemKind::Int8QTy, {1, 1, 4, 4}, 1, 0, "output1",
                         WeightVar::MutabilityKind::Mutable);

  auto *lmem_in = bb.createAllocActivationInst("lmem.in", input->getType());
  auto *lmem_out = bb.createAllocActivationInst("lmem.out", output->getType());
  bb.createSophonReluQ8Inst("relu", lmem_out, lmem_in);

  bb.createDeallocActivationInst("dealloc1", lmem_in);
  bb.createDeallocActivationInst("dealloc4", lmem_out);

  BM1880AllocationsInfo tmp;
  BM1880ExpandSophonInst(IR.get(), tmp).run();

  auto cur_inst = IR->getInstrs().begin();
  std::advance(cur_inst, 2);
  EXPECT_EQ(cur_inst->getKind(), glow::Kinded::Kind::SophonMIReluQ8InstKind);
  auto *relu_inst = llvm::cast<SophonMIReluQ8Inst>(&*cur_inst);
  EXPECT_EQ(relu_inst->getDest()->getName(), lmem_out->getName());
  EXPECT_EQ(relu_inst->getSrc()->getName(), lmem_in->getName());
}

TEST(BM1880ExpandTest, ExpandFcQ8) {
  Module mod;
  Function *F = mod.createFunction("TestFc");
  auto IR = llvm::make_unique<IRFunction>(F);
  IRBuilder bb(IR.get());

  // in_row * in_col
  auto *input = bb.createWeightVar(glow::ElemKind::Int8QTy, {3, 4}, 1, 0,
                                   "input", WeightVar::MutabilityKind::Mutable);

  // in_col * out_col
  auto *weight =
      bb.createWeightVar(glow::ElemKind::Int8QTy, {4, 2}, 1, 0, "weight",
                         WeightVar::MutabilityKind::Constant);

  // bias
  auto *bias = bb.createWeightVar(glow::ElemKind::Int16QTy, {2}, 1, 0, "bias",
                                  WeightVar::MutabilityKind::Constant);
  // in_row * out_col
  auto *output =
      bb.createWeightVar(glow::ElemKind::Int8QTy, {3, 2}, 1, 0, "output1",
                         WeightVar::MutabilityKind::Mutable);

  auto *lmem_in = bb.createAllocActivationInst("lmem.in", input->getType());
  auto *lmem_weights =
      bb.createAllocActivationInst("lmem.weights", weight->getType());
  auto *lmem_bias = bb.createAllocActivationInst("lmem.bias", bias->getType());
  auto *lmem_out = bb.createAllocActivationInst("lmem.out", output->getType());

  bool relu = false;
  uint32_t rshift_width = 5;
  uint32_t lshift_width = 4;
  bool result_add = false;

  bb.createSophonFullyConnectedQ8Inst("fc", lmem_out, lmem_in, lmem_weights,
                                      lmem_bias, relu, rshift_width,
                                      lshift_width, result_add);

  bb.createDeallocActivationInst("dealloc1", lmem_in);
  bb.createDeallocActivationInst("dealloc2", lmem_weights);
  bb.createDeallocActivationInst("dealloc3", lmem_bias);
  bb.createDeallocActivationInst("dealloc4", lmem_out);

  BM1880AllocationsInfo tmp;
  BM1880ExpandSophonInst(IR.get(), tmp).run();

  auto cur_inst = IR->getInstrs().begin();
  std::advance(cur_inst, 4);
  EXPECT_EQ(cur_inst->getKind(), glow::Kinded::Kind::SophonMIFCQ8InstKind);
  auto *fc_inst = llvm::cast<SophonMIFCQ8Inst>(&*cur_inst);
  EXPECT_EQ(fc_inst->getDest()->getName(), lmem_out->getName());
  EXPECT_EQ(fc_inst->getSrc()->getName(), lmem_in->getName());
  EXPECT_EQ(fc_inst->getFilter()->getName(), lmem_weights->getName());
  EXPECT_EQ(fc_inst->getBias()->getName(), lmem_bias->getName());
  EXPECT_EQ(fc_inst->getRShiftWidth(), rshift_width);
  // backend hack lshift value as 3
  EXPECT_EQ(fc_inst->getLShiftWidth(), 3);
  EXPECT_EQ(fc_inst->getResultAdd(), result_add);
}

TEST(BM1880ExpandTest, ExpandFcReluQ8) {
  Module mod;
  Function *F = mod.createFunction("TestFc");
  auto IR = llvm::make_unique<IRFunction>(F);
  IRBuilder bb(IR.get());

  // in_row * in_col
  auto *input = bb.createWeightVar(glow::ElemKind::Int8QTy, {3, 4}, 1, 0,
                                   "input", WeightVar::MutabilityKind::Mutable);

  // in_col * out_col
  auto *weight =
      bb.createWeightVar(glow::ElemKind::Int8QTy, {4, 2}, 1, 0, "weight",
                         WeightVar::MutabilityKind::Constant);

  // bias
  auto *bias = bb.createWeightVar(glow::ElemKind::Int16QTy, {2}, 1, 0, "bias",
                                  WeightVar::MutabilityKind::Constant);
  // in_row * out_col
  auto *output =
      bb.createWeightVar(glow::ElemKind::Int8QTy, {3, 2}, 1, 0, "output1",
                         WeightVar::MutabilityKind::Mutable);

  auto *lmem_in = bb.createAllocActivationInst("lmem.in", input->getType());
  auto *lmem_weights =
      bb.createAllocActivationInst("lmem.weights", weight->getType());
  auto *lmem_bias = bb.createAllocActivationInst("lmem.bias", bias->getType());
  auto *lmem_out = bb.createAllocActivationInst("lmem.out", output->getType());

  bool relu = true;
  uint32_t rshift_width = 5;
  uint32_t lshift_width = 4;
  bool result_add = false;

  bb.createSophonFullyConnectedQ8Inst("fc", lmem_out, lmem_in, lmem_weights,
                                      lmem_bias, relu, rshift_width,
                                      lshift_width, result_add);

  bb.createDeallocActivationInst("dealloc1", lmem_in);
  bb.createDeallocActivationInst("dealloc2", lmem_weights);
  bb.createDeallocActivationInst("dealloc3", lmem_bias);
  bb.createDeallocActivationInst("dealloc4", lmem_out);

  BM1880AllocationsInfo tmp;
  BM1880ExpandSophonInst(IR.get(), tmp).run();

  auto cur_inst = IR->getInstrs().begin();
  std::advance(cur_inst, 4);
  EXPECT_EQ(cur_inst->getKind(), glow::Kinded::Kind::SophonMIFCQ8InstKind);
  auto *fc_inst = llvm::cast<SophonMIFCQ8Inst>(&*cur_inst);
  EXPECT_EQ(fc_inst->getDest()->getName(), lmem_out->getName());
  EXPECT_EQ(fc_inst->getSrc()->getName(), lmem_in->getName());
  EXPECT_EQ(fc_inst->getFilter()->getName(), lmem_weights->getName());
  EXPECT_EQ(fc_inst->getBias()->getName(), lmem_bias->getName());
  EXPECT_EQ(fc_inst->getRShiftWidth(), rshift_width);
  // backend hack lshift value as 3
  EXPECT_EQ(fc_inst->getLShiftWidth(), 3);
  EXPECT_EQ(fc_inst->getResultAdd(), result_add);
  std::advance(cur_inst, 1);
  EXPECT_EQ(cur_inst->getKind(), glow::Kinded::Kind::SophonMIReluQ8InstKind);
}

TEST(BM1880ExpandTest, ExpandLoadQ8) {
  Module mod;
  Function *F = mod.createFunction("ExpandLoadStoreQ8");
  auto *input = mod.createPlaceholder(ElemKind::Int8QTy, {1, 1, 3, 3}, 1, 0,
                                      "input", false);
  auto *filter =
      mod.createConstant(ElemKind::Int8QTy, {3, 1, 3, 3}, 1, 0, "filter");
  auto *bias = mod.createConstant(ElemKind::Int16QTy, {3}, 1, 0, "bias");
  auto output_type = mod.uniqueType(ElemKind::Int8QTy, {1, 3, 3, 3}, 1, 0);

  auto *conv = F->addNode(
      new SophonConvolutionQ8Node("conv", output_type, input, filter, bias,
                                  {1, 1}, {1, 1, 1, 1}, {1, 1}, 0, false));
  auto *result = F->createSave("ret2", conv);
  result->getPlaceholder();

  auto IR = llvm::make_unique<IRFunction>(F);
  IR->generateIR();
  glow::optimize(*IR, true);
  sophon::runInsertLoadStorePass(IR.get());
  BM1880AllocationsInfo tmp;
  BM1880ExpandSophonInst(IR.get(), tmp).run();
  auto cur_inst = IR->getInstrs().begin();
  std::advance(cur_inst, 2);
  EXPECT_EQ(cur_inst->getKind(),
            glow::Kinded::Kind::SophonMIGDMAGlobalToLocalInstKind);
  auto *load_input = llvm::cast<SophonMIGDMAGlobalToLocalInst>(&*cur_inst);
  EXPECT_EQ(load_input->getShapeNCHW(), (llvm::ArrayRef<unsigned>{1, 1, 3, 3}));
  EXPECT_EQ(load_input->getGlobalStrideNCH(),
            (llvm::ArrayRef<unsigned>{1 * 3 * 3, 3 * 3, 3}));
  EXPECT_EQ(load_input->getIsGlobalWeightSpace(), false);
  EXPECT_EQ(load_input->getIsLocalAligned(), true);

  std::advance(cur_inst, 2);
  EXPECT_EQ(cur_inst->getKind(),
            glow::Kinded::Kind::SophonMIGDMAGlobalToLocalInstKind);
  auto *load_weight = llvm::cast<SophonMIGDMAGlobalToLocalInst>(&*cur_inst);
  EXPECT_EQ(load_weight->getShapeNCHW(),
            (llvm::ArrayRef<unsigned>{1, 3, 9, 1}));
  EXPECT_EQ(load_weight->getGlobalStrideNCH(),
            (llvm::ArrayRef<unsigned>{3 * 9, 9, 1}));
  EXPECT_EQ(load_weight->getIsGlobalWeightSpace(), true);
  EXPECT_EQ(load_weight->getIsLocalAligned(), false);

  std::advance(cur_inst, 2);
  EXPECT_EQ(cur_inst->getKind(),
            glow::Kinded::Kind::SophonMIGDMAGlobalToLocalInstKind);
  auto *load_bias = llvm::cast<SophonMIGDMAGlobalToLocalInst>(&*cur_inst);
  EXPECT_EQ(load_bias->getShapeNCHW(), (llvm::ArrayRef<unsigned>{2, 3, 1, 1}));
  EXPECT_EQ(load_bias->getGlobalStrideNCH(),
            (llvm::ArrayRef<unsigned>{3 * 1 * 1, 1 * 1, 1}));
  EXPECT_EQ(load_bias->getIsGlobalWeightSpace(), true);
  EXPECT_EQ(load_bias->getIsLocalAligned(), false);

  std::advance(cur_inst, 2);
  EXPECT_EQ(cur_inst->getKind(),
            glow::Kinded::Kind::SophonMIGDMALocalToGlobalInstKind);
  auto *store = llvm::cast<SophonMIGDMALocalToGlobalInst>(&*cur_inst);
  EXPECT_EQ(store->getShapeNCHW(), (llvm::ArrayRef<unsigned>{1, 3, 3, 3}));
  EXPECT_EQ(store->getGlobalStrideNCH(),
            (llvm::ArrayRef<unsigned>{3 * 3 * 3, 3 * 3, 3}));
  EXPECT_EQ(store->getIsGlobalWeightSpace(), false);
}

TEST(BM1880ExpandTest, ExpandLoadStoreFcQ8) {
  Module mod;
  Function *F = mod.createFunction("TestLIRFc");

  auto *input = mod.createPlaceholder(glow::ElemKind::Int8QTy, {1, 800}, 1, 0,
                                      "input", false);

  auto *fc_right =
      mod.createConstant(glow::ElemKind::Int8QTy, {800, 500}, 1, 0, "fc_right");

  // bias
  auto *fc_b =
      mod.createConstant(glow::ElemKind::Int16QTy, {500}, 1, 0, "fc.b");

  TypeRef ty = mod.uniqueType(glow::ElemKind::Int8QTy, {1, 500}, 1, 0);
  auto *fc = F->addNode(new SophonFullyConnectedQ8Node(
      "fc", ty, input, fc_right, fc_b, false, 1, 2, false));

  auto *result = F->createSave("ret2", fc);

  auto IR = llvm::make_unique<IRFunction>(F);
  IR->generateIR();
  glow::optimize(*IR, true);
  sophon::runInsertLoadStorePass(IR.get());
  BM1880AllocationsInfo tmp;
  BM1880ExpandSophonInst(IR.get(), tmp).run();
  // TODO check result
  auto cur_inst = IR->getInstrs().begin();
  // load input
  std::advance(cur_inst, 2);
  EXPECT_EQ(cur_inst->getKind(),
            glow::Kinded::Kind::SophonMIGDMAGlobalToLocalInstKind);
  auto *load_input = llvm::cast<SophonMIGDMAGlobalToLocalInst>(&*cur_inst);
  EXPECT_EQ(load_input->getShapeNCHW(),
            (llvm::ArrayRef<unsigned>{1, 25, 1, 32})); // 25=800/32
  EXPECT_EQ(load_input->getGlobalStrideNCH(),
            (llvm::ArrayRef<unsigned>{25 * 1 * 32, 1 * 32, 32}));
  EXPECT_EQ(load_input->getIsGlobalWeightSpace(), false);
  EXPECT_EQ(load_input->getIsLocalAligned(), true);

  // loadr fc_right
  std::advance(cur_inst, 2);
  EXPECT_EQ(cur_inst->getKind(),
            glow::Kinded::Kind::SophonMIGDMAGlobalToLocalInstKind);
  auto *load_weight = llvm::cast<SophonMIGDMAGlobalToLocalInst>(&*cur_inst);
  EXPECT_EQ(load_weight->getShapeNCHW(),
            (llvm::ArrayRef<unsigned>{800, 16, 1, 32})); // 16=500/32
  EXPECT_EQ(load_weight->getGlobalStrideNCH(),
            (llvm::ArrayRef<unsigned>{500, 1 * 32, 32}));
  EXPECT_EQ(load_weight->getIsGlobalWeightSpace(), true);
  EXPECT_EQ(load_weight->getIsLocalAligned(), true);

  // load fc_b
  std::advance(cur_inst, 2);
  EXPECT_EQ(cur_inst->getKind(),
            glow::Kinded::Kind::SophonMIGDMAGlobalToLocalInstKind);
  auto *load_bias = llvm::cast<SophonMIGDMAGlobalToLocalInst>(&*cur_inst);
  EXPECT_EQ(load_bias->getShapeNCHW(),
            (llvm::ArrayRef<unsigned>{2, 16, 1, 32})); // 16=500/32
  EXPECT_EQ(load_bias->getGlobalStrideNCH(),
            (llvm::ArrayRef<unsigned>{500, 1 * 32, 32}));
  EXPECT_EQ(load_bias->getIsGlobalWeightSpace(), true);
  EXPECT_EQ(load_bias->getIsLocalAligned(), false);

  // store result
  std::advance(cur_inst, 2);
  EXPECT_EQ(cur_inst->getKind(),
            glow::Kinded::Kind::SophonMIGDMALocalToGlobalInstKind);
  // TODO add check
}
