#include "Backends/Sophon/BM188x/BM1880TargetTransformInfo.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "gtest/gtest.h"

using namespace glow;
TEST(BM1880GetLMemSizeTest, LIRConv) {
  Module mod;
  Function *F = mod.createFunction("TestGetLMemSize");
  auto IR = llvm::make_unique<IRFunction>(F);
  IRBuilder bb(IR.get());

  auto q8 = [&](llvm::ArrayRef<size_t> dims) {
    return mod.uniqueType(glow::ElemKind::Int8QTy, dims, 1.0, 0);
  };
  auto q16 = [&](llvm::ArrayRef<size_t> dims) {
    return mod.uniqueType(glow::ElemKind::Int16QTy, dims, 1.0, 0);
  };
  auto *lmem_in = bb.createAllocActivationInst("tensor.in", q8({1, 64, 3, 3}));
  auto *lmem_weight = bb.createAllocActivationInst("weight", q8({64, 3, 3, 3}));
  auto *lmem_out = bb.createAllocActivationInst("tensor.out", q8({1, 3, 3, 3}));
  auto *lmem_bias =
      bb.createAllocActivationInst("bias", q16({64})); // <2, 64, 1, 1>

  unsigned int pad_top, pad_left, pad_bottom, pad_right;
  unsigned int dilation_h, dilation_w;
  unsigned int stride_h, stride_w;
  int rshift_width;
  bool enable_relu = false;

  pad_top = pad_left = pad_bottom = pad_right = 0;
  dilation_h = dilation_w = 1;
  stride_h = stride_w = 1;
  rshift_width = 0;

  bb.createSophonConvolutionQ8Inst(
      "conv1", lmem_out, lmem_in, lmem_weight, lmem_bias, {stride_h, stride_w},
      {pad_top, pad_left, pad_bottom, pad_right}, {dilation_h, dilation_w},
      rshift_width, enable_relu);

  auto *TTI = sophon::BM1880TargetTransformInfo::getInstance();
  size_t sz_in = TTI->getLMemSizeFromValue(lmem_in);
  size_t sz_out = TTI->getLMemSizeFromValue(lmem_out);
  size_t sz_weight = TTI->getLMemSizeFromValue(lmem_weight);
  size_t sz_bias = TTI->getLMemSizeFromValue(lmem_bias);
  EXPECT_EQ(sz_in,
            16 * 2); // channel size = 3*3 aligned to 16, 2 channels per lane
  EXPECT_EQ(sz_weight, 64 * 3 * 3); // channel size = 3*3*64, 1 channel per lane
  EXPECT_EQ(sz_out,
            16 * 1); // channel size = 3*3 aligned to 16, 1 channel per lane
  EXPECT_EQ(sz_bias, 2 * 2); // channel size = 2*1*1, 2 channel per lane
}

TEST(BM1880GetLMemSizeTest, LIRFC) {
  Module mod;
  Function *F = mod.createFunction("TestGetLMemSize");
  auto IR = llvm::make_unique<IRFunction>(F);
  IRBuilder bb(IR.get());

  auto q8 = [&](llvm::ArrayRef<size_t> dims) {
    return mod.uniqueType(glow::ElemKind::Int8QTy, dims, 1.0, 0);
  };
  auto q16 = [&](llvm::ArrayRef<size_t> dims) {
    return mod.uniqueType(glow::ElemKind::Int16QTy, dims, 1.0, 0);
  };
  auto *lmem_L =
      bb.createAllocActivationInst("matrix.L", q8({1, 800})); // <1, 26, 1, 32>
  auto *lmem_R = bb.createAllocActivationInst(
      "matrix.R", q8({800, 500})); // <800, 16, 1, 32>
  auto *lmem_Y =
      bb.createAllocActivationInst("matrix.Y", q8({1, 500})); // <1, 16, 1, 32>
  auto *lmem_bias =
      bb.createAllocActivationInst("bias", q16({500})); // <2, 16, 1, 32>

  bool enable_relu = false;
  int shift_width = 0;
  bool result_add = false;

  bb.createSophonFullyConnectedQ8Inst("fc1", lmem_Y, lmem_L, lmem_R, lmem_bias,
                                      enable_relu, shift_width, shift_width,
                                      result_add);

  auto *TTI = sophon::BM1880TargetTransformInfo::getInstance();
  size_t sz_L = TTI->getLMemSizeFromValue(lmem_L);
  size_t sz_R = TTI->getLMemSizeFromValue(lmem_R);
  size_t sz_Y = TTI->getLMemSizeFromValue(lmem_Y);
  size_t sz_bias = TTI->getLMemSizeFromValue(lmem_bias);

  EXPECT_EQ(sz_L, 32); // channel size = 1 * 1 * 32, 1 channel per lane
  EXPECT_EQ(sz_R,
            800 * 1 * 32); // channel size = 800 * 1 * 32, one channel per lane
  EXPECT_EQ(sz_Y, 32);     // channel size = 1 * 1 * 32, one channel per lane
  EXPECT_EQ(sz_bias, 32 * 2); // channel size = 2 * 1 * 32, one channel per lane
}
