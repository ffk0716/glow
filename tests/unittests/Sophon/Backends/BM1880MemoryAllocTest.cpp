#include "Backends/Sophon/BM188x/BM1880AllocationsInfo.h"
#include "Backends/Sophon/BM188x/BM1880Backend.h"
#include "Backends/Sophon/BM188x/BM1880CodeGen.h"
#include "Backends/Sophon/BM188x/BM1880ExpandSophonInst.h"
#include "Backends/Sophon/BM188x/BM1880InsertLoadStorePass.h"

#include "glow/Base/Type.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IRBuilder.h"
#include "glow/Support/Debug.h"
#include "llvm/Support/Debug.h"

#include "gtest/gtest.h"

#define DEBUG_TYPE "memory_alloc_test"

using namespace glow;

template <class T>
static auto Inst(T &t, size_t idx) -> decltype(t->getInstrs().begin()) {
  auto cur_inst = t->getInstrs().begin();
  std::advance(cur_inst, idx);
  return cur_inst;
}

TEST(BM1880MemAllocTest, ConvMemAllocRun) {
  Module mod;
  Function *F = mod.createFunction("TestConvQ8Run");

  // Because all importers will add "save_" prefix for input placeholder
  // currently Sophon Backend uses this prefix to recognize input/output
  auto *in_var = mod.createPlaceholder(glow::ElemKind::Int8QTy, {1, 1, 3, 3}, 1,
                                       0, "input", false);
  auto *conv_w_var = mod.createConstant(glow::ElemKind::Int8QTy, {3, 1, 3, 3},
                                        1, 0, "conv1.w");
  auto *conv_b_var =
      mod.createConstant(glow::ElemKind::Int16QTy, {3}, 1, 0, "conv1.b");

  // init input/output/weight
  Context ctx;
  auto *inputTensor = ctx.allocate(in_var);
  // clang-format off
  inputTensor->getHandle<int8_t>()= {1, 2, 3, 4, 5, 6, 7, 8, 9};
  conv_w_var->getHandle<int8_t>() = {
    0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 2, 0, 0, 0, 0,
    0, 0, 0, 0, 3, 0, 0, 0, 0
  };
  conv_b_var->getHandle<int16_t>() = {0, 1, 2};
  // clang-format on

  DEBUG_GLOW(inputTensor->getHandle<int8_t>().dump());
  DEBUG_GLOW(conv_w_var->getHandle<int8_t>().dump());
  DEBUG_GLOW(conv_b_var->getHandle<int16_t>().dump());

  // create glow HIR
  TypeRef ty = mod.uniqueType(glow::ElemKind::Int8QTy, {1, 3, 3, 3}, 1, 0);
  auto *conv = F->addNode(
      new SophonConvolutionQ8Node("conv", ty, in_var, conv_w_var, conv_b_var,
                                  {1, 1}, {1, 1, 1, 1}, {1, 1}, 0, false));
  auto *save = F->createSave("save", conv);
  auto *savePlaceholder = save->getPlaceholder();
  savePlaceholder->setName("save_output");
  auto *outputTensor = ctx.allocate(savePlaceholder);

  // create glow LIR
  auto IR = llvm::make_unique<IRFunction>(F);
  IR->generateIR();

  // run backend flow
  std::unique_ptr<BM1880Backend> backend(new BM1880Backend());
  BM1880AllocationsInfo alloc_info(ctx, backend->getTTI());
  backend->runOptimizationPasses(IR.get(), &alloc_info);

  // check alloc info
  // [conv_conv1_b] = 128
  // [conv1_b] = 27
  // [save_output] = 9
  // [conv_conv1_w] = 64
  // [conv1_w] = 0
  // [conv_res] = 192
  // [conv_input] = 0
  // [input] = 0
  std::map<std::string, size_t> result = {
      {"conv1_b", 27},      {"conv_conv1_b", 128}, {"save_output", 9},
      {"conv_conv1_w", 64}, {"conv1_w", 0},        {"input", 0},
      {"conv_res", 192},    {"conv_input", 0}};
  auto &mem_lut =
      alloc_info.getAllocatedAddress(); // Memory Look-up Table for Codegen
  for (auto entry : mem_lut) {
    EXPECT_EQ(result[entry.first->getName().str()], entry.second);
    std::cout << "[" << entry.first->getName().str() << "] = " << entry.second
              << "\n";
  }

#if 0
  // after ExpandSophonInst
  %conv1_w(0) = WeightVar i8[S:1.0000 O:0][-128.000,127.000]<1 x 3 x 3 x 3> const // size: 27 // Users: @in 4
  %conv1_b(27) = WeightVar i16[S:1.0000 O:0][-32768.000,32767.000]<3> const // size: 6 // Users: @in 6
  %input(0) = WeightVar i8[S:1.0000 O:0][-128.000,127.000]<1 x 1 x 3 x 3> mutable // size: 9 // Users: @in 2
  %save_output(9) = WeightVar i8[S:1.0000 O:0][-128.000,127.000]<1 x 3 x 3 x 3> mutable // size: 27 // Users: @out 8

  0 %conv_res(192) = allocactivation  { Ty: i8[S:1.0000 O:0][-128.000,127.000]<1 x 3 x 3 x 3>} // size: 27 // Users: @in 8, @out 7, @out 9
  1 %conv_input(0) = allocactivation  { Ty: i8[S:1.0000 O:0][-128.000,127.000]<1 x 1 x 3 x 3>} // size: 9 // Users: @in 7, @out 2, @out 10
  2 %conv_input_load1 = sophonmigdmaglobaltolocal @out %conv_input(16), @in %input { ShapeNCHW: [1, 1, 3, 3], GlobalStrideNCH: [9, 9, 3], IsGlobalWeightSpace: 0, IsLocalAligned: 1}
  3 %conv_conv1_w(64) = allocactivation  { Ty: i8[S:1.0000 O:0][-128.000,127.000]<1 x 3 x 3 x 3>} // size: 27 // Users: @in 7, @out 4, @out 11
  4 %conv_conv1_w_load1 = sophonmigdmaglobaltolocal @out %conv_conv1_w(32), @in %conv1_w { ShapeNCHW: [1, 3, 9, 1], GlobalStrideNCH: [27, 9, 1], IsGlobalWeightSpace: 1, IsLocalAligned: 0}
  5 %conv_conv1_b(128) = allocactivation  { Ty: i16[S:1.0000 O:0][-32768.000,32767.000]<3>} // size: 6 // Users: @in 7, @out 6, @out 12
  6 %conv_conv1_b_load1 = sophonmigdmaglobaltolocal @out %conv_conv1_b, @in %conv1_b { ShapeNCHW: [2, 3, 1, 1], GlobalStrideNCH: [3, 1, 1], IsGlobalWeightSpace: 1, IsLocalAligned: 0}
  7 %conv = sophonmiconvolutionq8 @out %conv_res(192), @in %conv_input(0), @in %conv_conv1_w(64), @in %conv_conv1_b(128) { StrideHW: [1, 1], PadTLBR: [1, 1, 1, 1], DilationHW: [1, 1], RShiftWidth: 0, EnableRelu: 0, StreamID: 0, InstID: 0, Depends: []}
  8 %save11 = sophonmigdmalocaltoglobal @out %save_output, @in %conv_res { ShapeNCHW: [1, 3, 3, 3], GlobalStrideNCH: [27, 9, 3], IsGlobalWeightSpace: 0, IsLocalAligned: 1}
#endif

  // codegen and run
  backend->codegen(std::move(IR), &alloc_info)->execute(ctx);

  // check result
  auto H = outputTensor->getHandle<int8_t>();
  EXPECT_EQ(H.at({0, 0, 0, 0}), 1);
  EXPECT_EQ(H.at({0, 0, 0, 1}), 2);
  EXPECT_EQ(H.at({0, 0, 0, 2}), 3);
  EXPECT_EQ(H.at({0, 0, 1, 0}), 4);
  EXPECT_EQ(H.at({0, 0, 1, 1}), 5);
  EXPECT_EQ(H.at({0, 0, 1, 2}), 6);
  EXPECT_EQ(H.at({0, 0, 2, 0}), 7);
  EXPECT_EQ(H.at({0, 0, 2, 1}), 8);
  EXPECT_EQ(H.at({0, 0, 2, 2}), 9);

  EXPECT_EQ(H.at({0, 1, 0, 0}), 3);
  EXPECT_EQ(H.at({0, 1, 0, 1}), 5);
  EXPECT_EQ(H.at({0, 1, 0, 2}), 7);
  EXPECT_EQ(H.at({0, 1, 1, 0}), 9);
  EXPECT_EQ(H.at({0, 1, 1, 1}), 11);
  EXPECT_EQ(H.at({0, 1, 1, 2}), 13);
  EXPECT_EQ(H.at({0, 1, 2, 0}), 15);
  EXPECT_EQ(H.at({0, 1, 2, 1}), 17);
  EXPECT_EQ(H.at({0, 1, 2, 2}), 19);

  EXPECT_EQ(H.at({0, 2, 0, 0}), 5);
  EXPECT_EQ(H.at({0, 2, 0, 1}), 8);
  EXPECT_EQ(H.at({0, 2, 0, 2}), 11);
  EXPECT_EQ(H.at({0, 2, 1, 0}), 14);
  EXPECT_EQ(H.at({0, 2, 1, 1}), 17);
  EXPECT_EQ(H.at({0, 2, 1, 2}), 20);
  EXPECT_EQ(H.at({0, 2, 2, 0}), 23);
  EXPECT_EQ(H.at({0, 2, 2, 1}), 26);
  EXPECT_EQ(H.at({0, 2, 2, 2}), 29);
}
