#include "Backends/Sophon/BM188x/BM1880Backend.h"
#include "glow/Base/Type.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"

#include "gtest/gtest.h"

using namespace glow;

TEST(BM1880DeleteQuantizeNodeTest, DeleteQNode) {
  Module mod;
  Context ctx;
  Function *F = mod.createFunction("main");
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 1, 3, 3}, "input", false);
  auto *conv_w_var = mod.createConstant(glow::ElemKind::Int8QTy, {1, 3, 3, 3},
                                        1, 0, "conv1.w");
  auto *conv_b_var =
      mod.createConstant(glow::ElemKind::Int16QTy, {3}, 1, 0, "conv1.b");

  auto *inputTensor = ctx.allocate(input);
  // clang-format off
  inputTensor->getHandle<float>()= {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  conv_w_var->getHandle<int8_t>() = {
    0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 2, 0, 0, 0, 0,
    0, 0, 0, 0, 3, 0, 0, 0, 0
  };
  conv_b_var->getHandle<int16_t>() = {0, 1, 2};
  // clang-format on

  TypeRef qty = mod.uniqueType(glow::ElemKind::Int8QTy, {1, 1, 3, 3}, 1, 0);
  auto *QN = F->createQuantize("Quan", input, qty);
  TypeRef ty = mod.uniqueType(glow::ElemKind::Int8QTy, {1, 3, 3, 3}, 1, 0);
  auto *conv = F->addNode(
      new SophonConvolutionQ8Node("conv", ty, QN, conv_w_var, conv_b_var,
                                  {1, 1}, {1, 1, 1, 1}, {1, 1}, 0, false));
  auto *DN = F->createDequantize("DQuan", conv);
  auto *SN = F->createSave("save", DN);
  auto *savePlaceholder = SN->getPlaceholder();
  savePlaceholder->setName("save_output");
  auto *outputTensor = ctx.allocate(savePlaceholder);

  EXPECT_EQ(4, F->getNodes().size());
  std::unique_ptr<BM1880Backend> backend(new BM1880Backend());
  backend->transformPreLowering(F, CompilationMode::Infer);
  EXPECT_EQ(2, F->getNodes().size());
}
