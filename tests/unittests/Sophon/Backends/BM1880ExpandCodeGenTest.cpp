#include "Backends/Sophon/BM188x/BM1880AllocationsInfo.h"
#include "Backends/Sophon/BM188x/BM1880Backend.h"
#include "Backends/Sophon/BM188x/BM1880CodeGen.h"
#include "Backends/Sophon/BM188x/BM1880ExpandSophonInst.h"
#include "Backends/Sophon/BM188x/BM1880InsertLoadStorePass.h"

#include "glow/Base/Type.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IRBuilder.h"
#include "glow/Optimizer/Optimizer.h"
#include "glow/Support/Debug.h"

#include "gtest/gtest.h"

#define DEBUG_TYPE "SophonLIRTest"

using namespace glow;

template <class T>
static auto Inst(T &t, size_t idx) -> decltype(t->getInstrs().begin()) {
  auto cur_inst = t->getInstrs().begin();
  std::advance(cur_inst, idx);
  return cur_inst;
}

TEST(BM1880ExapndCodeGenTest, ConvQ8Run) {
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
  // auto *outputTensor = ctx.allocate(out_var);
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
  glow::optimize(*IR, true);

  // Hand-coded Memory Allocation
  BM1880AllocationsInfo allocInfo;
  auto &mem_lut =
      allocInfo.getAllocatedAddress(); // Memory Look-up Table for Codegen
  //   For Global Neuron
  mem_lut[IR->getVariableMap()[in_var]] = 0x0;
  mem_lut[IR->getVariableMap()[savePlaceholder]] = 0x0;
  //   For Global Weight
  mem_lut[IR->getVariableMap()[conv_w_var]] = 0x0;
  mem_lut[IR->getVariableMap()[conv_b_var]] = 27;

  // run backend flow
  std::unique_ptr<BM1880Backend> backend(new BM1880Backend());
  backend->reorderWeights(IR.get());
  sophon::runInsertLoadStorePass(IR.get());
  BM1880ExpandSophonInst(IR.get(), allocInfo).run();

  // after ExpandSophonInst
  //  0 %fc_save_output = allocactivation
  //  1 %fc_fc_l = allocactivation
  //  2 %fc_fc_l_load1 = sophonmigdmaglobaltolocal
  //  3 %fc_fc_r = allocactivation
  //  4 %fc_fc_r_load1 = sophonmigdmaglobaltolocal
  //  5 %fc_fc_b = allocactivation
  //  6 %fc_fc_b_load1 = sophonmigdmaglobaltolocal
  //  7 %fc = sophonmifcq8
  //  8 %fc_save_output_store1 = sophonmigdmalocaltoglobal
  //  9 %dealloc1 = deallocactivation
  //  10 %dealloc2 = deallocactivation
  //  11 %dealloc3 = deallocactivation
  //  12 %dealloc4 = deallocactivation

  DEBUG_GLOW(IR->dump());

  // Hand-coded for Local Memory
  // output
  mem_lut[llvm::cast<Value>(Inst(IR, 0))] = 0; // size: 1 x 32
  // input
  mem_lut[llvm::cast<Value>(Inst(IR, 1))] = 1 * 32; // size: 1 x 32
  // filter
  mem_lut[llvm::cast<Value>(Inst(IR, 3))] = 2 * 32; // size: 1024 x 32
  // bias
  mem_lut[llvm::cast<Value>(Inst(IR, 5))] = 2 * 32 + 1024 * 32; // size: 2 * 32

  // run
  backend->codegen(std::move(IR), &allocInfo)->execute(ctx);

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

TEST(BM1880ExapndCodeGenTest, LIRFC) {
  Module mod;
  Function *F = mod.createFunction("TestLIRFC");

  // L Matrix = M x K
  // R Matrix = K x N
  // Y Matrix = M x N
  // B = N
  const unsigned M = 1;
  const unsigned K = 1024;
  const unsigned N = 1024;

  auto *var_fc_l = mod.createPlaceholder(glow::ElemKind::Int8QTy, {M, K}, 1, 0,
                                         "fc.l", false);
  auto *var_fc_r =
      mod.createConstant(glow::ElemKind::Int8QTy, {K, N}, 1, 0, "fc.r");
  auto *var_fc_b =
      mod.createConstant(glow::ElemKind::Int16QTy, {N}, 1, 0, "fc.b");

  // Initialize Constant

  // K x N  8,  8,  8, ...,
  //        8,  8,  8, ...,
  //        8,  8,  8, ...,
  //        ...
  //       -8, -8, -8, ...,
  //       -8, -8, -8, ...,
  //       -8, -8, -8, ...,
  //

  for (unsigned i = 0; i < K; i++) {
    for (unsigned j = 0; j < N; j++) {
      int val;
      if (i < (K / 2 + 1)) {
        val = 8;
      } else {
        val = -8;
      }
      var_fc_r->getHandle<int8_t>().at({i, j}) = val;
    }
  }

  // { 2, 2, 2, 2, ..., 0, 0, 0}
  for (unsigned i = 0; i < N; i++) {
    int val;
    if ((i / 32) % 2 == 0) {
      val = 2;
    } else {
      val = 0;
    }
    var_fc_b->getHandle<int16_t>().at({i}) = val;
  }

  Context ctx;
  auto *ctx_fc_l = ctx.allocate(var_fc_l);

  // Initialize Placeholder
  for (unsigned i = 0; i < M; i++) {
    for (unsigned j = 0; j < K; j++) {
      ctx_fc_l->getHandle<int8_t>().at({i, j}) = 1;
    }
  }

  // 1. create glow HIR
  TypeRef ty = mod.uniqueType(glow::ElemKind::Int8QTy, {M, N}, 1, 0);
  auto *fc = F->addNode(new SophonFullyConnectedQ8Node(
      "fc", ty, var_fc_l, var_fc_r, var_fc_b, false, 0, 0, false));

  auto *save = F->createSave("save", fc);
  auto *savePlaceholder = save->getPlaceholder();
  savePlaceholder->setName("save_output");
  auto *outputTensor = ctx.allocate(savePlaceholder);

  // 2. create glow LIR
  auto IR = llvm::make_unique<IRFunction>(F);
  IR->generateIR();
  glow::optimize(*IR, true);

  // 3. Hand-coded for Global Memory Allocation
  BM1880AllocationsInfo allocInfo;
  auto &mem_lut =
      allocInfo.getAllocatedAddress(); // Memory Look-up Table for Codegen
  //   For Global Neuron
  mem_lut[IR->getVariableMap()[var_fc_l]] = 0x0;          // size: M x K
  mem_lut[IR->getVariableMap()[savePlaceholder]] = M * K; // size: M x N
  //   For Global Weight
  mem_lut[IR->getVariableMap()[var_fc_r]] = 0x0;   // size: K x N
  mem_lut[IR->getVariableMap()[var_fc_b]] = K * N; // size: 2 X N

  // 4. run backend flow
  std::unique_ptr<BM1880Backend> backend(new BM1880Backend());
  backend->reorderWeights(IR.get());
  sophon::runInsertLoadStorePass(IR.get());
  BM1880ExpandSophonInst(IR.get(), allocInfo).run();

  // 5. Hand-coded for Local Memory allocation
  // expected MI LIR
  // 0 %fc_save_output = allocactivation  { Ty: i8[S:1.0000
  // O:0][-128.000,127.000]<1 x 1024>} 1 %fc_fc_l = allocactivation  { Ty:
  // i8[S:1.0000 O:0][-128.000,127.000]<1 x 1024>} 2 %fc_fc_l_load1 =
  // sophonmigdmaglobaltolocal @out %fc_fc_l, @in %fc_l { ShapeNCHW: [1, 32, 1,
  // 32], GlobalStrideNCH: [1024, 32, 32], IsGlobalWeightSpace: 0,
  // IsLocalAligned: 1} 3 %fc_fc_r = allocactivation  { Ty: i8[S:1.0000
  // O:0][-128.000,127.000]<1024 x 1024>} 4 %fc_fc_r_load1 =
  // sophonmigdmaglobaltolocal @out %fc_fc_r, @in %fc_r { ShapeNCHW: [1024, 32,
  // 1, 32], GlobalStrideNCH: [1024, 32, 32], IsGlobalWeightSpace: 1,
  // IsLocalAligned: 1} 5 %fc_fc_b = allocactivation  { Ty: i16[S:1.0000
  // O:0][-32768.000,32767.000]<1024>} 6 %fc_fc_b_load1 =
  // sophonmigdmaglobaltolocal @out %fc_fc_b, @in %fc_b { ShapeNCHW: [2, 32, 1,
  // 32], GlobalStrideNCH: [1024, 32, 32], IsGlobalWeightSpace: 1,
  // IsLocalAligned: 0}

  IR->dump();
  // f_save_output
  mem_lut[llvm::cast<Value>(Inst(IR, 0))] = 0; // size: 1 x 32
  // fc_l
  mem_lut[llvm::cast<Value>(Inst(IR, 1))] = 1 * 32; // size: 1 x 32
  // fc_r
  mem_lut[llvm::cast<Value>(Inst(IR, 3))] = 2 * 32; // size: 1024 x 32
  // fc_b
  mem_lut[llvm::cast<Value>(Inst(IR, 5))] = 2 * 32 + 1024 * 32; // size: 2 * 32

  // 6. run on cmodel
  backend->codegen(std::move(IR), &allocInfo)->execute(ctx);

  auto H = outputTensor->getHandle<int8_t>();
  for (unsigned i = 0; i < N; i++) {
    int ans;
    if ((i / 32) % 2 == 0) {
      ans = 18;
    } else {
      ans = 16;
    }
    EXPECT_EQ(H.at({0, i}), ans);
  }
}
