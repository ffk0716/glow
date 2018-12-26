#include "Backends/Sophon/BM188x/BM1880CodeGen.h"
#include "Backends/Sophon/BM188x/BM1880AllocationsInfo.h"
#include "Backends/Sophon/BM188x/BM1880Backend.h"
#include "Backends/Sophon/BM188x/BM1880TargetTransformInfo.h"
#include "glow/Base/Type.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IRBuilder.h"

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

using namespace glow;

TEST(SophonLIRTest, HIRExample) {
  // HIR example
  Module mod;
  Function *F = mod.createFunction("TestHIR");
  auto *X = mod.createPlaceholder(ElemKind::FloatTy, {1, 1, 3}, "X", false);
  auto *Pow1 = F->createPow("Pow1", X, 2.0);
  auto *save1 = F->createSave("save", Pow1);
  auto IR = llvm::make_unique<IRFunction>(F);
  IR->generateIR();
  IR->dump();
}

TEST(BM1880CodeGenTest, LIRConv) {
  // LIR example
  Module mod;
  Function *F = mod.createFunction("TestLIRConv");
  auto IR = llvm::make_unique<IRFunction>(F);
  IRBuilder bb(IR.get());

  // N, C, H, W
  auto *input =
      bb.createWeightVar(glow::ElemKind::FloatTy, {1, 1, 3, 3}, "input",
                         WeightVar::MutabilityKind::Constant);

  // OC, IC, KH, KW
  auto *conv_w =
      bb.createWeightVar(glow::ElemKind::FloatTy, {3, 1, 3, 3}, "conv1.w",
                         WeightVar::MutabilityKind::Constant);

  // OC
  auto *conv_b = bb.createWeightVar(glow::ElemKind::FloatTy, {3}, "conv1.b",
                                    WeightVar::MutabilityKind::Constant);

  // N, C, H, W
  auto *output =
      bb.createWeightVar(glow::ElemKind::FloatTy, {1, 3, 3, 3}, "output1",
                         WeightVar::MutabilityKind::Mutable);

  auto *lmem_in = bb.createAllocActivationInst("lmem.in", input->getType());
  auto *lmem_weight =
      bb.createAllocActivationInst("lmem.weight", conv_w->getType());
  auto *lmem_bias =
      bb.createAllocActivationInst("lmem.bias", conv_b->getType());
  auto *lmem_out = bb.createAllocActivationInst("lmem.out", output->getType());

  int rshift_width;
  bool enable_relu = false;
  int stream_id = 0;
  int inst_id = 0;

  bool local_aligned = true;
  bool local_not_aligned = false;

  std::vector<unsigned_t> depends;
  rshift_width = 0;
  // dma shape  {1, 1, 3, 3}
  // dma stride {9, 9, 3}
  bb.createSophonMIGDMAGlobalToLocalInst("load.conv1.in", lmem_in, input,
                                         {1, 1, 3, 3}, {9, 9, 3}, false,
                                         local_aligned);
  // dma shape  {1, 3, 9, 1}
  // dma stride {27, 9, 1}
  bb.createSophonMIGDMAGlobalToLocalInst("load.conv1.weight", lmem_weight,
                                         conv_w, {1, 3, 9, 1}, {27, 9, 1}, true,
                                         local_not_aligned);

  bb.createSophonMIConvolutionQ8Inst(
      "conv1", lmem_out, lmem_in, lmem_weight, lmem_bias, {1, 1}, {0, 0, 0, 0},
      {1, 1}, rshift_width, enable_relu, stream_id, inst_id, depends);

  // dma shape  {1, 3, 3, 3}
  // dma stride {27, 9, 3}
  bb.createSophonMIGDMALocalToGlobalInst("store.conv1.output", output, lmem_out,
                                         {1, 3, 3, 3}, {27, 9, 3}, false,
                                         local_aligned);

  bb.createDeallocActivationInst("dealloc1", lmem_in);
  bb.createDeallocActivationInst("dealloc2", lmem_weight);
  bb.createDeallocActivationInst("dealloc3", lmem_bias);
  bb.createDeallocActivationInst("dealloc4", lmem_out);

  IR->dump();
  BM1880AllocationsInfo allocInfo;
  allocInfo.getAllocatedAddress()[input];
  allocInfo.getAllocatedAddress()[conv_w];
  allocInfo.getAllocatedAddress()[conv_b];
  allocInfo.getAllocatedAddress()[output];
  allocInfo.getAllocatedAddress()[lmem_in];
  allocInfo.getAllocatedAddress()[lmem_weight];
  allocInfo.getAllocatedAddress()[lmem_bias];
  allocInfo.getAllocatedAddress()[lmem_out];

  std::unique_ptr<BM1880CodeGen> codegen =
      BM1880CodeGen::createCodeGen(IR.get(), allocInfo);
  codegen->performCodeGen();
}

TEST(BM1880CodeGenTest, GenWeights) {
  Module mod;
  Function *F = mod.createFunction("TestGenWeights");
  auto IR = llvm::make_unique<IRFunction>(F);

  auto *conv1_w_var = mod.createConstant(glow::ElemKind::Int8QTy, {1, 1, 1, 1},
                                         1, 0, "conv1.w");
  auto *conv1_b_var =
      mod.createConstant(glow::ElemKind::Int16QTy, {1}, 1, 0, "conv1.b");
  auto *conv2_w_var = mod.createConstant(glow::ElemKind::Int8QTy, {1, 1, 1, 1},
                                         1, 0, "conv2.w");
  auto *conv2_b_var =
      mod.createConstant(glow::ElemKind::Int16QTy, {1}, 1, 0, "conv2.b");

  conv1_w_var->getHandle<int8_t>() = {1};
  conv1_b_var->getHandle<int16_t>() = {2};
  conv2_w_var->getHandle<int8_t>() = {3};
  conv2_b_var->getHandle<int16_t>() = {4};

  BM1880AllocationsInfo allocInfo;

  auto &mem_lut = allocInfo.getAllocatedAddress();
  {
    IRBuilder bb(IR.get());

    auto *conv1_w =
        bb.createWeightVar(glow::ElemKind::Int8QTy, {1, 1, 1, 1}, 1, 0,
                           "conv1.w", WeightVar::MutabilityKind::Constant);

    auto *conv1_b =
        bb.createWeightVar(glow::ElemKind::Int16QTy, {1}, 1, 0, "conv1.b",
                           WeightVar::MutabilityKind::Constant);

    auto *conv2_w =
        bb.createWeightVar(glow::ElemKind::Int8QTy, {1, 1, 1, 1}, 1, 0,
                           "conv2.w", WeightVar::MutabilityKind::Constant);

    auto *conv2_b =
        bb.createWeightVar(glow::ElemKind::Int16QTy, {1}, 1, 0, "conv2.b",
                           WeightVar::MutabilityKind::Constant);

    IR->getVariableMap()[conv1_w_var] = conv1_w;
    IR->getVariableMap()[conv1_b_var] = conv1_b;
    IR->getVariableMap()[conv2_w_var] = conv2_w;
    IR->getVariableMap()[conv2_b_var] = conv2_b;

    // The insert order is:
    //   conv1_w | conv1_b | conv2_w | conv2_b
    // but the mem_allocator might want:
    //   conv2_w | conv2_b | conv1_w | conv1_b
    mem_lut[conv1_w] = 0x3;
    mem_lut[conv1_b] = 0x4;
    mem_lut[conv2_w] = 0x0;
    mem_lut[conv2_b] = 0x1;
  }
  std::unique_ptr<BM1880Backend> backend(new BM1880Backend());
  std::vector<uint8_t> u8_weights;
  // Gen weights base on mem_lut
  backend->generateWeights(IR.get(), allocInfo, u8_weights);

  // Check weight size
  EXPECT_EQ(6, u8_weights.size());

  // Check weight data
  EXPECT_EQ(3, u8_weights[0]);
  EXPECT_EQ(4, u8_weights[1]);
  EXPECT_EQ(1, u8_weights[3]);
  EXPECT_EQ(2, u8_weights[4]);
}

TEST(BM1880CodeGenTest, LIRConvCompileRun) {
  Module mod;
  Function *F = mod.createFunction("TestLIRConv");
  auto IR = llvm::make_unique<IRFunction>(F);

  // Because all importers will add "save_" prefix for input placeholder
  // currently Sophon Backend uses this prefix to recognize input/output
  auto *in_var = mod.createPlaceholder(glow::ElemKind::Int8QTy, {1, 1, 3, 3}, 1,
                                       0, "input", false);
  auto *out_var = mod.createPlaceholder(glow::ElemKind::Int8QTy, {1, 3, 3, 3},
                                        1, 0, "save_output", false);
  auto *conv_w_var = mod.createConstant(glow::ElemKind::Int8QTy, {3, 1, 3, 3},
                                        1, 0, "conv1.w");
  auto *conv_b_var =
      mod.createConstant(glow::ElemKind::Int16QTy, {3}, 1, 0, "conv1.b");
  Context ctx;
  auto *inputTensor = ctx.allocate(in_var);
  inputTensor->zero();
  auto input_handle = inputTensor->getHandle<int8_t>();
  input_handle = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  input_handle.dump();

  // clang-format off
  conv_w_var->getHandle<int8_t>() = {
    0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 2, 0, 0, 0, 0,
    0, 0, 0, 0, 3, 0, 0, 0, 0
  };
  conv_b_var->getHandle<int16_t>() = {0, 1, 2};
  // clang-format on

  conv_w_var->getHandle<int8_t>().dump();
  conv_b_var->getHandle<int16_t>().dump();
  auto *outputTensor = ctx.allocate(out_var);
  // Hand-coded Memory Allocation
  BM1880AllocationsInfo allocInfo;
  auto &mem_lut =
      allocInfo.getAllocatedAddress(); // Memory Look-up Table for Codegen
  std::unique_ptr<BM1880Backend> backend(new BM1880Backend());
  auto *TTI = backend->getTTI();
  {
    IRBuilder bb(IR.get());

    // N, C, H, W
    auto *input =
        bb.createWeightVar(glow::ElemKind::Int8QTy, {1, 1, 3, 3}, 1, 0, "input",
                           WeightVar::MutabilityKind::Mutable);

    // IC, OC, KH, KW
    auto *conv_w =
        bb.createWeightVar(glow::ElemKind::Int8QTy, {3, 1, 3, 3}, 1, 0,
                           "conv1.w", WeightVar::MutabilityKind::Constant);

    // OC
    auto *conv_b =
        bb.createWeightVar(glow::ElemKind::Int16QTy, {3}, 1, 0, "conv1.b",
                           WeightVar::MutabilityKind::Constant);

    // N, C, H, W
    auto *output =
        bb.createWeightVar(glow::ElemKind::Int8QTy, {1, 3, 3, 3}, 1, 0,
                           "output1", WeightVar::MutabilityKind::Mutable);
    IR->getVariableMap()[in_var] = input;
    IR->getVariableMap()[out_var] = output;
    IR->getVariableMap()[conv_w_var] = conv_w;
    IR->getVariableMap()[conv_b_var] = conv_b;

    auto *lmem_in = bb.createAllocActivationInst("lmem.in", input->getType());
    auto *lmem_weight =
        bb.createAllocActivationInst("lmem.weight", conv_w->getType());
    auto *lmem_bias =
        bb.createAllocActivationInst("lmem.bias", conv_b->getType());
    auto *lmem_out =
        bb.createAllocActivationInst("lmem.out", output->getType());

    bool enable_relu = false;
    int stream_id = 0;
    int inst_id = 0;

    bool local_aligned = true;
    bool local_not_aligned = false;

    std::vector<unsigned_t> depends;
    int rshift_width = 0;

    // Input
    // dma shape  {1, 1, 3, 3}
    // dma stride {9, 9, 3}
    bb.createSophonMIGDMAGlobalToLocalInst("load.conv1.in", lmem_in, input,
                                           {1, 1, 3, 3}, {9, 9, 3}, false,
                                           local_aligned);

    // Weight
    // dma shape  {1, 3, 9, 1}
    // dma stride {27, 9, 1}
    bb.createSophonMIGDMAGlobalToLocalInst("load.conv1.weight", lmem_weight,
                                           conv_w, {1, 3, 9, 1}, {27, 9, 1},
                                           true, local_not_aligned);

    // Bias
    // dma shape  {2, 3, 1, 1}
    // dma stride {3, 1, 1}
    bb.createSophonMIGDMAGlobalToLocalInst("load.conv1.bias", lmem_bias, conv_b,
                                           {2, 3, 1, 1}, {3, 1, 1}, true,
                                           local_not_aligned);

    bb.createSophonMIConvolutionQ8Inst("conv1", lmem_out, lmem_in, lmem_weight,
                                       lmem_bias, {1, 1}, {1, 1, 1, 1}, {1, 1},
                                       rshift_width, enable_relu, stream_id,
                                       inst_id, depends);

    // Output
    // dma shape  {1, 3, 3, 3}
    // dma stride {27, 9, 3}
    bb.createSophonMIGDMALocalToGlobalInst("store.conv1.output", output,
                                           lmem_out, {1, 3, 3, 3}, {27, 9, 3},
                                           false, local_aligned);

    bb.createDeallocActivationInst("dealloc1", lmem_in);
    bb.createDeallocActivationInst("dealloc2", lmem_weight);
    bb.createDeallocActivationInst("dealloc3", lmem_bias);
    bb.createDeallocActivationInst("dealloc4", lmem_out);

    // Global Memory Allocation
    //   For Neuron
    mem_lut[input] = 0x0;
    mem_lut[output] = 0x10;
    //   For Weight
    mem_lut[conv_w] = 0x0;
    mem_lut[conv_b] = 27;
    allocInfo.setActivationsMemSize(48);

    // Local Memory Allocation
    //   Rule: allocate aligned tensors, then non-aligned tensors
    //   Order: lmem_in, lmem_out, lmem_weight, lmem_bias
    mem_lut[lmem_in] = 0x0;
    mem_lut[lmem_out] = mem_lut[lmem_in] + TTI->getLMemSizeFromValue(lmem_in);
    mem_lut[lmem_weight] =
        mem_lut[lmem_out] + TTI->getLMemSizeFromValue(lmem_out);
    mem_lut[lmem_bias] =
        mem_lut[lmem_weight] + TTI->getLMemSizeFromValue(lmem_weight);
  }

  backend->reorderWeights(IR.get());
  backend->codegen(std::move(IR), &allocInfo)->execute(ctx);

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

TEST(BM1880CodeGenTest, LIRFCRun) {
  Module mod;
  Function *F = mod.createFunction("TestLIRFC");

  // L Matrix = M x K
  // R Matrix = K x N
  // Y Matrix = M x N
  // B = N
  const unsigned M = 1;
  const unsigned K = 1024;
  const unsigned N = 1024;

  auto IR = llvm::make_unique<IRFunction>(F);
  auto *var_fc_l = mod.createPlaceholder(glow::ElemKind::Int8QTy, {M, K}, 1, 0,
                                         "fc.l", false);
  auto *var_fc_r =
      mod.createConstant(glow::ElemKind::Int8QTy, {K, N}, 1, 0, "fc.r");
  auto *var_fc_b =
      mod.createConstant(glow::ElemKind::Int16QTy, {N}, 1, 0, "fc.b");
  auto *var_fc_y = mod.createPlaceholder(glow::ElemKind::Int8QTy, {M, N}, 1, 0,
                                         "save_fc_y", false);

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
  auto *ctx_fc_y = ctx.allocate(var_fc_y);

  // Initialize Placeholder
  for (unsigned i = 0; i < M; i++) {
    for (unsigned j = 0; j < K; j++) {
      ctx_fc_l->getHandle<int8_t>().at({i, j}) = 1;
    }
  }

  IRBuilder bb(IR.get());

  // <1, 1024> = <1, 32, 1, 32>
  auto *mat_L = bb.createWeightVar(glow::ElemKind::Int8QTy, {M, K}, 1, 0,
                                   "mat.L", WeightVar::MutabilityKind::Mutable);
  // <1024, 1024> = <1024, 32, 1, 32>
  auto *mat_R =
      bb.createWeightVar(glow::ElemKind::Int8QTy, {K, N}, 1, 0, "mat.R",
                         WeightVar::MutabilityKind::Constant);
  // <1024> = <2, 32, 1, 32>
  auto *bias = bb.createWeightVar(glow::ElemKind::Int16QTy, {N}, 1, 0, "bias",
                                  WeightVar::MutabilityKind::Constant);
  // <1, 1024> = <1, 32, 1, 32>
  auto *mat_Y = bb.createWeightVar(glow::ElemKind::Int8QTy, {M, N}, 1, 0,
                                   "mat.Y", WeightVar::MutabilityKind::Mutable);

  IR->getVariableMap()[var_fc_l] = mat_L;
  IR->getVariableMap()[var_fc_r] = mat_R;
  IR->getVariableMap()[var_fc_b] = bias;
  IR->getVariableMap()[var_fc_y] = mat_Y;

  auto *lmem_l = bb.createAllocActivationInst("lmem.L", mat_L->getType());
  auto *lmem_r = bb.createAllocActivationInst("lmem.R", mat_R->getType());
  auto *lmem_b = bb.createAllocActivationInst("lmem.B", bias->getType());
  auto *lmem_y = bb.createAllocActivationInst("lmem.Y", mat_Y->getType());

  bool local_aligned = true;
  bool local_not_aligned = false;

  // Matrix L
  // dma shape  {      1, 32,  1, 32}
  // dma stride {32 * 32, 32, 32}
  bb.createSophonMIGDMAGlobalToLocalInst("load.fc.l", lmem_l, mat_L,
                                         {1, 32, 1, 32}, {128 * 32, 32, 32},
                                         false, local_aligned);

  // Matrix R
  // dma shape  {   1024, 32,  1, 32}
  // dma stride {32 * 32, 32, 32}
  bb.createSophonMIGDMAGlobalToLocalInst("load.fc.r", lmem_r, mat_R,
                                         {1024, 32, 1, 32}, {32 * 32, 32, 32},
                                         true, local_aligned);

  // Bias
  // dma shape  {      2, 32,  1, 32}
  // dma stride {32 * 32, 32, 32}
  bb.createSophonMIGDMAGlobalToLocalInst("load.fc.b", lmem_b, bias,
                                         {2, 32, 1, 32}, {32 * 32, 32, 32},
                                         true, local_not_aligned);

  unsigned rshift = 0;
  unsigned lshift = 0;
  bool result_add = false;

  bb.createSophonMIFCQ8Inst("FC1", lmem_y, lmem_l, lmem_r, lmem_b, rshift,
                            lshift, result_add);

  // Matrix Y
  // dma shape  {      1, 32,  1, 32}
  // dma stride {32 * 32, 32, 32}
  bb.createSophonMIGDMALocalToGlobalInst("store.fc.y", mat_Y, lmem_y,
                                         {1, 32, 1, 32}, {32 * 32, 32, 32},
                                         false, local_aligned);

  BM1880AllocationsInfo allocInfo;
  auto &mem_lut = allocInfo.getAllocatedAddress();

  // Global Memory Allocation
  //   For Neuron
  mem_lut[mat_L] = 0;     // size: M x K
  mem_lut[mat_Y] = M * K; // size: M x N
  //   For Weight
  mem_lut[mat_R] = 0;    // size: K x N
  mem_lut[bias] = K * N; // size: 2 x N

  // Local Memory Allocation
  mem_lut[lmem_l] = 0x0;                // size: 1 x 32
  mem_lut[lmem_y] = 1 * 32;             // size: 1 x 32
  mem_lut[lmem_r] = 2 * 32;             // size: 1024 x 32
  mem_lut[lmem_b] = 2 * 32 + 1024 * 32; // size: 2 x 32

  std::unique_ptr<BM1880Backend> backend(new BM1880Backend());
  backend->reorderWeights(IR.get());
  backend->codegen(std::move(IR), &allocInfo)->execute(ctx);

  auto H = ctx_fc_y->getHandle<int8_t>();
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
