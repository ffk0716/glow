
#include "Backends/Sophon/BM188x/BM1880HandleReshapePass.h"
#include "Backends/Sophon/BM188x/BM1880InsertLoadStorePass.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/Optimizer.h"
#include "glow/Support/Debug.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"
#include <iostream>

using namespace glow;
using namespace std;

template <class T>
static auto Inst(T &t, size_t idx) -> decltype(t->getInstrs().begin()) {
  auto cur_inst = t->getInstrs().begin();
  std::advance(cur_inst, idx);
  return cur_inst;
}

class LoadStoreTest : public ::testing::Test {
protected:
  void SetUp() override {
    F = mod.createFunction("TestHIR");
    auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 32, 32, 3},
                                        "input", false);
    ElemKind inputTy = input->getType()->getElementType();
    auto *filter =
        mod.createPlaceholder(inputTy, {16, 5, 5, 3}, "filter", true);
    auto *bias = mod.createPlaceholder(inputTy, {16}, "bias", true);
    auto OT = mod.uniqueType(inputTy, {1, 32, 32, 16});

    auto *C = F->addNode(new SophonConvolutionQ8Node("conv", OT, input, filter,
                                                     bias, {1, 1}, {2, 2, 2, 2},
                                                     {0, 0}, 0, false));
    auto *R = F->addNode(new SophonReluQ8Node("relu", OT, C));
    auto *S = F->createSave("ret2", R);
    IR = llvm::make_unique<IRFunction>(F);
    IR->generateIR();
    glow::optimize(*IR, true);
  }

  void TearDown() override {
    auto size = IR->getInstrs().size();
    IRBuilder B(IR.get()); // parser Function again for finding missing Inst,
                           // ex. deallocactivation
    EXPECT_EQ(size, IR->getInstrs().size());
  }

  Module mod;
  Function *F;
  std::unique_ptr<IRFunction> IR;
};

#define DEBUG_TYPE "InsertLoadStorePass"
TEST_F(LoadStoreTest, InsertLoadStorePass) {
  DEBUG_GLOW(IR->dump());
  EXPECT_EQ(4, IR->getInstrs().size());
  sophon::runInsertLoadStorePass(IR.get());
  DEBUG_GLOW(IR->dump());
  EXPECT_EQ(16, IR->getInstrs().size());
  EXPECT_TRUE(llvm::isa<glow::SophonStoreInst>(Inst(IR, 10)));
}

#define DEBUG_TYPE "InsertLoadStorePass"
TEST_F(LoadStoreTest, InsertLoadStorePassTwice) {
  EXPECT_EQ(4, IR->getInstrs().size());
  sophon::runInsertLoadStorePass(IR.get());
  sophon::runInsertLoadStorePass(IR.get());
  DEBUG_GLOW(IR->dump());
  EXPECT_EQ(16, IR->getInstrs().size());
}

class ReshapeTest : public ::testing::Test {
protected:
  void SetUp() override {
    F = mod.createFunction("TestHIR");
    auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, 32, 32, 3},
                                        "input", false);
    ElemKind inputTy = input->getType()->getElementType();
    auto *filter =
        mod.createPlaceholder(inputTy, {16, 5, 5, 3}, "filter", true);
    auto *bias = mod.createPlaceholder(inputTy, {16}, "bias", true);
    auto OT = mod.uniqueType(inputTy, {1, 32, 32, 16});

    auto *C = F->addNode(new SophonConvolutionQ8Node("conv", OT, input, filter,
                                                     bias, {1, 1}, {2, 2, 2, 2},
                                                     {0, 0}, 0, false));
    auto OT2 = mod.uniqueType(inputTy, {1, 32, 16, 32});
    auto *RE = F->addNode(new ReshapeNode("reshape", OT2, C, {1, 32, 16, 32}));
    auto *R = F->addNode(new SophonReluQ8Node("relu", OT, RE));
    auto *S = F->createSave("ret2", R);
    IR = llvm::make_unique<IRFunction>(F);
    IR->generateIR();
    glow::optimize(*IR, true);
  }

  Module mod;
  Function *F;
  std::unique_ptr<IRFunction> IR;
};

TEST_F(ReshapeTest, InsertLoadStorePass) {
  IR->dump();
  EXPECT_EQ(5, IR->getInstrs().size());
  EXPECT_TRUE(llvm::isa<TensorViewInst>(Inst(IR, 2)));

  sophon::runHandleReshape(IR.get());

  IR->dump();
  EXPECT_EQ(8, IR->getInstrs().size());
  EXPECT_TRUE(llvm::isa<SophonStoreInst>(Inst(IR, 2)));
  EXPECT_TRUE(llvm::isa<SophonLoadInst>(Inst(IR, 4)));
}
