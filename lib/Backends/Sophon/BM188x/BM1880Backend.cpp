/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "BM1880Backend.h"
#include "BM1880AllocationsInfo.h"
#include "BM1880CodeGen.h"
#include "BM1880DumpAllPass.h"
#include "BM1880ExpandSophonInst.h"
#include "BM1880HandleReshapePass.h"
#include "BM1880InsertLoadStorePass.h"
#include "BM1880MemoryAllocPass.h"
#include "BM1880TargetTransformInfo.h"
#include "Backends/Sophon/Bundle.h"
#include "Backends/Sophon/CommandLine.h"
#include "Backends/Sophon/SophonFunction.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/NodeValue.h"
#include "glow/Optimizer/Optimizer.h"
#include "glow/Support/Debug.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "bm1880_backend"

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;

extern llvm::cl::opt<Sophon::Target> target;
static llvm::cl::opt<bool> dumpAll("dump-all-neuron",
                                   llvm::cl::desc("dump all neuron"),
                                   llvm::cl::init(false));

std::unique_ptr<CompiledFunction>
BM1880Backend::codegen(std::unique_ptr<IRFunction> IR,
                       AllocationsInfo *allocationsInfo) const {
  auto model = Bundle(this, *allocationsInfo).codegen(IR.get());
  return llvm::make_unique<SophonFunction>(std::move(model));
}

std::unique_ptr<CompiledFunction>
BM1880Backend::compileIR(std::unique_ptr<IRFunction> IR,
                         const Context &ctx) const {
  BM1880AllocationsInfo allocationsInfo(ctx, getTTI());
  runOptimizationPasses(IR.get(), &allocationsInfo);
  return codegen(std::move(IR), &allocationsInfo);
}

std::unique_ptr<CompiledFunction>
BM1880Backend::compile(Function *F, const Context &ctx) const {
  auto IR = generateAndOptimizeIR(F, true /*shouldShareBuffers*/);
  return compileIR(std::move(IR), ctx);
}

void BM1880Backend::save(Function *F, llvm::StringRef outputDir,
                         llvm::StringRef networkName) const {
  auto IR = generateAndOptimizeIR(F, true /*shouldShareBuffers*/);
  Context ctx;
  BM1880AllocationsInfo allocationsInfo(ctx, getTTI());
  runOptimizationPasses(IR.get(), &allocationsInfo);
  auto b = Bundle(this, allocationsInfo);
  auto model = b.codegen(IR.get());
  Bundle::saveBmodelFile(std::move(model), outputDir);
}

bool BM1880Backend::isOpSupported(Kinded::Kind opKind,
                                  ElemKind elementTy) const {
  // Check for quantization support.
  if (elementTy == ElemKind::Int8QTy) {
    switch (opKind) {
    case Kinded::Kind::SophonConvolutionNodeKind:
    case Kinded::Kind::FullyConnectedNodeKind:
    case Kinded::Kind::SophonReluNodeKind:
    case Kinded::Kind::ReshapeNodeKind:
    case Kinded::Kind::SophonMaxPoolNodeKind:
      return true;
    default:
      return false;
    }
  }
  return false;
}

bool BM1880Backend::shouldLower(const Node *N) const {
  switch (N->getKind()) {
  default:
    return true;
  case Kinded::Kind::ConvolutionNodeKind:
  case Kinded::Kind::FullyConnectedNodeKind:
  case Kinded::Kind::ReluNodeKind:
  case Kinded::Kind::BatchNormalizationNodeKind:
    return false;
  }
}

static void reorderConvWeight(const SophonConvolutionQ8Inst *Inst,
                              std::vector<uint8_t> &Vec) {
  std::vector<uint8_t> ref_data = Vec;

  auto in_dim = Inst->getSrc()->getType()->dims();
  auto out_dim = Inst->getDest()->getType()->dims();
  auto kern_dim = Inst->getFilter()->getType()->dims();

  const int oc = out_dim[1];
  const int kh = kern_dim[2];
  const int kw = kern_dim[3];
  // support group size > 1?
  const int group_size = 1;
  const int ic = in_dim[1] / group_size;
  // conv weight is arranged by (1, oc, kh*kw, ic)
  // convert (oc, ic, kh, kw) to (1, oc, kh*kw, ic)
  for (int oc_i = 0; oc_i < oc; ++oc_i) {
    for (int k_i = 0; k_i < kh * kw; ++k_i) {
      for (int ic_i = 0; ic_i < ic; ++ic_i) {
        int to = oc_i * (ic * kh * kw) + k_i * ic + ic_i;
        int from = oc_i * (ic * kh * kw) + ic_i * (kh * kw) + k_i;
        Vec[to] = ref_data[from];
      }
    }
  }
}

static void reorder16bit(std::vector<uint8_t> &Vec) {
  std::vector<uint8_t> ref_data = Vec;
  assert(Vec.size() % 2 == 0);
  size_t count = Vec.size() / 2;
  for (size_t i = 0; i < count; ++i) {
    Vec[i] = ref_data[i * 2];
    Vec[i + count] = ref_data[i * 2 + 1];
  }
}

void BM1880Backend::reorderWeights(IRFunction *F) const {
  for (auto &v : F->getGraph()->getParent()->getConstants()) {
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    auto numBytes = w->getSizeInBytes();
    auto payload = v->getPayload().getUnsafePtr();

    if (w->getElementType() == glow::ElemKind::Int8QTy) {
      if (auto *conv =
              dyn_cast<SophonConvolutionQ8Inst>(w->getUsers().begin()->get())) {
        std::vector<uint8_t> orig_weight((uint8_t *)payload,
                                         (uint8_t *)(payload + numBytes));
        reorderConvWeight(conv, orig_weight);
        memcpy(payload, orig_weight.data(), numBytes);
      }
    } else if (w->getElementType() == glow::ElemKind::Int16QTy) {
      std::vector<uint8_t> orig_weight((uint8_t *)payload,
                                       (uint8_t *)(payload + numBytes));
      reorder16bit(orig_weight);
      memcpy(payload, orig_weight.data(), numBytes);
    } else {
      llvm_unreachable("unsupport type!");
    }
  }
}

void BM1880Backend::generateWeights(IRFunction *F,
                                    AllocationsInfo &allocationsInfo,
                                    std::vector<uint8_t> &weights) const {
  size_t weights_total_bytes = 0;
  for (auto &v : F->getGraph()->getParent()->getConstants()) {
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    weights_total_bytes += w->getSizeInBytes();
  }

  weights.resize(weights_total_bytes);

  DEBUG_GLOW(llvm::dbgs() << "generateWeights:\n");
  for (auto &v : F->getGraph()->getParent()->getConstants()) {
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    auto numBytes = w->getSizeInBytes();
    auto payload = v->getPayload().getUnsafePtr();
    auto addr = allocationsInfo.getAllocatedAddress()[w];
    memcpy(&(weights.data()[addr]), (uint8_t *)payload, numBytes);
    DEBUG_GLOW(llvm::dbgs()
               << "weights[" << addr << "]=" << w->getName() << "\n");
  }
}

void BM1880Backend::codeGenCmdbuf(IRFunction *F,
                                  AllocationsInfo &allocationsInfo,
                                  SophonCmdBuf &cmdbuf) const {
  auto codegen = BM1880CodeGen::createCodeGen(F, allocationsInfo);
  codegen->performCodeGen();
  cmdbuf = codegen->getCmdbuf();
}

void BM1880Backend::runOptimizationPasses(
    IRFunction *IR, BM1880AllocationsInfo *allocationsInfo) const {
  sophon::runHandleReshape(IR);
  reorderWeights(IR); // before insert
  sophon::runInsertLoadStorePass(IR);
  if (dumpAll)
    sophon::runDumpAllPass(IR);
  glow::optimize(*IR, true /*shouldShareBuffers*/);
  sophon::runMemoryAllocPass(IR, allocationsInfo);
  BM1880ExpandSophonInst(IR, *allocationsInfo).run();
}

sophon::SophonTargetTransformInfo *BM1880Backend::getTTI() const {
  return sophon::BM1880TargetTransformInfo::getInstance();
}

bool BM1880Backend::deleteQuantizeNodes(Function *F) const {
  auto *module = F->getParent();

  DEBUG_GLOW(F->dump());
  {
    DEBUG_GLOW(PlaceholderList &all_PHs_ = module->getPlaceholders();
               llvm::dbgs() << "init:\n"; for (auto &ph
                                               : all_PHs_) {
                 llvm::dbgs() << "PH: " << ph->getName() << "\n";
               });
  }

  bool changed = false;
  llvm::SmallPtrSet<Node *, 8> eraseNodes;
  // map[oldPH] = newPH
  std::map<Placeholder *, Placeholder *> PH_map;
  for (auto &n : F->getNodes()) {
    if (n.getKind() == Kinded::Kind::QuantizeNodeKind) {
      // create new PH and delete quantize node
      auto *QN = llvm::cast<QuantizeNode>(&n);
      auto old_PH = cast<Placeholder>(QN->getInput());
      auto *new_PH =
          module->createPlaceholder(QN->getType(0), old_PH->getName(), false);
      QN->getResult().replaceAllUsesOfWith(new_PH, F);
      eraseNodes.insert(QN);
      changed = true;
      PH_map.insert({old_PH, new_PH});
    } else if (n.getKind() == Kinded::Kind::DequantizeNodeKind) {
      // assume all users is Save node, create a new one with Q8 type
      // delete Dequantize and Save nodes
      auto *DQN = llvm::cast<DequantizeNode>(&n);
      auto *SN = llvm::cast<SaveNode>(DQN->getUsers().begin()->getUser());
      auto old_PH = SN->getPlaceholder();
      auto *new_save = F->createSave(old_PH->getName(), DQN->getInput());
      eraseNodes.insert(SN);
      eraseNodes.insert(DQN);
      PH_map.insert({old_PH, new_save->getPlaceholder()});
      changed = true;
    }
  }

  {
    DEBUG_GLOW(PlaceholderList &all_PHs_ = module->getPlaceholders();
               llvm::dbgs() << "before:\n"; for (auto &ph
                                                 : all_PHs_) {
                 llvm::dbgs() << "PH: " << ph->getName() << "\n";
               });
  }

  // delete old PHs and update new PHs name
  PlaceholderList &all_PHs = module->getPlaceholders();
  for (auto &map : PH_map) {
    std::string old_name = map.first->getName();
    map.second->setName(old_name);
    all_PHs.remove(map.first);
    delete (map.first);
  }

  {
    DEBUG_GLOW(PlaceholderList &all_PHs_ = module->getPlaceholders();
               llvm::dbgs() << "after:\n"; for (auto &ph
                                                : all_PHs_) {
                 llvm::dbgs() << "PH: " << ph->getName() << "\n";
               });
  }

  for (auto *node : eraseNodes) {
    F->eraseNode(node);
  }

  DEBUG_GLOW(F->dump());
  return changed;
}

bool BM1880Backend::transformPreLowering(Function *F,
                                         CompilationMode mode) const {
  bool changed;
  changed = deleteQuantizeNodes(F);
  return changed;
}
