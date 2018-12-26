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
#define DEBUG_TYPE "bm1880-alloc"

#include "BM1880AllocationsInfo.h"
#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Debug.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

BM1880AllocationsInfo::BM1880AllocationsInfo() : ctx_(nullptr), TTI_(nullptr) {}

BM1880AllocationsInfo::BM1880AllocationsInfo(
    const Context &ctx, const sophon::SophonTargetTransformInfo *TTI)
    : ctx_(&ctx), TTI_(TTI) {}

void BM1880AllocationsInfo::allocateWeightVars(const IRFunction *F) {

  size_t weightOffset = 0;
  // Compute the new offsets for all the weights, do not reuse their current
  // addresses. Process all constant WeightVars first.
  for (auto &v : F->getGraph()->getParent()->getConstants()) {
    assert(isa<WeightVar>(F->getWeightForNode(v)));
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    auto numBytes = w->getSizeInBytes();
    // size_t addr = constantWeightVarsAllocator.allocate(numBytes, w);
    size_t addr = weightOffset;
    weightOffset += numBytes;
    allocatedAddressed_[w] = addr;

    DEBUG_GLOW(llvm::errs() << "Allocated global weight " << w->getName()
                            << " size: " << numBytes << "  address range:  ["
                            << addr << ", " << addr + numBytes << "]\n");
  }

  // Remember that max required memory size for each kind of weights.
  globa_weight_sizes_ = weightOffset;
}

void BM1880AllocationsInfo::allocateNeuron(WeightVar *W, size_t &neuronOffset) {
  auto numBytes = W->getSizeInBytes();
  // size_t addr = activationsAllocator.allocate(numBytes, w);
  size_t addr = neuronOffset;
  neuronOffset += numBytes;
  allocatedAddressed_[W] = addr;

  DEBUG_GLOW(llvm::errs() << "Allocated global input/output " << W->getName()
                          << " size: " << numBytes << "  address range:  ["
                          << addr << ", " << addr + numBytes << "]\n");
}

void BM1880AllocationsInfo::allocateActivations(const IRFunction *F) {
  // Maps activations and views to some offset within the heap.
  llvm::DenseMap<const Value *, uint64_t> activationAddr;

  // global offset start from 0, input data need to start from 0
  size_t neuronOffset = 0;
  // allocate input
  for (auto &v : F->getGraph()->getParent()->getPlaceholders()) {
    assert(isa<WeightVar>(F->getWeightForNode(v)));
    // unfortunately we need to use prefix to separate input/output
    if (v->getName().find("save_") != llvm::StringRef::npos)
      continue;

    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    allocateNeuron(w, neuronOffset);
  }

  // alloc output.
  for (auto &v : F->getGraph()->getParent()->getPlaceholders()) {
    assert(isa<WeightVar>(F->getWeightForNode(v)));
    if (v->getName().find("save_") == llvm::StringRef::npos)
      continue;
    auto *w = cast<WeightVar>(F->getWeightForNode(v));
    allocateNeuron(w, neuronOffset);
  }

  global_neuron_sizes_ = neuronOffset;

  MemoryAllocator activationsAllocator("activations",
                                       TTI_->getLocalMemSizeInBytes());

  // Assign device-space addresses to the activations.
  for (auto &I : F->getInstrs()) {
    if (auto *A = dyn_cast<AllocActivationInst>(&I)) {
      // FIXME, getLMemSizeFromValue can not accept const variable currently
      auto *nonConstI = const_cast<glow::Instruction *>(&I);
      auto numBytes = TTI_->getLMemSizeFromValue(nonConstI);

      // default MemoryAllocator is always 64-bytes aligned which has meet
      // ours eu_num(16) alignment requirement
      // TODO it's not efficieny for memory usage. support unaligned alloc
      uint64_t addr = activationsAllocator.allocate(numBytes, A);
      assert(!activationAddr.count(A) && "Allocation already made!");
      assert(MemoryAllocator::npos != addr);
      activationAddr[A] = addr;
      continue;
    }

    if (auto *D = dyn_cast<DeallocActivationInst>(&I)) {
      auto *A = D->getAlloc();
      assert(activationAddr.count(A) && "Invalid deallocation!");
      activationsAllocator.deallocate(A);
      continue;
    }
  }

  local_memory_sizes_ = activationsAllocator.getMaxMemoryUsage();

  // Register specific addresses within the heap to activations.
  for (auto &A : activationAddr) {
    allocatedAddressed_[A.first] = A.second;
    uint64_t size =
        TTI_->getLMemSizeFromValue(const_cast<glow::Value *>(A.first));
    DEBUG_GLOW(llvm::errs() << "Allocated activation " << A.first->getName()
                            << " size: " << size << "  address range:  ["
                            << allocatedAddressed_[A.first] << ", "
                            << allocatedAddressed_[A.first] + size << "]\n";);
  }
}

llvm::DenseMap<const Value *, uint64_t> &
BM1880AllocationsInfo::getAllocatedAddress() {
  return allocatedAddressed_;
}

void BM1880AllocationsInfo::setActivationsMemSize(size_t v) {
  global_neuron_sizes_ = v;
}

size_t BM1880AllocationsInfo::getActivationsMemSize() {
  return global_neuron_sizes_;
}
