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
#define DEBUG_TYPE "BM188xLMemSizeVisitor"

#include "BM188xLMemSizeVisitor.h"
#include "BM1880TargetTransformInfo.h"
#include "Backends/Sophon/Utility/memory.h"
#include "glow/IR/IRUtils.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Debug.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <vector>

using llvm::cast;
using llvm::dyn_cast;

namespace glow {
namespace sophon {
BM188xLMemSizeVisitor::BM188xLMemSizeVisitor() {
  npu_num_ =
      glow::sophon::BM1880TargetTransformInfo::getInstance()->getNPUNum();
  eu_num_ = glow::sophon::BM1880TargetTransformInfo::getInstance()->getEUNum();
}

size_t BM188xLMemSizeVisitor::calFCBias(glow::Value *value) {
  auto orig_dim = value->getType()->dims();
  SophonFCBiasDim dim(orig_dim);
  size_t sz =
      glow::sophon::formula_4d_nonaligned(dim.n, dim.c, dim.h, dim.w, npu_num_);
  return sz;
}

size_t BM188xLMemSizeVisitor::calAligned(glow::Value *value) {
  auto orig_dim = value->getType()->dims();
  glow::sophon::SophonDim dim(orig_dim);
  size_t sz = glow::sophon::formula_4d_aligned(dim.n, dim.c, dim.h, dim.w,
                                               npu_num_, eu_num_);
  return sz;
}

size_t BM188xLMemSizeVisitor::calNonAligned(glow::Value *value) {
  auto orig_dim = value->getType()->dims();
  glow::sophon::SophonDim dim(orig_dim);
  size_t sz =
      glow::sophon::formula_4d_nonaligned(dim.n, dim.c, dim.h, dim.w, npu_num_);
  return sz;
}

template <typename T>
std::vector<size_t> BM188xLMemSizeVisitor::calGeneralAligned(T *inst) {
  std::vector<size_t> operand_size;
  unsigned opnd_num = inst->getNumOperands();
  for (unsigned i = 0; i < opnd_num; i++) {
    auto *value = inst->getOperand(i).first;
    size_t sz = calAligned(value);
    operand_size.push_back(sz);
  }
  return operand_size;
}

template <typename T>
void BM188xLMemSizeVisitor::calOperand(T *inst, std::vector<size_t> *result,
                                       const int opnd_id, bool aligned) {
  size_t sz;
  auto *value = inst->getOperand(opnd_id).first;
  if (aligned)
    sz = calAligned(value);
  else
    sz = calNonAligned(value);
  result->push_back(sz);
}

template <int ID, typename T, typename Type, typename... Types>
void BM188xLMemSizeVisitor::calEach(T *inst, std::vector<size_t> *result,
                                    const Type &oprnd,
                                    const Types &... oprnds) {

  calOperand(inst, result, ID, oprnd);
  calEach<ID + 1>(inst, result, oprnds...);
}

template <int ID, typename T, typename Type>
void BM188xLMemSizeVisitor::calEach(T *inst, std::vector<size_t> *result,
                                    const Type &oprnd) {
  calOperand(inst, result, ID, oprnd);
}

template <typename T, typename Type, typename... Types>
std::vector<size_t>
BM188xLMemSizeVisitor::calOperandList(T *inst, const Type &oprnd,
                                      const Types &... oprnds) {
  std::vector<size_t> operand_size;
  calEach<0>(inst, &operand_size, oprnd, oprnds...);
  return operand_size;
}

void BM188xLMemSizeVisitor::default_method(glow::Instruction *Inst) {
  DEBUG_GLOW(llvm::dbgs() << "unsupport Instr " << Inst->getKindName() << " "
                          << Inst->getName() << "\n");
  oprnd_size_.clear();
  assert(false && "getLMemSize: Unsupport Instrution");
}

//
// SophonOp
//
void BM188xLMemSizeVisitor::visit(glow::SophonReluQ8Inst *Inst) {
  oprnd_size_ = calGeneralAligned(Inst);
}

void BM188xLMemSizeVisitor::visit(glow::SophonAvgPoolQ8Inst *Inst) {
  oprnd_size_ = calGeneralAligned(Inst);
}

void BM188xLMemSizeVisitor::visit(glow::SophonMaxPoolQ8Inst *Inst) {
  oprnd_size_ = calGeneralAligned(Inst);
}

void BM188xLMemSizeVisitor::visit(glow::SophonConvolutionQ8Inst *Inst) {
  oprnd_size_ = calOperandList(Inst, ALIGN, ALIGN, /* weight */ NONALIGN,
                               /* bias */ NONALIGN);
}

void BM188xLMemSizeVisitor::visit(glow::SophonFullyConnectedQ8Inst *Inst) {
  oprnd_size_ = calOperandList(Inst, ALIGN, ALIGN, /* weight */ ALIGN);
  // handle special case: fc bias
  size_t sz_bias = calFCBias(Inst->getOperand(3).first);
  oprnd_size_.push_back(sz_bias);
}

size_t BM188xLMemSizeVisitor::calValueSize(glow::Value *Value,
                                           bool calByOutUser) {
  for (auto &use : Value->getUsers()) {

    if (calByOutUser && use.getOperand().second != OperandKind::Out) {
      continue;
    }

    // ignore Out user
    if (!calByOutUser && use.getOperand().second == OperandKind::Out) {
      continue;
    }
    auto *instr = use.get();
    accept_helper(instr);
    auto opnd_size = getResult();
    auto opnd_idx = use.idx_;

    if (opnd_idx < opnd_size.size()) {
      size_t lmem_size = opnd_size.at(opnd_idx);
      if (lmem_size > 0) {
        return lmem_size;
      }
    }
  }
}

void BM188xLMemSizeVisitor::visit(glow::SophonLoadInst *Inst) {
  // load dest, src
  // load global to dest local

  // dest size depend by memory shape
  size_t dest_size = calValueSize(Inst->getDest());

  // src size depend by tensor shape
  auto *w = cast<WeightVar>(Inst->getSrc());
  size_t src_size = w->getSizeInBytes();

  // clear oprnd_size_ used by calValueSize
  oprnd_size_.clear();
  oprnd_size_.push_back(dest_size);
  oprnd_size_.push_back(src_size);
}

void BM188xLMemSizeVisitor::visit(glow::SophonStoreInst *Inst) {
  // store dest, src
  // store local to global

  // dest size depend by tensor shape
  auto *w = cast<WeightVar>(Inst->getDest());
  size_t dest_size = w->getSizeInBytes();

  // src size depend by memory shape
  // because the input user of store Inst is itself.
  // we need to use output user to decide which value to calculate size
  size_t src_size = calValueSize(getOrigin(Inst->getSrc()), true);

  // clear oprnd_size_ used by calValueSize
  oprnd_size_.clear();
  oprnd_size_.push_back(dest_size);
  oprnd_size_.push_back(src_size);
}

//
// SophonMI
//
void BM188xLMemSizeVisitor::visit(glow::SophonMIGDMAGlobalToLocalInst *Inst) {
  oprnd_size_.clear();
}

void BM188xLMemSizeVisitor::visit(glow::SophonMIGDMALocalToGlobalInst *Inst) {
  oprnd_size_.clear();
}

void BM188xLMemSizeVisitor::visit(glow::SophonMIMulConstQ16Inst *Inst) {
  oprnd_size_ = calGeneralAligned(Inst);
}

void BM188xLMemSizeVisitor::visit(glow::SophonMIMulConstQ8Inst *Inst) {
  oprnd_size_ = calGeneralAligned(Inst);
}

void BM188xLMemSizeVisitor::visit(glow::SophonMIMacConstQ8Inst *Inst) {
  oprnd_size_ = calGeneralAligned(Inst);
}

void BM188xLMemSizeVisitor::visit(glow::SophonMIReluQ8Inst *Inst) {
  oprnd_size_ = calGeneralAligned(Inst);
}

void BM188xLMemSizeVisitor::visit(glow::SophonMIAvgPoolingQ8Inst *Inst) {
  oprnd_size_ = calGeneralAligned(Inst);
}

void BM188xLMemSizeVisitor::visit(glow::SophonMIMaxPoolingQ8Inst *Inst) {
  oprnd_size_ = calGeneralAligned(Inst);
}

void BM188xLMemSizeVisitor::visit(glow::SophonMIConvolutionQ8Inst *Inst) {
  oprnd_size_ = calOperandList(Inst, ALIGN, ALIGN, /* weight */ NONALIGN,
                               /* bias */ NONALIGN);
}

void BM188xLMemSizeVisitor::visit(
    glow::SophonMIDepthwiseConvolutionQ8Inst *Inst) {
  oprnd_size_ = calOperandList(Inst, ALIGN, ALIGN, /* weight */ ALIGN,
                               /* bias */ NONALIGN);
}

void BM188xLMemSizeVisitor::visit(glow::SophonMIFCQ8Inst *Inst) {
  oprnd_size_ = calOperandList(Inst, ALIGN, ALIGN, /* weight */ ALIGN);
  // handle special case: fc bias
  size_t sz_bias = calFCBias(Inst->getOperand(3).first);
  oprnd_size_.push_back(sz_bias);
}

void BM188xLMemSizeVisitor::visit(glow::SophonMIFCQ16Inst *Inst) {
  oprnd_size_ = calOperandList(Inst, ALIGN, ALIGN, /* weight */ ALIGN);
  // handle special case: fc bias
  size_t sz_bias = calFCBias(Inst->getOperand(3).first);
  oprnd_size_.push_back(sz_bias);
}

std::vector<size_t> BM188xLMemSizeVisitor::getResult() { return oprnd_size_; }

} // namespace sophon
} // namespace glow
