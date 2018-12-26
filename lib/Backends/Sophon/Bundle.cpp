/*
 * bmnet/lib/Backends/Sophon/Bundle.cpp
 *
 * Copyright Bitmain Technologies Inc.
 * Written by:
 *   Wanwei CAI <wanwei.cai@bitmain.com>
 * Created Time: 2018-10-13 17:24
 */

#define DEBUG_TYPE "bundle_saver"

#include "Bundle.h"
#include "CommandLine.h"
#include "SophonBackend.h"

#include "glow/Graph/Graph.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Debug.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include <bmnet/utils/io.hpp>

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

Bundle::Bundle(const SophonBackend *backend, AllocationsInfo &allocationsInfo)
    : backend_(backend), allocationsInfo_(allocationsInfo) {}

void Bundle::getInputs(IRFunction *F, InputList &inputs) {
  for (auto v : F->getGraph()->getParent()->getPlaceholders()) {
    if (v->getName().find("save_") != llvm::StringRef::npos)
      continue;
    inputs.push_back(v);
  }
}

void Bundle::getOutputs(IRFunction *F, OutputList &outputs) {
  for (auto v : F->getGraph()->getParent()->getPlaceholders()) {
    if (v->getName().find("save_") == llvm::StringRef::npos)
      continue;
    if (v->getName().find("_sophon_spill") != llvm::StringRef::npos)
      continue;
    outputs.push_back(v);
  }
}

static uint32_t glueGetFmt(ElemKind kind) {
  switch (kind) {
  case ElemKind::FloatTy:
    return FMT_F32;
  case ElemKind::Float16Ty:
    return FMT_F16;
  case ElemKind::Int8QTy:
    return FMT_I8;
  case ElemKind::Int16QTy:
    return FMT_I16;
  case ElemKind::Int32QTy:
    return FMT_I32;
  default:
    llvm_unreachable("type not support now");
    break;
  }
}

std::unique_ptr<bmodel::Model> Bundle::produceBmodel(IRFunction *F) {

  std::unique_ptr<bmodel::Model> model(new bmodel::Model());
  float threshold = 0.0f;

  model->set_net_name(F->getGraph()->getName());
  uint32_t target = backend_->getTarget();
  model->set_chip(target);
  // fmt here is default value
  switch (target) {
  case 1682:
  case 1684:
    model->set_fmt(FMT_F32);
    break;
  case 1880:
  case 1882:
    model->set_fmt(FMT_I8);
    break;
  default:
    llvm_unreachable("chip not support now");
    break;
  }

  // inputs
  InputList inputs;
  getInputs(F, inputs);

  auto command = model->add_command();
  for (auto &v : inputs) { // support multi input
    auto input = command->add_input();
    for (auto &dim : v->getType()->dims()) {
      input->mutable_shape()->add_dim(dim);
      input->set_threshold(threshold); // 1880 need
      // input->set_fmt(glueGetFmt(v->getElementType()));
    }
  }

  // outputs
  OutputList outputs;
  getOutputs(F, outputs);
  for (auto v : outputs) {
    auto *o = cast<WeightVar>(F->getWeightForNode(v));
    size_t output_offset = allocationsInfo_.getAllocatedAddress()[o];
    auto output = command->add_output();
    output->set_name(v->getName());
    // output->set_threshold(0.0f); // 1880 need
    output->set_offset(output_offset);
    auto dimSize = v->getType()->dims().size();
    if (dimSize != 4 && dimSize != 2) {
      llvm_unreachable("Unsupported output shape");
    }
    for (auto dim : v->getType()->dims()) {
      output->mutable_shape()->add_dim(dim);
    }
    // output->set_fmt(glueGetFmt(v->getElementType()));
  }

  size_t total_neuron_size = allocationsInfo_.getActivationsMemSize();
  command->set_neuron_size(total_neuron_size);
  *(command->mutable_cmdbuf()) = {cmdbuf_.begin(), cmdbuf_.end()};
  auto weight = model->add_weight();
  *weight = {u8_weights_.begin(), u8_weights_.end()};

#if 0 // for debug
  std::string file("/tmp/glow.cmdbuf");
  bmnet::WriteFloatDataToBinaryFile(cg.getCmdbuf().data(), cg.getCmdbuf().size(), file);
  std::string file1("/tmp/glow.weight");
  bmnet::WriteFloatDataToBinaryFile(weight_u8.data(), weight_u8.size(), file1);
#endif

#if 0
  // cpu layer here
  findCpuLayers(cpu_layers);
  VLOG(2) << "cpu_layers: " << cpu_layers.size();
#endif

  return model;
}

void Bundle::saveBmodelFile(std::unique_ptr<bmodel::Model> model,
                            const std::string &outputDir) {
  assert(model != nullptr);
  DEBUG_GLOW(bmodel::print(*model));

  auto fileName = outputDir + bmodelFileNameOpt;
  bmerr_t ret = bmodel::save(*model, fileName);
  assert(ret == BM_SUCCESS);

  DEBUG_GLOW(llvm::dbgs() << "Save bmodel to: " << fileName << "\n");
}

std::unique_ptr<bmodel::Model> Bundle::codegen(IRFunction *F) {
  // generate weight
  backend_->generateWeights(F, allocationsInfo_, u8_weights_);

  // generate cmdbuf
  backend_->codeGenCmdbuf(F, allocationsInfo_, cmdbuf_);

  auto model = produceBmodel(F);
  return model;
}
