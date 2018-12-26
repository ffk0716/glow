#include "BM1880CodeGenBMK.h"
#include "glow/Support/Debug.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <assert.h>
#include <bmkernel/bm1880/bmkernel_1880.h>
#include <cassert>
#include <glow/IR/IRUtils.h>
#include <iostream>

#define DEBUG_TYPE "BM1880_codegenBMK"

namespace glow {

BM1880CodeGenBMK::BM1880CodeGenBMK(IRFunction *F, AllocationsInfo &allocInfo)
    : F_(F), allocInfo_(allocInfo) {
  cmdbuf_size_ = 0x10000000;
}

void BM1880CodeGenBMK::bmk_init() {
  bmk_info_.chip_version = 1880;
  bmk_info_.cmdbuf_size = cmdbuf_size_;
  bmk_info_.cmdbuf = (u8 *)malloc(bmk_info_.cmdbuf_size);
  bmk_ctx_ = nullptr;
  bmk_ctx_ = bmk1880_register(&bmk_info_);
}

void BM1880CodeGenBMK::bmk_deinit() {
  if (bmk_ctx_) {
    // hack api, update sync id
    u32 size;
    const u8 *cmdbuf = bmk1880_acquire_cmdbuf(bmk_ctx_, &size);
    cmdbuf_.resize(size);
    memcpy(&cmdbuf_[0], cmdbuf, size);
    bmk1880_cleanup(bmk_ctx_);
  }
  if (bmk_info_.cmdbuf) {
    free(bmk_info_.cmdbuf);
    bmk_info_.cmdbuf = nullptr;
  }
}

uint64_t BM1880CodeGenBMK::emitValueAddress(const glow::Value *val) {
  assert(allocInfo_.getAllocatedAddress().count(val));
  return allocInfo_.getAllocatedAddress()[val];
}

void BM1880CodeGenBMK::visit(const SophonMIGDMAGlobalToLocalInst *inst) {
  auto *global = inst->getGlobal();
  auto *local = inst->getLocal();
  auto addr_local = emitValueAddress(local);
  auto addr_global = emitValueAddress(global);

  auto shapeNCHW = inst->getShapeNCHW();
  auto globalStrideNCH = inst->getGlobalStrideNCH();
  int n = shapeNCHW[0];
  int c = shapeNCHW[1];
  int h = shapeNCHW[2];
  int w = shapeNCHW[3];
  int stride_n = globalStrideNCH[0];
  int stride_c = globalStrideNCH[1];
  int stride_h = globalStrideNCH[2];

  // TODO(arcbb): support LocalStride for user would be easier
  bool is_local_aligned = inst->getIsLocalAligned();
  bool in_weight_space = inst->getIsGlobalWeightSpace();

  // bmkernel code
  u64 gaddr = addr_global;
  laddr_t lmem_addr = addr_local;
  tensor_lmem *lmem;
  ctrl_t ctrls;

  if (is_local_aligned) {
    lmem = bmk1880_tl_prealloc_align(bmk_ctx_, lmem_addr, shape_t4(n, c, h, w),
                                     FMT_I8);
  } else {
    lmem =
        bmk1880_tl_prealloc(bmk_ctx_, lmem_addr, shape_t4(n, c, h, w), FMT_I8);
  }

  if (in_weight_space)
    ctrls = CTRL_WEIGHT;
  else
    ctrls = CTRL_NEURON;

  stride_t stride = stride_st4(stride_n, stride_c, stride_h, 1);
  bmk1880_gdma_load_stride(bmk_ctx_, lmem, gaddr, stride, ctrls);
  bmk1880_tl_free(bmk_ctx_, lmem);

  DEBUG_GLOW(
      llvm::dbgs() << llvm::format(
          "%d\tgdma_load_stride: local=0x%x, global=0x%x, shape=(%d,%d,%d,%d), "
          "stride=(%d,%d,%d,%d), aligned=%d, WEIGHT=%d\n",
          index++, addr_local, addr_global, n, c, h, w, stride_n, stride_c,
          stride_h, 1, is_local_aligned, in_weight_space));
}

void BM1880CodeGenBMK::visit(const SophonMIGDMALocalToGlobalInst *inst) {
  auto *global = inst->getGlobal();
  auto *local = inst->getLocal();
  auto addr_local = emitValueAddress(local);
  auto addr_global = emitValueAddress(global);

  std::vector<unsigned int> shapeNCHW = inst->getShapeNCHW();
  auto globalStrideNCH = inst->getGlobalStrideNCH();
  while (shapeNCHW.size() < 4)
    shapeNCHW.insert(shapeNCHW.begin(), 1);
  assert(shapeNCHW.size() == 4);
  int n = shapeNCHW[0];
  int c = shapeNCHW[1];
  int h = shapeNCHW[2];
  int w = shapeNCHW[3];
  assert(globalStrideNCH.size() == 3);
  int stride_n = globalStrideNCH[0];
  int stride_c = globalStrideNCH[1];
  int stride_h = globalStrideNCH[2];

  // TODO(arcbb): support LocalStride for user would be easier
  bool is_local_aligned = inst->getIsLocalAligned();
  bool in_weight_space = inst->getIsGlobalWeightSpace();

  // bmkernel code
  u64 gaddr = addr_global;
  laddr_t lmem_addr = addr_local;
  tensor_lmem *lmem;
  ctrl_t ctrls;

  if (is_local_aligned) {
    lmem = bmk1880_tl_prealloc_align(bmk_ctx_, lmem_addr, shape_t4(n, c, h, w),
                                     FMT_I8);
  } else {
    lmem =
        bmk1880_tl_prealloc(bmk_ctx_, lmem_addr, shape_t4(n, c, h, w), FMT_I8);
  }

  if (in_weight_space)
    ctrls = CTRL_WEIGHT;
  else
    ctrls = CTRL_NEURON;

  stride_t stride = stride_st4(stride_n, stride_c, stride_h, 1);
  bmk1880_gdma_store_stride(bmk_ctx_, lmem, gaddr, stride, ctrls);
  bmk1880_tl_free(bmk_ctx_, lmem);
  DEBUG_GLOW(llvm::dbgs() << llvm::format(
                 "%d\tgdma_store_stride: global=0x%x, local=0x%x, "
                 "shape=(%d,%d,%d,%d), "
                 "stride=(%d,%d,%d,%d), aligned=%d, WEIGHT=%d\n",
                 index++, addr_global, addr_local, n, c, h, w, stride_n,
                 stride_c, stride_h, 1, is_local_aligned, in_weight_space));
}

void BM1880CodeGenBMK::visit(const SophonMIMacConstQ8Inst *inst) {
  auto value_input = inst->getSrc();
  auto value_output_high = inst->getDestHigh();
  auto value_output_low = inst->getDestLow();

  auto addr_ifmap = emitValueAddress(value_input);
  auto addr_ofmap_high = emitValueAddress(value_output_high);
  auto addr_ofmap_low = emitValueAddress(value_output_low);

  auto in_dim = value_input->getType()->dims();
  auto out_dim = value_output_high->getType()->dims();

  int input_n;
  int ic;
  int ih;
  int iw;
  int oc;
  int oh;
  int ow;

  input_n = in_dim[0];
  ic = in_dim[1];
  ih = in_dim[2];
  iw = in_dim[3];

  oc = out_dim[1];
  oh = out_dim[2];
  ow = out_dim[3];

  int right_shift_width = inst->getRShiftWidth();
  int left_shift_width = inst->getLShiftWidth();
  bool res_is_int8 = inst->getIsResultI8();

  int multiplier = inst->getMultiplier();
  bool is_multiplier_signed = inst->getIsMultiplierSigned();

  // bmkernel begins
  tensor_lmem *output_high;
  tensor_lmem *output_low;
  tensor_lmem *input;

  output_high = bmk1880_tl_prealloc_align(
      bmk_ctx_, addr_ofmap_high, shape_t4(input_n, oc, oh, ow), FMT_I8);

  output_low = bmk1880_tl_prealloc_align(bmk_ctx_, addr_ofmap_low,
                                         shape_t4(input_n, oc, oh, ow), FMT_I8);

  input = bmk1880_tl_prealloc_align(bmk_ctx_, addr_ifmap,
                                    shape_t4(input_n, ic, ih, iw), FMT_I8);

  bmk1880_mac_const_param_t param;
  param.res_high = output_high;
  param.res_low = output_low;
  param.res_is_int8 = res_is_int8;
  param.a = input;
  param.b = multiplier;
  param.b_is_signed = is_multiplier_signed;
  param.lshift_width = left_shift_width;
  param.rshift_width = right_shift_width;
  bmk1880_tpu_mac_const(bmk_ctx_, &param);

  bmk1880_tl_free(bmk_ctx_, input);
  bmk1880_tl_free(bmk_ctx_, output_low);
  bmk1880_tl_free(bmk_ctx_, output_high);
  DEBUG_GLOW(llvm::dbgs() << llvm::format(
                 "%d\tmac_imm: output_high=0x%x, output_low=0x%x, input=0x%x, "
                 "imm=%d, output_shape=(%d,%d,%d,%d), "
                 "input_shape=(%d,%d,%d,%d), is_imm_signed=%d, "
                 "res_is_int8=%d, lshift=%d, rshift=%d\n",
                 index++, addr_ofmap_high, addr_ofmap_low, addr_ifmap,
                 multiplier, input_n, oc, oh, ow, input_n, ic, ih, iw,
                 is_multiplier_signed, res_is_int8, left_shift_width,
                 right_shift_width));
}

void BM1880CodeGenBMK::visit(const SophonMIMulConstQ8Inst *inst) {
  auto *src = inst->getSrc();
  auto value_output_low = inst->getDest();

  auto addr_ifmap = emitValueAddress(src);
  auto addr_ofmap_low = emitValueAddress(value_output_low);

  auto tensor_dim = src->getType()->dims();
  int n = tensor_dim[0];
  int c = tensor_dim[1];
  int h = tensor_dim[2];
  int w = tensor_dim[3];

  u8 b = inst->getMultiplier();
  bool b_is_signed = inst->getIsMultiplierSigned();
  int rshift_width = inst->getRShiftWidth();

  // bmkernel begins
  tensor_lmem *input;
  tensor_lmem *output_low;

  input = bmk1880_tl_prealloc_align(bmk_ctx_, addr_ifmap, shape_t4(n, c, h, w),
                                    FMT_I8);

  output_low = bmk1880_tl_prealloc_align(bmk_ctx_, addr_ofmap_low,
                                         shape_t4(n, c, h, w), FMT_I8);

  bmk1880_mul_const_param_t param;
  param.res_high = NULL;
  param.res_low = output_low;
  param.a = input;
  param.b = b;
  param.b_is_signed = b_is_signed;
  param.rshift_width = rshift_width;
  bmk1880_tpu_mul_const(bmk_ctx_, &param);
  bmk1880_tl_free(bmk_ctx_, output_low);
  bmk1880_tl_free(bmk_ctx_, input);
  DEBUG_GLOW(llvm::dbgs() << llvm::format(
                 "%d\tmul_imm: output_high=0x%x, output_low=0x%x, input=%x, "
                 "imm=%d, shape=(%d,%d,%d,%d), is_imm_signed=%d, rshift=%d\n",
                 index++, NULL, addr_ofmap_low, addr_ifmap, b, n, c, h, w,
                 b_is_signed, rshift_width));
}

void BM1880CodeGenBMK::visit(const SophonMIMulConstQ16Inst *inst) {
  auto *src = inst->getSrc();
  auto value_output_high = inst->getDestHigh();
  auto value_output_low = inst->getDestLow();

  auto addr_ifmap = emitValueAddress(src);
  auto addr_ofmap_high = emitValueAddress(value_output_high);
  auto addr_ofmap_low = emitValueAddress(value_output_low);

  auto tensor_dim = src->getType()->dims();
  int n = tensor_dim[0];
  int c = tensor_dim[1];
  int h = tensor_dim[2];
  int w = tensor_dim[3];

  u8 b = inst->getMultiplier();
  bool b_is_signed = inst->getIsMultiplierSigned();
  int rshift_width = inst->getRShiftWidth();

  // bmkernel begins
  tensor_lmem *input;
  tensor_lmem *output_high;
  tensor_lmem *output_low;

  input = bmk1880_tl_prealloc_align(bmk_ctx_, addr_ifmap, shape_t4(n, c, h, w),
                                    FMT_I8);

  output_high = bmk1880_tl_prealloc_align(bmk_ctx_, addr_ofmap_high,
                                          shape_t4(n, c, h, w), FMT_I8);

  output_low = bmk1880_tl_prealloc_align(bmk_ctx_, addr_ofmap_low,
                                         shape_t4(n, c, h, w), FMT_I8);

  bmk1880_mul_const_param_t param;
  param.res_high = output_high;
  param.res_low = output_low;
  param.a = input;
  param.b = b;
  param.b_is_signed = b_is_signed;
  param.rshift_width = rshift_width;
  bmk1880_tpu_mul_const(bmk_ctx_, &param);
  bmk1880_tl_free(bmk_ctx_, output_low);
  bmk1880_tl_free(bmk_ctx_, output_high);
  bmk1880_tl_free(bmk_ctx_, input);
  DEBUG_GLOW(llvm::dbgs() << llvm::format(
                 "%d\tmul_imm: output_high=%x, output_low=%x, input=%x, "
                 "imm=%d, shape=(%d,%d,%d,%d), is_imm_signed=%d, rshift=%d\n",
                 index++, addr_ofmap_high, addr_ofmap_low, addr_ifmap, b, n, c,
                 h, w, b_is_signed, rshift_width));
}

void BM1880CodeGenBMK::visit(const SophonMIReluQ8Inst *inst) {
  auto *dst = inst->getDest();
  auto *src = inst->getSrc();
  auto addr_ifmap = emitValueAddress(src);
  auto addr_ofmap = emitValueAddress(dst);

  auto tensor_dim = src->getType()->dims();

  // should assert input dim == output dim

  int n, c, h, w;
  if (tensor_dim.size() == 4) {
    n = tensor_dim[0];
    c = tensor_dim[1];
    h = tensor_dim[2];
    w = tensor_dim[3];
  } else {
    assert(tensor_dim.size() == 2);
    // FIXME, use getMemoryShape
    int M = tensor_dim[0];
    int N = tensor_dim[1];
    auto idiv_round = [](int Num, int Denominator) {
      return (Num + Denominator - 1) / Denominator;
    };

    if (N > 32) {
      w = 32;
    } else {
      w = 16;
    }
    n = M;
    h = 1;
    c = idiv_round(N, w);
  }

  // bmkernel code
  laddr_t in_addr = addr_ifmap;
  laddr_t out_addr = addr_ofmap;
  shape_t dim = shape_t4(n, c, h, w);

  tensor_lmem *input =
      bmk1880_tl_prealloc_align(bmk_ctx_, in_addr, dim, FMT_I8);
  tensor_lmem *output =
      bmk1880_tl_prealloc_align(bmk_ctx_, out_addr, dim, FMT_I8);
  bmk1880_relu_param_t relu_param;
  relu_param.ofmap = output;
  relu_param.ifmap = input;
  bmk1880_tpu_relu(bmk_ctx_, &relu_param);
  bmk1880_tl_free(bmk_ctx_, output);
  bmk1880_tl_free(bmk_ctx_, input);
  DEBUG_GLOW(llvm::dbgs() << llvm::format(
                 "%d\trelu: ouput=%x, input=%x, shape=(%d,%d,%d,%d)\n", index++,
                 addr_ofmap, addr_ifmap, n, c, h, w));
}

void BM1880CodeGenBMK::visit(const SophonMIAvgPoolingQ8Inst *inst) {
  auto *dst = inst->getDest();
  auto *src = inst->getSrc();
  auto addr_ifmap = emitValueAddress(src);
  auto addr_ofmap = emitValueAddress(dst);

  auto in_dim = src->getType()->dims();
  auto out_dim = dst->getType()->dims();

  int input_n;
  int ic;
  int ih;
  int iw;
  int oc;
  int oh;
  int ow;
  int kh;
  int kw;
  u8 stride_h;
  u8 stride_w;
  u8 pad_top;
  u8 pad_left;
  u8 pad_bottom;
  u8 pad_right;
  int rshift_width;

  input_n = in_dim[0];

  ic = in_dim[1];
  ih = in_dim[2];
  iw = in_dim[3];

  oc = out_dim[1];
  oh = out_dim[2];
  ow = out_dim[3];

  kh = inst->getKernelHW()[0];
  kw = inst->getKernelHW()[1];

  stride_h = inst->getStrideHW()[0];
  stride_w = inst->getStrideHW()[1];

  pad_top = inst->getPadTLBR()[0];
  pad_left = inst->getPadTLBR()[1];
  pad_bottom = inst->getPadTLBR()[2];
  pad_right = inst->getPadTLBR()[3];

  rshift_width = inst->getRShiftWidth();

  // bmkernel code begins
  tensor_lmem *output;
  tensor_lmem *input;

  output = bmk1880_tl_prealloc_align(bmk_ctx_, addr_ofmap,
                                     shape_t4(input_n, oc, oh, ow), FMT_I8);

  input = bmk1880_tl_prealloc_align(bmk_ctx_, addr_ifmap,
                                    shape_t4(input_n, ic, ih, iw), FMT_I8);

  bmk1880_avg_pooling_param_t param;
  param.ofmap = output;
  param.ifmap = input;
  param.kh = kh;
  param.kw = kw;
  param.ins_h = 0;
  param.ins_last_h = 0;
  param.ins_w = 0;
  param.ins_last_w = 0;
  param.pad_top = pad_top;
  param.pad_left = pad_left;
  param.pad_bottom = pad_bottom;
  param.pad_right = pad_right;
  param.stride_h = stride_h;
  param.stride_w = stride_w;
  param.avg_pooling_const = 0;
  param.rshift_width = rshift_width;
  bmk1880_tpu_avg_pooling(bmk_ctx_, &param);
  bmk1880_tl_free(bmk_ctx_, input);
  bmk1880_tl_free(bmk_ctx_, output);
  DEBUG_GLOW(
      llvm::dbgs() << llvm::format(
          "%d\tavg_pooling: ouput=0x%x, input=0x%x, out_shape=(%d,%d,%d,%d), "
          "input_shape=(%d,%d,%d,%d), kernel=(%d,%d), padTLBR=(%d,%d,%d,%d), "
          "strideHW=(%d,%d), avg_pooling_const=%d, rshift=%d\n",
          index++, addr_ofmap, addr_ifmap, input_n, oc, oh, ow, input_n, ic, ih,
          iw, kh, kw, pad_top, pad_left, pad_bottom, pad_right, stride_h,
          stride_w, 0, rshift_width));
}

void BM1880CodeGenBMK::visit(const SophonMIMaxPoolingQ8Inst *inst) {
  auto *dst = inst->getDest();
  auto *src = inst->getSrc();
  auto addr_ifmap = emitValueAddress(src);
  auto addr_ofmap = emitValueAddress(dst);

  auto in_dim = src->getType()->dims();
  auto out_dim = dst->getType()->dims();

  int input_n;
  int ic;
  int ih;
  int iw;
  int oc;
  int oh;
  int ow;
  int kh;
  int kw;
  u8 stride_h;
  u8 stride_w;
  u8 pad_top;
  u8 pad_left;
  u8 pad_bottom;
  u8 pad_right;

  input_n = in_dim[0];

  ic = in_dim[1];
  ih = in_dim[2];
  iw = in_dim[3];

  oc = out_dim[1];
  oh = out_dim[2];
  ow = out_dim[3];

  kh = inst->getKernelHW()[0];
  kw = inst->getKernelHW()[1];

  stride_h = inst->getStrideHW()[0];
  stride_w = inst->getStrideHW()[1];

  pad_top = inst->getPadTLBR()[0];
  pad_left = inst->getPadTLBR()[1];
  pad_bottom = inst->getPadTLBR()[2];
  pad_right = inst->getPadTLBR()[3];
  // bmkernel begins
  tensor_lmem *output;
  tensor_lmem *input;

  output = bmk1880_tl_prealloc_align(bmk_ctx_, addr_ofmap,
                                     shape_t4(input_n, oc, oh, ow), FMT_I8);

  input = bmk1880_tl_prealloc_align(bmk_ctx_, addr_ifmap,
                                    shape_t4(input_n, ic, ih, iw), FMT_I8);

  bmk1880_max_pooling_param_t param;
  param.ofmap = output;
  param.ifmap = input;
  param.kh = kh;
  param.kw = kw;
  param.pad_top = pad_top;
  param.pad_bottom = pad_bottom;
  param.pad_left = pad_left;
  param.pad_right = pad_right;
  param.stride_h = stride_h;
  param.stride_w = stride_w;

  bmk1880_tpu_max_pooling(bmk_ctx_, &param);
  bmk1880_tl_free(bmk_ctx_, input);
  bmk1880_tl_free(bmk_ctx_, output);
  DEBUG_GLOW(
      llvm::dbgs() << llvm::format(
          "%d\tmax_pooling: ouput=0x%x, input=0x%x, out_shape=(%d,%d,%d,%d), "
          "input_shape=(%d,%d,%d,%d), kernel=(%d,%d), padTLBR=(%d,%d,%d,%d), "
          "strideHW=(%d,%d)\n",
          index++, addr_ofmap, addr_ifmap, input_n, oc, oh, ow, input_n, ic, ih,
          iw, kh, kw, pad_top, pad_left, pad_bottom, pad_right, stride_h,
          stride_w));
}

void BM1880CodeGenBMK::visit(const SophonMIConvolutionQ8Inst *inst) {
  auto value_input = inst->getSrc();
  auto value_output = inst->getDest();
  auto value_bias = inst->getBias();
  auto value_filter = inst->getFilter();

  auto addr_ifmap = emitValueAddress(value_input);
  auto addr_ofmap = emitValueAddress(value_output);
  auto addr_bias = emitValueAddress(value_bias);
  auto addr_filter = emitValueAddress(value_filter);

  auto in_dim = value_input->getType()->dims();
  auto out_dim = value_output->getType()->dims();
  auto kern_dim = value_filter->getType()->dims();

  int input_n;
  int ic;
  int ih;
  int iw;
  int oc;
  int oh;
  int ow;
  int kh;
  int kw;
  u8 stride_h;
  u8 stride_w;
  u8 pad_top;
  u8 pad_left;
  u8 pad_bottom;
  u8 pad_right;
  u8 ins_h;
  u8 ins_w;
  u8 ins_last_h;
  u8 ins_last_w;
  u8 dilation_h;
  u8 dilation_w;
  bool relu_enable;
  int rshift_width;

  input_n = in_dim[0];
  ic = in_dim[1];
  ih = in_dim[2];
  iw = in_dim[3];

  oc = out_dim[1];
  oh = out_dim[2];
  ow = out_dim[3];

  kh = kern_dim[2];
  kw = kern_dim[3];

  stride_h = inst->getStrideHW()[0];
  stride_w = inst->getStrideHW()[1];

  pad_top = inst->getPadTLBR()[0];
  pad_left = inst->getPadTLBR()[1];
  pad_bottom = inst->getPadTLBR()[2];
  pad_right = inst->getPadTLBR()[3];

  ins_h = 0;
  ins_w = 0;
  ins_last_h = 0;
  ins_last_w = 0;

  dilation_h = inst->getDilationHW()[0];
  dilation_w = inst->getDilationHW()[1];

  relu_enable = inst->getEnableRelu();
  rshift_width = inst->getRShiftWidth();

  // below bmkernel code begins

  tensor_lmem *output;
  tensor_lmem *input;
  tensor_lmem *weight;
  tensor_lmem *bias;

  output = bmk1880_tl_prealloc_align(bmk_ctx_, addr_ofmap,
                                     shape_t4(input_n, oc, oh, ow), FMT_I8);

  input = bmk1880_tl_prealloc_align(bmk_ctx_, addr_ifmap,
                                    shape_t4(input_n, ic, ih, iw), FMT_I8);

  weight = bmk1880_tl_prealloc(bmk_ctx_, addr_filter, shape_t4(ic, oc, kh, kw),
                               FMT_I8);

  bias =
      bmk1880_tl_prealloc(bmk_ctx_, addr_bias, shape_t4(2, oc, 1, 1), FMT_I8);

  bmk1880_conv_param_t param;
  param.ofmap = output;
  param.ifmap = input;
  param.weight = weight;
  param.bias = bias;

  param.ins_h = ins_h;
  param.ins_last_h = ins_last_h;
  param.ins_w = ins_w;
  param.ins_last_w = ins_last_w;

  param.pad_top = pad_top;
  param.pad_bottom = pad_bottom;
  param.pad_left = pad_left;
  param.pad_right = pad_right;
  param.stride_h = stride_h;
  param.stride_w = stride_w;
  param.dilation_h = dilation_h;
  param.dilation_w = dilation_w;
  param.relu_enable = relu_enable;
  param.rshift_width = rshift_width;
  bmk1880_tpu_conv(bmk_ctx_, &param);

  bmk1880_tl_free(bmk_ctx_, bias);
  bmk1880_tl_free(bmk_ctx_, weight);
  bmk1880_tl_free(bmk_ctx_, input);
  bmk1880_tl_free(bmk_ctx_, output);
  DEBUG_GLOW(
      llvm::dbgs() << llvm::format(
          "%d\tconvolution: ouput=0x%x, input=0x%x, filiter=0x%x, bias=0x%x, "
          "out_shape=(%d,%d,%d,%d), input_shape=(%d,%d,%d,%d), "
          "kernel_shape=(%d,%d,%d,%d), padTLBR=(%d,%d,%d,%d), "
          "strideHW=(%d,%d), dilationHW(%d,%d), enable_relu=%d, rshift=%d\n",
          index++, addr_ofmap, addr_ifmap, addr_filter, addr_bias, input_n, oc,
          oh, ow, input_n, ic, ih, iw, ic, oc, kh, kw, pad_top, pad_left,
          pad_bottom, pad_right, stride_h, stride_w, dilation_h, dilation_w,
          relu_enable, rshift_width));
}

void BM1880CodeGenBMK::visit(const SophonMIDepthwiseConvolutionQ8Inst *inst) {
  auto value_input = inst->getSrc();
  auto value_output = inst->getDest();
  auto value_bias = inst->getBias();
  auto value_filter = inst->getFilter();

  auto addr_ifmap = emitValueAddress(value_input);
  auto addr_ofmap = emitValueAddress(value_output);
  auto addr_bias = emitValueAddress(value_bias);
  auto addr_filter = emitValueAddress(value_filter);

  auto in_dim = value_input->getType()->dims();
  auto out_dim = value_output->getType()->dims();
  auto kern_dim = value_filter->getType()->dims();

  int input_n;
  int ic;
  int ih;
  int iw;
  int oc;
  int oh;
  int ow;
  int kh;
  int kw;
  u8 stride_h;
  u8 stride_w;
  u8 pad_top;
  u8 pad_left;
  u8 pad_bottom;
  u8 pad_right;
  u8 ins_h;
  u8 ins_w;
  u8 ins_last_h;
  u8 ins_last_w;
  int rshift_width;

  input_n = in_dim[0];
  ic = in_dim[1];
  ih = in_dim[2];
  iw = in_dim[3];

  oc = out_dim[1];
  oh = out_dim[2];
  ow = out_dim[3];

  kh = kern_dim[2];
  kw = kern_dim[3];

  stride_h = inst->getStrideHW()[0];
  stride_w = inst->getStrideHW()[1];

  pad_top = inst->getPadTLBR()[0];
  pad_left = inst->getPadTLBR()[1];
  pad_bottom = inst->getPadTLBR()[2];
  pad_right = inst->getPadTLBR()[3];

  ins_h = 0;
  ins_w = 0;
  ins_last_h = 0;
  ins_last_w = 0;

  rshift_width = inst->getRShiftWidth();

  // below bmkernel code begins
  tensor_lmem *output;
  tensor_lmem *input;
  tensor_lmem *weight;
  tensor_lmem *bias = nullptr;

  output = bmk1880_tl_prealloc_align(bmk_ctx_, addr_ofmap,
                                     shape_t4(input_n, oc, oh, ow), FMT_I8);

  input = bmk1880_tl_prealloc_align(bmk_ctx_, addr_ifmap,
                                    shape_t4(input_n, ic, ih, iw), FMT_I8);

  // Depthwise
  weight = bmk1880_tl_prealloc_align(bmk_ctx_, addr_filter,
                                     shape_t4(1, oc, kh, kw), FMT_I8);

  bias =
      bmk1880_tl_prealloc(bmk_ctx_, addr_bias, shape_t4(2, oc, 1, 1), FMT_I8);

  bmk1880_depthwise_param_t param;
  param.ofmap = output;
  param.ifmap = input;
  param.weight = weight;
  param.bias = bias;

  param.ins_h = ins_h;
  param.ins_last_h = ins_last_h;
  param.ins_w = ins_w;
  param.ins_last_w = ins_last_w;

  param.pad_top = pad_top;
  param.pad_bottom = pad_bottom;
  param.pad_left = pad_left;
  param.pad_right = pad_right;

  param.stride_h = stride_h;
  param.stride_w = stride_w;

  param.rshift_width = rshift_width;
  bmk1880_tpu_depthwise(bmk_ctx_, &param);

  bmk1880_tl_free(bmk_ctx_, bias);
  bmk1880_tl_free(bmk_ctx_, weight);
  bmk1880_tl_free(bmk_ctx_, input);
  bmk1880_tl_free(bmk_ctx_, output);
  DEBUG_GLOW(llvm::dbgs() << llvm::format(
                 "%d\tdepthwise: ouput=%d, input=%d, filiter=%d, bias=%d\n",
                 index++, addr_ofmap, addr_ifmap, addr_filter, addr_bias));
  DEBUG_GLOW(
      llvm::dbgs() << llvm::format(
          "%d\tdepthwise: ouput=0x%x, input=0x%x, filiter=0x%x, bias=0x%x, "
          "out_shape=(%d,%d,%d,%d), input_shape=(%d,%d,%d,%d), "
          "kernel_shape=(%d,%d,%d,%d), padTLBR=(%d,%d,%d,%d), "
          "strideHW=(%d,%d), ins_h=(%d,%d), ins_w=(%d,%d), rshift=%d\n",
          index++, addr_ofmap, addr_ifmap, addr_filter, addr_bias, input_n, oc,
          oh, ow, input_n, ic, ih, iw, 1, oc, kh, kw, pad_top, pad_left,
          pad_bottom, pad_right, stride_h, stride_w, ins_h, ins_last_h, ins_w,
          ins_last_w, rshift_width));
}

template <class T>
void BM1880CodeGenBMK::bmk_matrix_mac(const T *inst, bool res_is_int8) {
  auto value_input = inst->getSrc();
  auto value_output = inst->getDest();
  auto value_bias = inst->getBias();
  auto value_filter = inst->getFilter();

  auto addr_ifmap = emitValueAddress(value_input);
  auto addr_ofmap = emitValueAddress(value_output);
  auto addr_bias = emitValueAddress(value_bias);
  auto addr_filter = emitValueAddress(value_filter);

  int right_shift_width = inst->getRShiftWidth();
  int left_shift_width = inst->getLShiftWidth();
  bool res_add = inst->getResultAdd();

  int M;
  int K;
  int N;

  auto in_dim = value_input->getType()->dims();
  auto kern_dim = value_filter->getType()->dims();

  M = in_dim[0];
  K = in_dim[1];
  N = kern_dim[1];

  // assert in, kern, and out dim

  // bmkernel code
  tensor_lmem *output;
  tensor_lmem *input;
  tensor_lmem *weight;
  tensor_lmem *bias;

  if (res_is_int8) {
    output =
        bmk1880_tl_prealloc_align(bmk_ctx_, addr_ofmap, shape_t2(M, N), FMT_I8);
  } else {
    output = bmk1880_tl_prealloc_align(
        bmk_ctx_, addr_ofmap, shape_t2(2 * M, N), // reserve for 16-bit output
        FMT_I8);
  }

  input =
      bmk1880_tl_prealloc_align(bmk_ctx_, addr_ifmap, shape_t2(M, K), FMT_I8);

  weight =
      bmk1880_tl_prealloc_align(bmk_ctx_, addr_filter, shape_t2(K, N), FMT_I8);

  bias = bmk1880_tl_prealloc_align(bmk_ctx_, addr_bias, shape_t2(2, N), FMT_I8);

  bmk1880_matrix_mac_param_t p;
  p.res = output; // 16-bit output space
  p.left = input;
  p.right = weight;
  p.bias = bias;

  p.lshift_width = left_shift_width;
  p.rshift_width = right_shift_width;

  p.res_is_int8 = res_is_int8;
  p.ctrls = CTRL_NULL;
  if (res_add) {
    p.ctrls = CTRL_RA;
  }

  bmk1880_tpu_matrix_mac(bmk_ctx_, &p);
  bmk1880_tl_free(bmk_ctx_, bias);
  bmk1880_tl_free(bmk_ctx_, weight);
  bmk1880_tl_free(bmk_ctx_, input);
  bmk1880_tl_free(bmk_ctx_, output);

  DEBUG_GLOW(llvm::dbgs() << llvm::format(
                 "%d\tmatrix_mac: ouput=0x%x, input=0x%x, filiter=0x%x, "
                 "bias=0x%x, output_shape=(%d,%d), input_shape=(%d,%d), "
                 "weight_shape=(%d,%d), res_is_int8=%d, res_add=%d, lshift=%d, "
                 "rshift=%d\n",
                 index++, addr_ofmap, addr_ifmap, addr_filter, addr_bias, M, N,
                 M, K, K, N, res_is_int8, res_add, left_shift_width,
                 right_shift_width));
}

void BM1880CodeGenBMK::visit(const SophonMIFCQ16Inst *inst) {
  bool res_is_int8 = false;
  bmk_matrix_mac(inst, res_is_int8);
}

void BM1880CodeGenBMK::visit(const SophonMIFCQ8Inst *inst) {
  bool res_is_int8 = true;
  bmk_matrix_mac(inst, res_is_int8);
}

void BM1880CodeGenBMK::performCodeGen() {
  DEBUG_GLOW(F_->dump());
  auto &instrs = F_->getInstrs();
  bmk_init();
  int index = 0;
  for (auto &I : instrs) {
    accept_helper(&I);
  }
  bmk_deinit();
}

std::vector<uint8_t> BM1880CodeGenBMK::getCmdbuf() { return cmdbuf_; }

} // namespace glow
