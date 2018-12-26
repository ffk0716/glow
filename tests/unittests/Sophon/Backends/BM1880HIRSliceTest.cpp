#include "glow/Base/Type.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"

#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "gtest/gtest.h"

using namespace glow;

template <typename T> class SD;
inline static int idiv_round(int pNumerator, int pDenominator) {
  return (pNumerator + pDenominator - 1) / pDenominator;
}

void sliceWeight(SophonConvolutionQ8Node *node, unsigned oh_max_slice) {}

void sliceData(SophonConvolutionQ8Node *node, unsigned oh_max_slice) {
  Function *F = node->getParent();
  auto dim_i = node->getInput().getType()->dims();
  auto dim_f = node->getFilter().getType()->dims();
  auto dim_r = node->getResult().getType()->dims();
  ElemKind elemtype = node->getResult().getType()->getElementType();

  auto stride = node->getStrideHW();
  auto padTLBR = node->getPadTLBR();
  auto dilation = node->getDilationHW();

  // create new Node
  unsigned batch_sz = dim_i[0];
  unsigned ic = dim_i[1];
  unsigned ih = dim_i[2];
  unsigned iw = dim_i[3];

  unsigned oc = dim_r[1];
  unsigned oh = dim_r[2];
  unsigned ow = dim_r[3];

  unsigned kh = dim_f[2];
  unsigned stride_h = stride[0];
  std::vector<unsigned_t> pad_tlbr{1, padTLBR[1], 1, padTLBR[3]};
  std::vector<NodeValue> slice_result;
  unsigned ohsec = idiv_round(oh, oh_max_slice);

  for (int sec = 0; sec < ohsec; sec++) {
    // Get current padding

    if (sec == 0) {
      // First H slice
      pad_tlbr[0] = padTLBR[0];
      pad_tlbr[2] = 0;
    } else if (sec == (ohsec - 1)) {
      // Last H slice
      pad_tlbr[0] = 0;
      pad_tlbr[2] = padTLBR[2];
    } else {
      // Middle H slice
      pad_tlbr[0] = 0;
      pad_tlbr[2] = 0;
    }

    unsigned oh_idx = sec * oh_max_slice;
    unsigned oh_slice = oh_max_slice;

    // Last slice may be smaller
    if (sec == ohsec - 1) {
      oh_slice = oh - oh_idx;
    }

    unsigned cur_slice_pad = pad_tlbr[0] + pad_tlbr[2];
    unsigned pad_begin = padTLBR[0];

    unsigned ih_slice = (oh_slice - 1) * stride_h + kh - cur_slice_pad;

    // Make sure idx >= 0
    unsigned ih_idx =
        std::max(0, static_cast<int>(oh_idx * stride_h - pad_begin));

    // Create input slice
    std::array<size_t, 4> islice_shape{batch_sz, ic, ih_slice, iw};
    std::array<size_t, 4> oslice_shape{batch_sz, oc, oh_slice, ow};

    auto type_islice = F->getParent()->uniqueType(elemtype, islice_shape, 1, 0);

    auto input = node->getInput();
    auto input_name = input.getNode()->getName();
    auto *node_input = F->addNode(
        new SliceNode(input_name, type_islice, input, {0, 0, ih_idx, 0}));
    auto nodevalue_filter = node->getFilter();
    auto nodevalue_bias = node->getBias();
    auto *type_output =
        F->getParent()->uniqueType(elemtype, oslice_shape, 1, 0);
    auto *node_conv = F->addNode(new SophonConvolutionQ8Node(
        node->getName(), type_output, node_input, nodevalue_filter,
        nodevalue_bias, stride, pad_tlbr, dilation, 0, false));

    slice_result.push_back(node_conv->getResult());
  }

  // Concat all slices at H axis
  auto result_name = node->getResult().getNode()->getName();
  auto *node_concat = F->createConcat(result_name, slice_result, 2);
  NodeValue(node, 0).replaceAllUsesOfWith(node_concat);

  // Remove original node
  F->eraseNode(node);
}

TEST(BM1880HIRSliceTest, ConvQ8SliceData) {
  Module mod;
  Function *F = mod.createFunction("TestHIRSlice");

  ElemKind dataty = ElemKind::Int8QTy;
  SophonConvolutionQ8Node *node_conv;
  {
    std::vector<unsigned_t> stride_hw{2, 2};
    std::vector<unsigned_t> pad_tlbr{3, 3, 3, 3};
    std::vector<unsigned_t> dilation_hw{1, 1};

    std::array<size_t, 4> shape_input{1, 3, 224, 224};
    std::array<size_t, 4> shape_output{1, 64, 112, 112};
    std::array<size_t, 4> shape_kernel{3, 64, 7, 7};
    std::array<size_t, 1> shape_bias{64};

    auto *node_input =
        mod.createPlaceholder(dataty, shape_input, 1, 0, "conv.in", false);
    auto *node_filter =
        mod.createConstant(dataty, shape_kernel, 1, 0, "conv.f");
    auto *node_bias = mod.createConstant(dataty, shape_bias, 1, 0, "conv.b");
    auto *type_output = mod.uniqueType(dataty, shape_output, 1, 0);

    unsigned rshift_width = 0;
    bool enable_relu = false;
    node_conv = F->addNode(new SophonConvolutionQ8Node(
        "conv.1", type_output, node_input, node_filter, node_bias, stride_hw,
        pad_tlbr, dilation_hw, rshift_width, enable_relu));
  }
  {
    std::vector<unsigned_t> kernel_hw{3, 3};
    std::vector<unsigned_t> stride_hw{2, 2};
    std::vector<unsigned_t> pad_tlbr{0, 0, 0, 0};
    std::array<size_t, 4> shape_output{1, 64, 56, 56};
    auto *type_output = mod.uniqueType(dataty, shape_output, 1, 0);
    unsigned rshift = 0;
    unsigned mul = 1;
    bool round_mode = true;
    F->addNode(new SophonMaxPoolQ8Node("maxpool.1", type_output, node_conv,
                                       kernel_hw, stride_hw, pad_tlbr, rshift,
                                       mul, true));
  }

  F->dumpDAG("conv_before.dot");
  sliceData(node_conv, 38);
  F->dumpDAG("conv_after.dot");

  auto IR = llvm::make_unique<IRFunction>(F);
  IR->generateIR();
  IR->dump();
}
