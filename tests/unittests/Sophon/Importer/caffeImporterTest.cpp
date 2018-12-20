/**
 * Copyright (c) 2018-present, Bitmain, Inc.
 *
 *  The Author is PeiXu(pei.xu@bitmain.com)
 */
#include <fcntl.h>
//#include "ImporterTestUtils.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "Importer/CaffeModelLoader.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "gtest/gtest.h"

using namespace glow;
using llvm::dyn_cast;
using CaffeOperatorDef = bmnet::caffe::LayerParameter;
using google::protobuf::io::FileInputStream;
#define TEST_LAYER_NUM 1
#define RESILT_TYPE_IDX 0
static void loadTensor(const CaffeBlob &in, Tensor *T) {
  std::vector<size_t> dim;
  for (int i = 0; i < in.shape().dim_size(); i++) {
    dim.push_back(in.shape().dim(i));
  }

  T->reset(ElemKind::FloatTy, dim);

  if (in.data_size() > 0) {
    auto TH = T->getHandle<>();
    size_t i = 0;
    for (auto f : in.data()) {
      TH.raw(i++) = f;
    }
  }
}

void getNCHWData(Tensor *result, size_t n, size_t c, size_t h, size_t w) {
  result->reset(ElemKind::FloatTy, {n, c, h, w});
  auto RH = result->getHandle<>();
  for (size_t i = 0, e = n * c * h * w; i < e; i++) {
    RH.raw(i) = i;
  }
}
static inline int Axis2Index(int axis_index, int num_axes) {
  if (axis_index < 0) {
    return axis_index + num_axes;
  }
  return axis_index;
}

bool loadProtoFile(CaffeNetDef &net, const std::string &filename) {
  int fd = open(filename.c_str(), O_RDONLY);

  bool ret = false;
  if (-1 == fd) {
    GLOW_ASSERT(ret && "Can't find the model or network files.");
    return ret;
  }

  google::protobuf::io::FileInputStream *input = new google::protobuf::io::FileInputStream(fd);
  if (!google::protobuf::TextFormat::Parse(input, &net)) {
    GLOW_ASSERT(ret && "Prototxt parsing failed.");
    delete input;
    return false;
  }
  delete input;
  return true;
}
template <typename T1, typename T2>
bool isEqual(T1 src, T2 dst, int len) {
  for (int i = 0; i < len; i++) {
    if (src[i] != dst[i]) {
      // printf("src = %d, dst = %d\n", src[i], dst[i]);
      return false;
    }
  }
  return true;
}
template <typename T>
void transforRawData(glow::Handle<T> handle, T *dst, int len) {
  for (int i = 0; i < len; i++) {
    dst[i] = handle.raw(i);
  }
}

/// Test loading conv op from a Caffe2 model.
/// The input is N*C*H*W (1*1*3*3), the kernel is 2,
/// stride is 1, pad is 1, group is 1.
TEST(caffe, TestConvolutionLayer) {
  ExecutionEngine EE{BackendKind::Interpreter};

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string protoFile("../../models/caffe2Models/convolution.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

  // Obtain the data from model file
  const CaffeOperatorDef &op = net.layer(TEST_LAYER_NUM - 1);
  int filterLen = op.blobs(0).data_size();
  int biasLen = op.blobs(1).data_size();
  float *ex_filterData = new float[filterLen];
  float *ex_biasData = new float[biasLen];
  int i = 0;

  for (auto f : op.blobs(0).data()) {
    ex_filterData[i++] = f;
  }
  int j = 0;
  for (auto f : op.blobs(1).data()) {
    ex_biasData[j++] = f;
  }
  const auto &in_param = op.convolution_param();

  // get kernel sizes
  std::vector<unsigned_t> ex_k = {1, 1};
  std::vector<unsigned_t> ex_s = {1, 1};
  std::vector<unsigned_t> ex_p = {0, 0, 0, 0};
  std::vector<unsigned_t> ex_d = {1, 1};
  if (in_param.has_kernel_h() || in_param.has_kernel_w()) {
    assert(in_param.kernel_size_size() == 0);
    ex_k[0] = in_param.kernel_h();
    ex_k[1] = in_param.kernel_w();
  } else {
    int k_size = in_param.kernel_size_size();
    assert(k_size == 1 || k_size == 2);
    for (int i = 0; i < 2; i++) {
      ex_k[i] = in_param.kernel_size((k_size == 1) ? 0 : i);
    }
  }

  // get stride sizes
  if (in_param.has_stride_h() || in_param.has_stride_w()) {
    assert(2 == 2);
    assert(in_param.stride_size() == 0);
    ex_s[0] = in_param.stride_h();
    ex_s[1] = in_param.stride_w();
  } else {
    int s_size = in_param.stride_size();
    assert(s_size == 0 || s_size == 1 || s_size == 2);
    if (s_size != 0) {
      for (int i = 0; i < 2; i++) {
        ex_s[i] = in_param.stride((s_size == 1) ? 0 : i);
      }
    }
  }

  // get pad sizes
  if (in_param.has_pad_h() || in_param.has_pad_w()) {
    assert(2 == 2);
    assert(in_param.pad_size() == 0);
    ex_p[0] = in_param.pad_h();
    ex_p[2] = in_param.pad_h();
    ex_p[1] = in_param.pad_w();
    ex_p[3] = in_param.pad_w();
  } else {
    int p_size = in_param.pad_size();
    assert(p_size == 0 || p_size == 1 || p_size == 2);
    if (p_size != 0) {
      for (int i = 0; i < 4; i++) {
        ex_p[i] = in_param.pad((p_size == 1) ? 0 : i);
      }
    }
  }

  // get dilation size
  int d_size = in_param.dilation_size();
  assert(d_size == 0 || d_size == 1 || d_size == 2);
  if (d_size != 0) {
    for (int i = 0; i < 2; i++) {
      ex_d[i] = in_param.dilation((d_size == 1) ? 0 : i);
    }
  }
  unsigned_t ex_group = in_param.group();
  /**Comparing load data is whether expected or not.**/
  // Load data
  Tensor convData;
  std::string modelFile;
  getNCHWData(&convData, 1, 1, 3, 3);
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"data"}, {&convData.getType()}, *F);
  auto sophonConvNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonConvolutionNode *sophonConvNode = (SophonConvolutionNode *)sophonConvNodeValue.getNode();
  // Obtain the node information from Node
  const NodeValue filter = sophonConvNode->getFilter();
  const NodeValue bias = sophonConvNode->getBias();

  // Extract data from Tesor
  /**filter**/
  // Tensor* filterTensor = &((Variable*)(filter.getNode()))->getPayload();
  auto &filterTensor = dyn_cast<Constant>(filter.getNode())->getPayload();
  auto filter_TH = filterTensor.getHandle<>();
  float *re_filter = new float[filterLen];
  transforRawData(filter_TH, re_filter, filterLen);

  /**bias**/
  // Tensor* biasTensor = &((Variable*)(bias.getNode()))->getPayload();
  auto &biasTensor = dyn_cast<Constant>(bias.getNode())->getPayload();
  auto bias_TH = biasTensor.getHandle<>();
  float *re_bias = new float[biasLen];
  transforRawData(bias_TH, re_bias, biasLen);

  /**k, s, p, d, group**/
  llvm::ArrayRef<unsigned_t> re_kernels = sophonConvNode->getKernels();
  llvm::ArrayRef<unsigned_t> re_strides = sophonConvNode->getStrides();
  llvm::ArrayRef<unsigned_t> re_pads = sophonConvNode->getPads();
  llvm::ArrayRef<unsigned_t> re_dilations = sophonConvNode->getDilations();
  unsigned_t re_group = sophonConvNode->getGroup();

  // Compare result
  EXPECT_EQ(isEqual(re_filter, ex_filterData, filterLen), true);
  EXPECT_EQ(isEqual(re_bias, ex_biasData, biasLen), true);
  EXPECT_EQ(re_kernels.size(), 2);
  EXPECT_EQ(re_strides.size(), 2);
  EXPECT_EQ(re_pads.size(), 4);
  EXPECT_EQ(re_dilations.size(), 2);
  EXPECT_EQ(isEqual(re_kernels, ex_k, 2), true);
  EXPECT_EQ(isEqual(re_strides, ex_s, 2), true);
  EXPECT_EQ(isEqual(re_pads, ex_p, 4), true);
  EXPECT_EQ(isEqual(re_dilations, ex_d, 2), true);
  EXPECT_EQ(re_group, ex_group);

  // Release memory
  delete[] re_filter;
  delete[] re_bias;
  delete[] ex_filterData;
  delete[] ex_biasData;
}

TEST(caffe, TestPoolingLayer) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/pooling.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);

  // Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile;
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"conv1"}, {&poolData.getType()}, *F);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  // Obtain the data from model file
  auto &in_param = op.pooling_param();
  NodeValue in = caffeLoader.getNodeValueByName(op.bottom(0));
  auto input_shape = ShapeNCHW(in.dims());  // Note: in is NCHW

  // Obtain k,s,p
  std::vector<unsigned_t> ex_k = {1, 1};
  std::vector<unsigned_t> ex_s = {1, 1};
  std::vector<unsigned_t> ex_p = {0, 0, 0, 0};
  /*k*/
  if (in_param.global_pooling()) {
    ex_k[0] = input_shape.h;
    ex_k[1] = input_shape.w;
  } else {
    if (in_param.has_kernel_size()) {
      ex_k[0] = ex_k[1] = in_param.kernel_size();
    } else if (in_param.has_kernel_h() && in_param.has_kernel_w()) {
      ex_k[0] = in_param.kernel_h();
      ex_k[1] = in_param.kernel_w();
    }

    if (!in_param.has_stride_h()) {
      ex_s[0] = ex_s[1] = in_param.stride();
    } else {
      ex_s[0] = in_param.stride_h();
      ex_s[1] = in_param.stride_w();
    }

    if (!in_param.has_pad_h()) {
      ex_p[0] = ex_p[1] = ex_p[2] = ex_p[3] = in_param.pad();
    } else {
      ex_p[0] = in_param.pad_h();
      ex_p[2] = in_param.pad_h();
      ex_p[1] = in_param.pad_w();
      ex_p[3] = in_param.pad_w();
    }
  }
  /**Comparing load data is whether expected or not.**/

  llvm::ArrayRef<unsigned_t> re_kernels;
  llvm::ArrayRef<unsigned_t> re_strides;
  llvm::ArrayRef<unsigned_t> re_pads;
  // Get data from NodeValue
  auto nodeValue = caffeLoader.getNodeValueByName(op.top(0));
  if (in_param.pool() == bmnet::caffe::PoolingParameter_PoolMethod_MAX) {
    SophonMaxPoolNode *maxNode = (SophonMaxPoolNode *)(nodeValue.getNode());
    re_kernels = maxNode->getKernels();
    re_strides = maxNode->getStrides();
    re_pads = maxNode->getPads();
  } else if (in_param.pool() == bmnet::caffe::PoolingParameter_PoolMethod_AVE) {
    SophonAvgPoolNode *avgNode = (SophonAvgPoolNode *)(nodeValue.getNode());
    re_kernels = avgNode->getKernels();
    re_strides = avgNode->getStrides();
    re_pads = avgNode->getPads();
  } else {
    // Do nothing
    return;
  }

  // Compare result
  EXPECT_EQ(re_kernels.size(), 2);
  EXPECT_EQ(re_strides.size(), 2);
  EXPECT_EQ(re_pads.size(), 4);
  EXPECT_EQ(isEqual(re_kernels, ex_k, 2), true);
  EXPECT_EQ(isEqual(re_strides, ex_s, 2), true);
  EXPECT_EQ(isEqual(re_pads, ex_p, 4), true);
}
TEST(caffe, TestInnerProductLayer) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string protoFile("../../models/caffe2Models/InnerProduct.prototxt");

  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  printf("after expect1\n");
  // Obtain data from protofile
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);
  int weightLen = op.blobs(0).data_size();
  int biasLen = op.blobs(1).data_size();
  float *ex_weightData = new float[weightLen];
  float *ex_biasData = new float[biasLen];
  int i = 0;

  for (auto f : op.blobs(0).data()) {
    ex_weightData[i++] = f;
  }
  printf("next loop\n");
  int j = 0;
  for (auto f : op.blobs(1).data()) {
    ex_biasData[j++] = f;
  }

  /**Comparing load data is whether expected or not.**/
  // Load data
  Tensor innerProData;
  getNCHWData(&innerProData, 1, 1, 3, 3);
  std::string modelFile;
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"ip1"}, {&innerProData.getType()}, *F);

  // Obtain data from tensor
  auto innerNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  FullyConnectedNode *innerNode = (FullyConnectedNode *)innerNodeValue.getNode();

  // Obtain the node information from Node
  const NodeValue weight = innerNode->getWeights();
  const NodeValue bias = innerNode->getBias();

  // Extract data from Tesor
  /**filter**/
  auto &weightTensor = dyn_cast<Constant>(weight.getNode())->getPayload();
  Tensor transposeTensor;
  weightTensor.transpose(&transposeTensor, {1, 0});
  auto weight_TH = transposeTensor.getHandle<>();
  float *re_weight = new float[weightLen];
  transforRawData(weight_TH, re_weight, weightLen);
  /**bias**/
  auto &biasTensor = dyn_cast<Constant>(bias.getNode())->getPayload();
  auto bias_TH = biasTensor.getHandle<>();
  float *re_bias = new float[biasLen];
  transforRawData(bias_TH, re_bias, biasLen);
  // Compare result
  EXPECT_EQ(isEqual(re_weight, ex_weightData, weightLen), true);
  EXPECT_EQ(isEqual(re_bias, ex_biasData, biasLen), true);

  // Release memory
  delete[] re_weight;
  delete[] re_bias;
}
TEST(caffe, TestReluLayer) {
  std::string protoFile("../../models/caffe2Models/Relu.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestSoftmaxWithLossLayer) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestLRN) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/LRN.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  // Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile;
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"pool1"}, {&poolData.getType()}, *F);
  const CaffeOperatorDef &op = net.layer(0);

  // Construct parameter from file
  auto &in_param = op.lrn_param();
  unsigned int local_size = in_param.local_size();
  float ex_alpha = in_param.alpha();
  float ex_beta = in_param.beta();
  float ex_k = in_param.k();
  unsigned int norm_region = in_param.norm_region();

  // Obtain data from NodeValue
  auto sophonLRNNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonLocalResponseNormalizationNode *sophonLRNNode =
      (SophonLocalResponseNormalizationNode *)sophonLRNNodeValue.getNode();
  float re_alpha = sophonLRNNode->getAlpha();
  float re_beta = sophonLRNNode->getBeta();
  float re_k = sophonLRNNode->getK();

  // Compare the results
  EXPECT_EQ(ex_alpha, re_alpha);
  EXPECT_EQ(ex_beta, re_beta);
  EXPECT_EQ(ex_k, re_k);
}
TEST(caffe, TestHingeLoss) {
  std::string protoFile("../../models/caffe2Models/HingeLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestEuclideanLoss) {
  std::string protoFile("../../models/caffe2Models/EuclideanLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestSigmoidCrossEntropyLoss) {
  std::string protoFile("../../models/caffe2Models/SigmoidCrossEntropyLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestInfogainLoss) {
  std::string protoFile("../../models/caffe2Models/InfogainLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestAccuracy) {
  std::string protoFile("../../models/caffe2Models/Accuracy.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestSigmoid) {
  std::string protoFile("../../models/caffe2Models/Sigmoid.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestTanh) {
  std::string protoFile("../../models/caffe2Models/Tanh.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestAbsVal) {
  std::string protoFile("../../models/caffe2Models/AbsVal.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestPower) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/Power.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);

  // Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile;
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"in"}, {&poolData.getType()}, *F);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  // Construct parameter from file
  auto &in_param = op.power_param();
  float ex_power = in_param.power();
  float ex_scale = in_param.scale();
  float ex_shift = in_param.shift();

  // Obtain data from NodeValue
  auto sophonPowerNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonPowNode *sophonPowerNode = (SophonPowNode *)sophonPowerNodeValue.getNode();
  float re_power = sophonPowerNode->getPower();
  float re_scale = sophonPowerNode->getScale();
  float re_shift = sophonPowerNode->getShift();

  // Compare the results
  EXPECT_EQ(ex_power, re_power);
  EXPECT_EQ(ex_scale, re_scale);
  EXPECT_EQ(ex_shift, re_shift);
}
TEST(caffe, TestBNLL) {
  std::string protoFile("../../models/caffe2Models/BNLL.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestSplit) {
  std::string protoFile("../../models/caffe2Models/Split.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestFlatten) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/Flatten.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);

  // Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile;
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"conv9_2_mbox_conf_perm"},
                               {&poolData.getType()}, *F);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  // Construct parameter from file
  auto in = caffeLoader.getNodeValueByName(op.bottom(0));
  auto &in_param = op.flatten_param();

  std::vector<size_t> ex_outDims;
  const int num_axes = in.dims().size();
  const int start_axis = Axis2Index(in_param.axis(), num_axes);
  const int end_axis = Axis2Index(in_param.end_axis(), num_axes);
  for (int i = 0; i < start_axis; ++i) {
    ex_outDims.push_back(in.dims()[i]);
  }

  int flattened_dim = 1;
  for (int i = start_axis; i < end_axis + 1; i++) {
    flattened_dim *= in.dims()[i];
  }
  ex_outDims.push_back(flattened_dim);

  for (int i = end_axis + 1; i < num_axes; ++i) {
    ex_outDims.push_back(in.dims()[i]);
  }

  // Construct data from node value
  auto sophonFlattenNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonFlattenNode *sophonFlattenNode = (SophonFlattenNode *)sophonFlattenNodeValue.getNode();
  EXPECT_EQ(
      isEqual(ex_outDims, sophonFlattenNode->getType(RESILT_TYPE_IDX)->sizes_, ex_outDims.size()),
      true);
}
//
TEST(caffe, TestReshape) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/Reshape.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);

  // Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile;
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"input"}, {&poolData.getType()}, *F);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  // Construct parameter from file
  auto in = caffeLoader.getNodeValueByName(op.bottom(0));
  auto &in_param = op.reshape_param();
  int constant_count_ = 1;
  std::vector<int> copy_axes_;
  const bmnet::caffe::BlobShape &blob_shape = in_param.shape();
  const int num_new_axes = blob_shape.dim_size();
  int inferred_axis_ = -1;

  for (int i = 0; i < num_new_axes; ++i) {
    const int top_dim = blob_shape.dim(i);
    if (top_dim == 0) {
      copy_axes_.push_back(i);
    } else if (top_dim == -1) {
      inferred_axis_ = i;
    } else {
      constant_count_ *= top_dim;
    }
  }

  const int input_start_axis = in_param.axis();
  const int start_axis = (input_start_axis >= 0)
                             ? input_start_axis
                             : in.dims().size() + input_start_axis + 1;  // by ycs: why plus 1?
  //(input_start_axis >= 0) ? input_start_axis : in.dims().size() + input_start_axis;
  // by ycs: not plus 1
  const int num_axes = in_param.num_axes();
  const int end_axis = (num_axes == -1) ? in.dims().size() : (start_axis + num_axes);
  const int num_axes_replaced = end_axis - start_axis;
  const int num_axes_retained = in.dims().size() - num_axes_replaced;
  std::vector<size_t> ex_top_shape(num_axes_retained + num_new_axes);
  int top_shape_index = 0;
  for (int i = 0; i < start_axis; ++i) {
    ex_top_shape[top_shape_index++] = in.dims()[i];
  }
  for (int i = 0; i < num_new_axes; ++i) {
    ex_top_shape[top_shape_index++] = blob_shape.dim(i);
  }
  for (int i = end_axis; i < in.dims().size(); ++i) {
    ex_top_shape[top_shape_index++] = in.dims()[i];
  }

  for (int i = 0; i < copy_axes_.size(); ++i) {
    const int copy_axis_index = copy_axes_[i];
    ex_top_shape[start_axis + copy_axis_index] = in.dims()[start_axis + copy_axis_index];
  }
  if (inferred_axis_ >= 0) {
    // A -1 dim was specified; infer the correct dimension by computing the
    // product of the other dimensions.
    int explicit_count = constant_count_;
    for (int i = 0; i < start_axis; ++i) {
      explicit_count *= in.dims()[i];
    }
    for (int i = end_axis; i < in.dims().size(); ++i) {
      explicit_count *= in.dims()[i];
    }
    for (int i = 0; i < copy_axes_.size(); ++i) {
      const int copy_axis_index = copy_axes_[i];
      explicit_count *= ex_top_shape[start_axis + copy_axis_index];
    }
    int count = 1;
    for (int i = 0; i < in.dims().size(); ++i) {
      count *= in.dims()[i];
    }
    const int inferred_dim = count / explicit_count;
    ex_top_shape[start_axis + inferred_axis_] = inferred_dim;
  }

  std::vector<size_t> ex_outDims;
  for (unsigned_t i = 0; i < ex_top_shape.size(); i++) {
    ex_outDims.push_back(ex_top_shape[i]);
  }
  // Obtain data from NodeValue
  auto sophonReshapeNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonReshapeNode *sophonReshapeNode = (SophonReshapeNode *)sophonReshapeNodeValue.getNode();
  EXPECT_EQ(isEqual(sophonReshapeNode->getDims(), ex_top_shape, num_axes_retained + num_new_axes),
            true);
  EXPECT_EQ(
      isEqual(ex_outDims, sophonReshapeNode->getType(RESILT_TYPE_IDX)->sizes_, ex_outDims.size()),
      true);
}

TEST(caffe, TestSlice) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestEltwise) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestArgMax) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestSoftmax) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/Softmax.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);

  // Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile;
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"cls_score"}, {&poolData.getType()}, *F);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  // Construct parameter from file
  auto in = caffeLoader.getNodeValueByName(op.bottom(0));
  auto &in_param = op.softmax_param();
  int _axis = in_param.axis();
  unsigned_t axis = (_axis < 0) ? (_axis + in.dims().size()) : _axis;

  // Obtain data from NodeValue
  auto sophonSoftMaxNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonSoftMaxNode *sophonSoftMaxNode = (SophonSoftMaxNode *)sophonSoftMaxNodeValue.getNode();
  EXPECT_EQ(
      isEqual(in.dims(), sophonSoftMaxNode->getType(RESILT_TYPE_IDX)->sizes_, in.dims().size()),
      true);
}
TEST(caffe, TestMVN) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
TEST(caffe, TestReorg) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/Reorg.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);

  // Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile;
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"scalexx"}, {&poolData.getType()}, *F);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  // Construct parameter from file
  auto in = caffeLoader.getNodeValueByName(op.bottom(0));
  auto &in_param = op.reorg_param();
  auto input_shape = ShapeNCHW(in.dims());  // Note: in is NCHW

  // set reorg stride
  int stride = in_param.stride();
  // set output shape
  size_t output_c = input_shape.c * stride * stride;
  size_t output_h = input_shape.h / stride;
  size_t output_w = input_shape.w / stride;
  std::array<size_t, 4> outDims = {{input_shape.n, output_c, output_h, output_w}};

  auto sophonReorgNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonReorgNode *sophonReorgNode = (SophonReorgNode *)sophonReorgNodeValue.getNode();

  EXPECT_EQ(stride, (int)sophonReorgNode->getStride());
  EXPECT_EQ(isEqual(outDims, sophonReorgNode->getType(RESILT_TYPE_IDX)->sizes_, in.dims().size()),
            true);
}
TEST(caffe, TestPermute) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/Permute.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);

  // Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile;
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"conv9_2_mbox_conf"}, {&poolData.getType()},
                               *F);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  // Construct parameter from file
  auto &in_param = op.permute_param();
  NodeValue in = caffeLoader.getNodeValueByName(op.bottom(0));
  auto input_shape = ShapeNCHW(in.dims());  // Note: in is NCHW

  const int num_axes_ = in.dims().size();

  std::vector<unsigned_t> orders;
  // Push the specified new orders.
  for (int i = 0; i < in_param.order_size(); ++i) {
    int order = in_param.order(i);
    assert(order < num_axes_);
    // std::cout << "order should be less than the input dimension.";
    if (std::find(orders.begin(), orders.end(), order) != orders.end()) {
      std::cout << "there are duplicate orders";
    }
    orders.push_back(order);
  }
  // Push the rest orders. And save original step sizes for each axis.
  for (int i = 0; i < num_axes_; ++i) {
    if (std::find(orders.begin(), orders.end(), i) == orders.end()) {
      orders.push_back(i);
    }
  }
  // set output shape
  std::array<size_t, 4> outDims;
  for (int i = 0; i < 4; i++) {
    outDims[i] = in.dims()[orders[i]];
  }
  // Construct the real data from NodeValue
  auto sophonPermuteNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonPermuteNode *sophonPermuteNode = (SophonPermuteNode *)sophonPermuteNodeValue.getNode();
  llvm::ArrayRef<unsigned_t> re_orders = sophonPermuteNode->getOrder();
  EXPECT_EQ(isEqual(orders, re_orders, orders.size()), true);
  EXPECT_EQ(isEqual(outDims, sophonPermuteNode->getType(RESILT_TYPE_IDX)->sizes_, 4), true);
}
TEST(caffe, TestDropout) {
  std::string protoFile("../../models/caffe2Models/Dropout.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}

TEST(caffe, TestNormalize) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/Normalize.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);

  // Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile;
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"conv4_3_relu4_3_0_split_1"},
                               {&poolData.getType()}, *F);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  // Construct parameter from file
  auto &in_param = op.norm_param();
  NodeValue in = caffeLoader.getNodeValueByName(op.bottom(0));
  auto input_shape = ShapeNCHW(in.dims());  // Note: in is NCHW

  float eps = in_param.eps();
  float scale = 1.0;
  auto across_spatial = in_param.across_spatial();
  auto channel_shared = in_param.channel_shared();

  int channels = input_shape.c;
  if (op.blobs_size() >= 0) {
    auto tensor_shape = op.blobs(0).shape();
    if (channel_shared) {
      int count = 1;
      for (int i = 0; i < tensor_shape.dim_size(); i++) {
        count *= tensor_shape.dim(i);
      }
    } else {
      int count = 1;
      for (int i = 0; i < tensor_shape.dim_size(); i++) {
        count *= tensor_shape.dim(i);
      }
    }
  } else {
    if (channel_shared) {
      if (in_param.has_scale_filler()) {
        scale = in_param.scale_filler().value();
      } else {
        scale = 1.0;
      }
    } else {
      for (int i = 0; i < channels; i++) {
        if (in_param.has_scale_filler()) {
          scale = in_param.scale_filler().value();
        } else {
          scale = 1.0;
        }
      }
    }
  }
  // Construct the real data from NodeValue
  auto sophonNormalizeNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonNormalizeNode *sophonNormalizeNode =
      (SophonNormalizeNode *)sophonNormalizeNodeValue.getNode();
  bool re_acrossSpatial = sophonNormalizeNode->getAcrossSpatial();
  bool re_channelShared = sophonNormalizeNode->getChannelShared();
  float re_epsilon = sophonNormalizeNode->getEpsilon();
  float re_scale = sophonNormalizeNode->getScale();
  EXPECT_EQ(re_acrossSpatial, (bool)across_spatial);
  EXPECT_EQ(re_channelShared, (bool)channel_shared);
  EXPECT_EQ(re_epsilon, eps);
  EXPECT_EQ(re_scale, scale);
}
TEST(caffe, TestPReLU) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/PReLU.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);

  // Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile;
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"conv1"}, {&poolData.getType()}, *F);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  // Construct parameter from file
  auto &in_param = op.prelu_param();
  auto channel_shared = in_param.channel_shared();
  int slopeLen = op.blobs(0).data_size();
  float *slope = new float[slopeLen];
  int i = 0;
  for (auto f : op.blobs(0).data()) {
    slope[i++] = f;
  }
  auto sophonPreluNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonPreluNode *sophonPreluNode = (SophonPreluNode *)sophonPreluNodeValue.getNode();
  bool re_channel_shared = sophonPreluNode->getChannelShared();
  EXPECT_EQ((bool)channel_shared, re_channel_shared);

  const NodeValue slopeHandle = sophonPreluNode->getSlope();
  auto &slopeTensor = dyn_cast<Constant>(slopeHandle.getNode())->getPayload();
  auto slope_TH = slopeTensor.getHandle<>();
  float *re_slope = new float[slopeLen];
  transforRawData(slope_TH, re_slope, slopeLen);
  EXPECT_EQ(isEqual(slope, re_slope, slopeLen), true);

  delete[] slope;
  delete[] re_slope;
}
TEST(caffe, TestDeconvolution) {
  ExecutionEngine EE{BackendKind::Interpreter};

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string protoFile("../../models/caffe2Models/Deconvolution.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

  // Obtain the data from model file
  const CaffeOperatorDef &op = net.layer(TEST_LAYER_NUM - 1);
  int filterLen = op.blobs(0).data_size();
  int biasLen = op.blobs(1).data_size();
  float *ex_filterData = new float[filterLen];
  float *ex_biasData = new float[biasLen];
  int i = 0;
  for (auto f : op.blobs(0).data()) {
    ex_filterData[i++] = f;
  }
  int j = 0;
  for (auto f : op.blobs(1).data()) {
    ex_biasData[j++] = f;
  }
  const auto &in_param = op.convolution_param();

  // get kernel sizes
  std::vector<unsigned_t> ex_k = {1, 1};
  std::vector<unsigned_t> ex_s = {1, 1};
  std::vector<unsigned_t> ex_p = {0, 0, 0, 0};
  std::vector<unsigned_t> ex_d = {1, 1};
  if (in_param.has_kernel_h() || in_param.has_kernel_w()) {
    assert(in_param.kernel_size_size() == 0);
    ex_k[0] = in_param.kernel_h();
    ex_k[1] = in_param.kernel_w();
  } else {
    int k_size = in_param.kernel_size_size();
    assert(k_size == 1 || k_size == 2);
    for (int i = 0; i < 2; i++) {
      ex_k[i] = in_param.kernel_size((k_size == 1) ? 0 : i);
    }
  }

  // get stride sizes
  if (in_param.has_stride_h() || in_param.has_stride_w()) {
    assert(2 == 2);
    assert(in_param.stride_size() == 0);
    ex_s[0] = in_param.stride_h();
    ex_s[1] = in_param.stride_w();
  } else {
    int s_size = in_param.stride_size();
    assert(s_size == 0 || s_size == 1 || s_size == 2);
    if (s_size != 0) {
      for (int i = 0; i < 2; i++) {
        ex_s[i] = in_param.stride((s_size == 1) ? 0 : i);
      }
    }
  }

  // get pad sizes
  if (in_param.has_pad_h() || in_param.has_pad_w()) {
    assert(2 == 2);
    assert(in_param.pad_size() == 0);
    ex_p[0] = in_param.pad_h();
    ex_p[2] = in_param.pad_h();
    ex_p[1] = in_param.pad_w();
    ex_p[3] = in_param.pad_w();
  } else {
    int p_size = in_param.pad_size();
    assert(p_size == 0 || p_size == 1 || p_size == 2);
    if (p_size != 0) {
      for (int i = 0; i < 4; i++) {
        ex_p[i] = in_param.pad((p_size == 1) ? 0 : i);
      }
    }
  }

  // get dilation size
  int d_size = in_param.dilation_size();
  assert(d_size == 0 || d_size == 1 || d_size == 2);
  if (d_size != 0) {
    for (int i = 0; i < 2; i++) {
      ex_d[i] = in_param.dilation((d_size == 1) ? 0 : i);
    }
  }
  unsigned_t ex_group = in_param.group();
  /**Comparing load data is whether expected or not.**/
  // Load data
  Tensor convData;
  getNCHWData(&convData, 1, 1, 3, 3);
  std::string modelFile;
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"u1d"}, {&convData.getType()}, *F);

  auto sophonConvNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonConvolutionNode *sophonConvNode = (SophonConvolutionNode *)sophonConvNodeValue.getNode();

  // Obtain the node information from Node
  const NodeValue filter = sophonConvNode->getFilter();
  const NodeValue bias = sophonConvNode->getBias();

  // Extract data from Tesor
  /**filter**/
  auto &filterTensor = dyn_cast<Constant>(filter.getNode())->getPayload();
  auto filter_TH = filterTensor.getHandle<>();
  float *re_filter = new float[filterLen];
  transforRawData(filter_TH, re_filter, filterLen);

  /**bias**/
  auto &biasTensor = dyn_cast<Constant>(bias.getNode())->getPayload();
  auto bias_TH = biasTensor.getHandle<>();
  float *re_bias = new float[biasLen];
  transforRawData(bias_TH, re_bias, biasLen);

  /**k, s, p, d, group**/
  llvm::ArrayRef<unsigned_t> re_kernels = sophonConvNode->getKernels();
  llvm::ArrayRef<unsigned_t> re_strides = sophonConvNode->getStrides();
  llvm::ArrayRef<unsigned_t> re_pads = sophonConvNode->getPads();
  llvm::ArrayRef<unsigned_t> re_dilations = sophonConvNode->getDilations();
  unsigned_t re_group = sophonConvNode->getGroup();

  // Compare result
  EXPECT_EQ(isEqual(re_filter, ex_filterData, filterLen), true);
  EXPECT_EQ(isEqual(re_bias, ex_biasData, biasLen), true);
  EXPECT_EQ(re_kernels.size(), 2);
  EXPECT_EQ(re_strides.size(), 2);
  EXPECT_EQ(re_pads.size(), 2);
  EXPECT_EQ(re_dilations.size(), 2);
  EXPECT_EQ(isEqual(re_kernels, ex_k, 2), true);
  EXPECT_EQ(isEqual(re_strides, ex_s, 2), true);
  EXPECT_EQ(isEqual(re_pads, ex_p, 2), true);
  EXPECT_EQ(isEqual(re_dilations, ex_d, 2), true);
  EXPECT_EQ(re_group, ex_group);

  // Release memory
  delete[] re_filter;
  delete[] re_bias;
  delete[] ex_filterData;
  delete[] ex_biasData;
}

#if 0
TEST(caffe, TestScale) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/Scale.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  
  //Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile = "";
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"conv2_1"}, {&poolData.getType()}, *F);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  //Construct parameter from file
  auto in = caffeLoader.getNodeValueByName(op.bottom(0));
  auto &in_param = op.scale_param();
  int num_axes = in_param.num_axes();
  int axis = in_param.axis();
  axis = (axis < 0) ? (axis + in.dims().size()) : axis; //by ycs: why puls num-axes, not in.dims()
  // calculate scale/inner dim
  int scale_dim = 1;
  for (int i = axis; i < axis + num_axes; i++) {
    scale_dim *= in.dims()[i];
  }
  int inner_dim = 1;
  for (int i = axis + num_axes; i < in.dims().size(); i++) {
    scale_dim *= in.dims()[i];
  }
  //get scale and bias in an array
  int scaleLen = op.blobs(0).data_size();
  int biasLen = op.blobs(1).data_size();
  float* ex_scale = new float[scaleLen];
  float* ex_bias = new float[biasLen];
  int i = 0;
  for (auto f : op.blobs(0).data()) {
    ex_scale[i++] = f;
  }
  int j = 0;
  for (auto f : op.blobs(1).data()) {
    ex_bias[i++] = f;
  }
  
  //Get data from loader
  auto sophonScaleNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonScaleNode* sophonScaleNode = (SophonScaleNode*)sophonScaleNodeValue.getNode();
  /*Get scale from tensor*/
  const NodeValue scaleHandle = sophonScaleNode->getScale();
  auto &scaleTensor = dyn_cast<Constant>(scaleHandle.getNode())->getPayload();
  auto scale_TH = scaleTensor.getHandle<>();
  float* re_scale = new float[scaleLen];
  transforRawData(scale_TH, re_scale, scaleLen);
  /*Get bias from tensor*/
  const NodeValue biasHandle = sophonScaleNode->getBias();
  auto &biasTensor = dyn_cast<Constant>(biasHandle.getNode())->getPayload();
  auto bias_TH = biasTensor.getHandle<>();
  float* re_bias = new float[biasLen];
  transforRawData(scale_TH, re_bias, biasLen);
  /*Get axis*/
  float re_axis = sophonScaleNode->getAxis();
  /*Get numAxis*/
  float re_numAxis = sophonScaleNode->getNumAxes();
  
  //Compare data
  EXPECT_EQ(isEqual(ex_scale, re_scale, scaleLen), true);
  EXPECT_EQ(isEqual(ex_bias, re_bias, biasLen), true);
  EXPECT_EQ(axis, re_axis);
  EXPECT_EQ(num_axes, re_numAxis);

  //Release memory
  delete ex_scale;
  delete re_scale;
  delete ex_bias;
  delete re_bias;
}

TEST(caffe, TestUpsample) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/Upsample.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
}
TEST(caffe, TestBatchNorm) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/BatchNorm.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  
  //Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile = "";
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"conv1"}, {&poolData.getType()}, *F);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  //Construct parameter from file
  auto &in_param = op.batch_norm_param();
  float epsilon = in_param.eps();
  float scale = 1.0;
  if ((op.blobs_size() == 3) && (op.blobs(2).offset() != 0xffffffffffff)) {
    Tensor *sv = new Tensor();
    loadTensor(op.blobs(2), sv);
    assert(sv->getType().size() == 1);
    scale = sv->getHandle<>().raw(0);
  }
  int meanLen = op.blobs(0).data_size();
  int varianceLen = op.blobs(1).data_size();
  float* ex_mean = new float[meanLen];
  float* ex_variance = new float[varianceLen];
  //int i = 0;
  //for (auto f : op.blobs[0].data()) { 
  //  ex_mean[i++] = f;
  //}
  //int j = 0;
  //for (auto f : op.blobs[1].data()) { 
  //  ex_variance[j++] = f;
  //}  
  //Construct the real data from NodeValue
  auto sophonBatchNormalizationNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonBatchNormalizationNode* sophonBatchNormalizationNode = (SophonBatchNormalizationNode*)sophonBatchNormalizationNodeValue.getNode();
  float re_epsilon = sophonBatchNormalizationNode->getEpsilon();
  float re_scale = sophonBatchNormalizationNode->getScale();
  EXPECT_EQ(scale, re_scale);
  EXPECT_EQ(epsilon, re_epsilon);
  
  //transform tensor to array
  const NodeValue meanHandle = sophonBatchNormalizationNode->getMean();
  auto &meanTensor = dyn_cast<Constant>(meanHandle.getNode())->getPayload();
  auto mean_TH = meanTensor.getHandle<>();
  float* re_mean = new float[meanLen];
  transforRawData(mean_TH, re_mean, meanLen);
  
  const NodeValue varianceHandle = sophonBatchNormalizationNode->getVariance();
  auto &varianceTensor = dyn_cast<Constant>(varianceHandle.getNode())->getPayload();
  auto variance_TH = varianceTensor.getHandle<>();
  float* re_variance = new float[varianceLen];
  transforRawData(mean_TH, re_mean, varianceLen); 
  
  EXPECT_EQ(isEqual(ex_mean, re_mean, meanLen), true);
  EXPECT_EQ(isEqual(ex_variance, re_variance, varianceLen), true);
  
  //Release memory
  delete re_mean;
  delete re_variance;
}

TEST(caffe, TestDetectionEvaluate) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}

TEST(caffe, TestInput) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}

TEST(caffe, TestYolo) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}
TEST(caffe, TestXavier) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}
TEST(caffe, TestAnnotatedData) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}
TEST(caffe, TestDetection) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}
TEST(caffe, TestPriorBox) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}
TEST(caffe, TestMsra) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}
TEST(caffe, TestMultiBoxLoss) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}
TEST(caffe, TestTile) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}

TEST(caffe, TestDetectOutput) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}
TEST(caffe, TestDeconvolution) {
  ExecutionEngine EE{BackendKind::Interpreter};
  
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  std::string protoFile("../../models/caffe2Models/Deconvolution.prototxt");
  CaffeNetDef net;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  
  //Obtain the data from model file
  CaffeOperatorDef& op = net.layer(TEST_LAYER_NUM - 1);
  int filterLen = op.blobs(0).data_size();
  int biasLen = op.blobs(1).data_size();
  float * ex_filterData = new float[filterLen];
  float * ex_biasData = new float[biasLen];
  int i = 0;
  for (auto f : op.blobs(0).data()) {
      ex_filterData[i++] = f;
  }
  int j = 0;
  for (auto f : op.blobs(1).data()) {
      ex_biasData[j++] = f;
  }  
  const auto &in_param = op.convolution_param();
  
  // get kernel sizes
  std::vector<unsigned_t> ex_k = {1, 1};
  std::vector<unsigned_t> ex_s = {1, 1};
  std::vector<unsigned_t> ex_p = {0, 0, 0, 0};
  std::vector<unsigned_t> ex_d = {1, 1};  
  if (in_param.has_kernel_h() || in_param.has_kernel_w()) {
    assert(in_param.kernel_size_size() == 0);
    ex_k[0] = in_param.kernel_h();
    ex_k[1] = in_param.kernel_w();
  } else {
    int k_size = in_param.kernel_size_size();
    assert(k_size == 1 || k_size == 2);
    for (int i = 0; i < 2; i++) {
      ex_k[i] = in_param.kernel_size((k_size == 1) ? 0 : i);
    }
  } 

  // get stride sizes
  if (in_param.has_stride_h() || in_param.has_stride_w()) {
    assert(2 == 2);
    assert(in_param.stride_size() == 0);
    ex_s[0] = in_param.stride_h();
    ex_s[1] = in_param.stride_w();
  } else {
    int s_size = in_param.stride_size();
    assert(s_size == 0 || s_size == 1 || s_size == 2);
    if (s_size != 0) {
      for (int i = 0; i < 2; i++) {
        ex_s[i] = in_param.stride((s_size == 1) ? 0 : i);
      }
    }
  }

  // get pad sizes
  if (in_param.has_pad_h() || in_param.has_pad_w()) {
    assert(2 == 2);
    assert(in_param.pad_size() == 0);
    ex_p[0] = in_param.pad_h();
    ex_p[2] = in_param.pad_h();
    ex_p[1] = in_param.pad_w();
    ex_p[3] = in_param.pad_w();
  } else {
    int p_size = in_param.pad_size();
    assert(p_size == 0 || p_size == 1 || p_size == 2);
    if (p_size != 0) {
      for (int i = 0; i < 4; i++) {
        ex_p[i] = in_param.pad((p_size == 1) ? 0 : i);
      }
    }
  }

  // get dilation size
  int d_size = in_param.dilation_size();
  assert(d_size == 0 || d_size == 1 || d_size == 2);
  if (d_size != 0) {
    for (int i = 0; i < 2; i++) {
      ex_d[i] = in_param.dilation((d_size == 1) ? 0 : i);
    }
  }
  unsigned_t ex_group = in_param.group();
  /**Comparing load data is whether expected or not.**/
  //Load data
  Tensor convData;
  getNCHWData(&convData, 1, 1, 3, 3);
  std::string modelFile = "";
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"u1d"}, {&convData.getType()}, *F);

  auto sophonConvNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonConvolutionNode* sophonConvNode = (SophonConvolutionNode*)sophonConvNodeValue.getNode();

  //Obtain the node information from Node
  const NodeValue filter = sophonConvNode->getFilter();
  const NodeValue bias = sophonConvNode->getBias();

  //Extract data from Tesor
  /**filter**/
  auto &filterTensor = dyn_cast<Constant>(filter.getNode())->getPayload();
  auto filter_TH = filterTensor.getHandle<>();
  float* re_filter = new float[filterLen];
  transforRawData(filter_TH, re_filter, filterLen);
  
  /**bias**/
  auto &biasTensor = dyn_cast<Constant>(bias.getNode())->getPayload();
  auto bias_TH = biasTensor.getHandle<>();
  float* re_bias = new float[biasLen];
  transforRawData(bias_TH, re_bias, biasLen);
  
  /**k, s, p, d, group**/
  llvm::ArrayRef<unsigned_t> re_kernels = sophonConvNode->getKernels();
  llvm::ArrayRef<unsigned_t> re_strides = sophonConvNode->getStrides();
  llvm::ArrayRef<unsigned_t> re_pads = sophonConvNode->getPads();
  llvm::ArrayRef<unsigned_t> re_dilations = sophonConvNode->getDilations();
  unsigned_t re_group = sophonConvNode->getGroup();

  //Compare result
  EXPECT_EQ(isEqual(re_filter, ex_filterData, filterLen), true);
  EXPECT_EQ(isEqual(re_bias, ex_biasData, biasLen), true);
  EXPECT_EQ(re_kernels.size(),2);
  EXPECT_EQ(re_strides.size(),2);
  EXPECT_EQ(re_pads.size(),4);
  EXPECT_EQ(re_dilations.size(),2);
  EXPECT_EQ(isEqual(re_kernels, ex_k, 2), true);
  EXPECT_EQ(isEqual(re_strides, ex_s, 2), true);
  EXPECT_EQ(isEqual(re_pads, ex_p, 4), true);
  EXPECT_EQ(isEqual(re_dilations, ex_d, 2), true);
  EXPECT_EQ(re_group, ex_group);

  //Release memory
  delete re_filter;
  delete re_bias;
  delete ex_filterData;
  delete ex_biasData;
}
TEST(caffe, TestDummyData) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}
TEST(caffe, TestSGD) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}
TEST(caffe, TestMemoryData) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}
TEST(caffe, TestRegion) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}
TEST(caffe, TestGaussian) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}
TEST(caffe, TestProposal) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}
TEST(caffe, TestROIPooling) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/ROIPooling.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  
  //Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile = "";
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"conv5_3_relu5_3_0_split_1"}, {&poolData.getType()}, *F);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  //Construct parameter from file
  auto &in_param = op.roi_pooling_param();
  std::vector<unsigned_t> pooled_shape = {in_param.pooled_h(), in_param.pooled_w()};
  std::array<size_t, 4> outDims = 
      {in0.dims()[0], in1.dims()[1], in_param.pooled_h(), in_param.pooled_w()};
  float spatialScale = in_param.spatial_scale();

  
  auto sophonROIPoolNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonROIPoolNode* sophonROIPoolNode = (SophonROIPoolNode*)sophonROIPoolNodeValue.getNode();
  llvm::ArrayRef<unsigned_t> re_poolShape = sophonROIPoolNode->getPoolShape();
  float re_spatialScale = sophonROIPoolNode->getSpatialScale();

  EXPECT_EQ(isEqual(outDims, sophonROIPoolNode->getType(RESILT_TYPE_IDX)->sizes_, 4), true);
  EXPECT_EQ(isEqual(pooled_shape, re_poolShape, pooled_shape.size()), true);
  EXPECT_EQ(spatialScale, re_spatialScale);
}
TEST(caffe, TestConstant) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}
TEST(caffe, TestShuffleChannel) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/ShuffleChannel.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  
  //Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile = "";
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"resx2_conv1"}, {&poolData.getType()}, *F);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  //Construct parameter from file
  auto &in_param = op.shuffle_channel_param();
  auto in = caffeLoader.getNodeValueByName(op.bottom(0));

  auto sophonShuffleChannelNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonShuffleChannelNode* sophonShuffleChannelNode = (SophonShuffleChannelNode*)sophonShuffleChannelNodeValue.getNode();
  EXPECT_EQ(isEqual(in.dims(), sophonShuffleChannelNode->getType(RESILT_TYPE_IDX)->sizes_, in.dims().size()), true);   
}
TEST(caffe, TestLSTM) {
  std::string protoFile("../../models/caffe2Models/SoftmaxWithLoss.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
}
//How to verify the all_in NodeValues
TEST(caffe, TestConcat) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/Concat.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  //Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile = "";

  llvm::ArrayRef<const char *> tensorNames = {"inception_c3_1x1_2"};
  CaffeModelLoader caffeLoader(modelFile, protoFile, tensorNames, {&poolData.getType()}, *F);


  //Construct parameter from file
  auto in = caffeLoader.getNodeValueByName(op.bottom(0)); 
  auto &in_param = op.concat_param();

  int _axis = in_param.axis();
  unsigned_t ex_outDim = (_axis < 0)? (_axis + in.dims().size()) : _axis;

  int concat_axis_dim = in.dims()[ex_outDim];
  std::vector<NodeValue> all_in;
  all_in.push_back(in);
  for (unsigned_t i = 1; i < op.bottom_size(); ++i) {
    auto in_b = caffeLoader.getNodeValueByName(op.bottom(i));
    all_in.push_back(in_b);
    concat_axis_dim += in_b.dims()[ex_outDim];
  }

  std::vector<size_t> ex_outDims;
  for (unsigned int i = 0; i < in.dims().size(); i++) {
    ex_outDims.push_back(in.dims()[i]);
  }
  ex_outDims[ex_outDim] = concat_axis_dim;

  //Obtain data from NodeValue
  auto sophonConcatNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonConcatNode* sophonConcatNode = (SophonConcatNode*)sophonConcatNodeValue.getNode();
  unsigned_t re_outDim =  sophonConcatNode->getDim();
  EXPECT_EQ(ex_outDim, re_outDim);
  EXPECT_EQ(isEqual(ex_outDims, sophonConcatNode->getType(RESILT_TYPE_IDX)->sizes_, ex_outDims.size()), true);
}
TEST(caffe, TestCrop) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/Crop.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  
  //Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile = "";
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"d0c_relu_d0c_0_split_1"}, {&poolData.getType()}, *F);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  //Construct parameter from file
  auto &in_param = op.crop_param();  
  NodeValue in0 = caffeLoader.getNodeValueByName(op.bottom(0));
  NodeValue in1 = caffeLoader.getNodeValueByName(op.bottom(1));
  
  const int num_axes = in0.dims().size();
  int axis_index = in_param.axis();  
  int start_axis;

  if (axis_index < 0) {
    start_axis = axis_index + num_axes;
  }
  start_axis = axis_index;
  std::array<size_t, 4> outDims ; 
  std::vector<unsigned_t> offsets ;

  // Determine crop offsets and the new shape post-crop.
  for (int i = 0; i < num_axes; ++i) {
    int crop_offset = 0;
    int new_size = in0.dims()[i];
    if (i >= start_axis) {
      new_size = in1.dims()[i];
      if (in_param.offset_size() == 1) {
        crop_offset = in_param.offset(0);
      } else if (in_param.offset_size() > 1) {
        crop_offset = in_param.offset(i - start_axis);
      }
    }
    outDims[i] = new_size ;
    offsets.push_back(crop_offset);
  }
  
  //Construct the real data from NodeValue
  auto sophonCropNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonCropNode* sophonCropNode = (SophonCropNode*)sophonCropNodeValue.getNode();
  
  EXPECT_EQ(isEqual(outDims, sophonCropNode->getType(RESILT_TYPE_IDX)->sizes_, num_axes), true);
  llvm::ArrayRef<unsigned_t>re_offsets = sophonCropNode->getOffsets();
  EXPECT_EQ(isEqual(offsets, re_offsets, offsets.size()),true);  
}
TEST(caffe, TestPSROIPooling) {
  ExecutionEngine EE{BackendKind::Interpreter};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  CaffeOperatorDef layer;
  std::string protoFile("../../models/caffe2Models/PSROIPooling.prototxt");
  CaffeNetDef net;
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  
  //Load data
  Tensor poolData;
  getNCHWData(&poolData, 1, 1, 3, 3);
  std::string modelFile = "";
  CaffeModelLoader caffeLoader(modelFile, protoFile, {"rfcn_bbox"}, {&poolData.getType()}, *F);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);
  const CaffeOperatorDef &op = net.layer(0);

  //Construct parameter from file
  auto &in_param = op.psroi_pooling_param(); 
  NodeValue in0 = caffeLoader.getNodeValueByName(op.bottom(0));
  
  // Get output shape
  size_t output_n = in0.dims()[0];
  size_t output_c = in_param.output_dim();
  size_t output_h = in_param.group_size();
  size_t output_w = in_param.group_size();
  std::array<size_t, 4> outDims ={{output_n, output_c, output_h, output_w }};
  float spatial_scale = in_param.spatial_scale();
  int group = in_param.group_size();

  //Construct the real data from NodeValue
  auto sophonPSROIPoolingNodeValue = caffeLoader.getNodeValueByName(op.top(0));
  SophonPSROIPoolNode* sophonPSROIPoolingNode = (SophonPSROIPoolNode*)sophonPSROIPoolingNodeValue.getNode();
  
  float re_spatial_scale = sophonPSROIPoolingNode->getSpatialScale();
  float re_group = sophonPSROIPoolingNode->getGroup();
  EXPECT_EQ(isEqual(outDims, sophonPSROIPoolingNode->getType(RESILT_TYPE_IDX)->sizes_, 4), true);
  EXPECT_EQ(spatial_scale, re_spatial_scale);
  EXPECT_EQ(group, re_group);
}
TEST(caffe, TestDetectionOutput) {
  std::string protoFile("../../models/caffe2Models/DetectionOutput.prototxt");
  CaffeNetDef net;
  CaffeOperatorDef layer;  
  EXPECT_EQ(loadProtoFile(net, protoFile), true);
  EXPECT_EQ(net.layer_size(), TEST_LAYER_NUM);

}

#endif
