/*
 * bmnet/lib/Backends/Sophon/SophonFunction.cpp
 *
 * Copyright Bitmain Technologies Inc.
 * Written by:
 *   Wanwei CAI <wanwei.cai@bitmain.com>
 * Created Time: 2018-10-15 10:03
 */

#define DEBUG_TYPE "sophon_func"

#include "SophonFunction.h"
#include "CommandLine.h"
#include "glow/Base/Tensor.h"
#include "glow/Graph/Context.h"
#include "glow/Support/Debug.h"
#include "llvm/Support/Debug.h"

#include <fstream>
#include <libbmruntime/bmruntime.h>
#include <libbmruntime/bmruntime_bmnet.h>

namespace glow {

SophonFunction::SophonFunction(std::unique_ptr<bmodel::Model> model) {
  model_ = std::move(model);
}

SophonFunction::~SophonFunction() {}

void SophonFunction::execute(Context &ctx) {
  DEBUG_GLOW(bmodel::print(*model_));

  bmctx_t bmctx;
  bmerr_t ret;
  ret = bm_init(0, &bmctx);
  if (ret != BM_SUCCESS) {
    llvm_unreachable("bm_init failed");
  }
  bmnet_t net;
  bmnet_output_info_t output_info;

  auto bmodel_filename = []() {
    char temp[] = "/tmp/glow-ut-temp.XXXXXX";
    return std::string(mkdtemp(temp)) + "/test.bmodel";
  }();

  ret = bmodel::save(*model_, bmodel_filename);
  if (ret != BM_SUCCESS) {
    llvm_unreachable("save bmodel failed");
  }

  ret = bmnet_register_bmodel(bmctx, bmodel_filename.c_str(), &net);
#if 0 // bmnet_register_bmodel_data has bug
  std::string bmodel_json = bmodel_->json_dump();
  ret = bmnet_register_bmodel_data(
      bmctx, reinterpret_cast<uint8_t *>(const_cast<char *>(bmodel_json.data())),
      bmodel_json.size(), &net);
#endif
  if (ret != BM_SUCCESS) {
    llvm_unreachable("register failed");
  }
  ret = bmnet_get_output_info(net, &output_info);
  if (ret != BM_SUCCESS) {
    llvm_unreachable("get output failed!");
  }

  // TODO support multiple inputs
  std::vector<uint8_t> input;

  for (auto PH : ctx.pairs()) {
    // input if fail to find "save_" prefix
    if (PH.first->getName().find("save_") == llvm::StringRef::npos) {
      DEBUG_GLOW(llvm::dbgs() << "input name is: " << PH.first->getName());
      auto *tensor = PH.second;
      input.resize(tensor->size());
      memcpy(input.data(), PH.second->getUnsafePtr(), tensor->size());
    }
  }

  // upload input data
  ret = bmnet_load_input(net, input.data());
  if (ret != BM_SUCCESS) {
    llvm_unreachable("load input failed!");
  }

  // run cmdbuf
  ret = bmnet_run(net);
  if (ret != BM_SUCCESS) {
    llvm_unreachable("run failed!");
  }

  size_t output_size = output_info.output_size;
  std::vector<uint8_t> output(output_size);
  // download output data
  ret = bmnet_store_output(net, output.data());
  if (ret != BM_SUCCESS) {
    llvm_unreachable("store output failed!");
  }

  bmnet_cleanup(net);
  bm_exit(bmctx);

  // TODO support multiple outputs
  for (auto PH : ctx.pairs()) {
    // Sophon Backend uses "save_" prefix to recognize output
    if (PH.first->getName().find("save_") != llvm::StringRef::npos) {
      DEBUG_GLOW(llvm::dbgs() << "output name is: " << PH.first->getName());
      memcpy(PH.second->getUnsafePtr(), output.data(), output_size);
    }
  }
}

} // namespace glow
