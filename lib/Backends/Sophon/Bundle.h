/*
 * bmnet/lib/Backends/Sophon/Bundle.h
 *
 * Copyright Bitmain Technologies Inc.
 * Written by:
 *   Wanwei CAI <wanwei.cai@bitmain.com>
 * Created Time: 2018-10-13 17:22
 */

#ifndef BMNET_BACKENDS_SOPHON_BUNDLESAVER_H
#define BMNET_BACKENDS_SOPHON_BUNDLESAVER_H

#include "AllocationsInfo.h"
#include "SophonBackend.h"
#include "glow/IR/IR.h"
#include <libbmodel/bmodel.hpp>

namespace glow {

class Bundle final {
public:
  explicit Bundle(const SophonBackend *backend,
                  AllocationsInfo &allocationsInfo);

  /// Save code bundle built for \p target to \p outputDir.
  /// Make \p networkName the function name for
  /// the entry point of the network and prepend all generated
  /// files with this name.
  std::unique_ptr<bmodel::Model> codegen(IRFunction *F);

  static void saveBmodelFile(std::unique_ptr<bmodel::Model> model,
                             const std::string &outputDir);

private:
  using InputList = std::vector<Placeholder *>;
  using OutputList = std::vector<Placeholder *>;

  const SophonBackend *backend_;
  /// Information about allocations.
  AllocationsInfo &allocationsInfo_;

  SophonCmdBuf cmdbuf_;
  std::vector<uint8_t> u8_weights_;

private:
  /// Perform IR group optimization.
  void performIRGroup();
  /// Produce a bundle.

  std::unique_ptr<bmodel::Model> produceBmodel(IRFunction *F);

  void getInputs(IRFunction *F, InputList &inputs);
  void getOutputs(IRFunction *F, OutputList &outputs);
};
} // namespace glow

#endif
