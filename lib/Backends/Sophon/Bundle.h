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
