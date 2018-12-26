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

#include "SophonBackend.h"
#include "BM188x/BM1880Backend.h"
#include "Bundle.h"
#include "CommandLine.h"
#include "SophonFunction.h"

using namespace glow;

llvm::cl::opt<Sophon::Target>
    target("target", llvm::cl::desc("Specify Sophon target"),
           llvm::cl::values(clEnumValN(Sophon::Target::BM1682, "bm1682",
                                       "Support float precision"),
                            clEnumValN(Sophon::Target::BM1880, "bm1880",
                                       "Support int8 symmetric precision"),
                            clEnumValN(Sophon::Target::BM1882, "bm1882",
                                       "Support int8 symmetric precision")),
           llvm::cl::init(Sophon::Target::BM1682),
           llvm::cl::Optional // TODO: change to Required in the future.
    );

llvm::cl::opt<std::string> loadCtable("load_ctable",
                                      llvm::cl::desc("Load calibration file"),
                                      llvm::cl::value_desc("ctable.pb2"),
                                      llvm::cl::Optional);

namespace glow {

Backend *SophonBackend::createBackend() {
  if (target == Sophon::Target::BM1880) {
    return new BM1880Backend();
  }
  llvm_unreachable("unsupport target!");
}

} // namespace glow
