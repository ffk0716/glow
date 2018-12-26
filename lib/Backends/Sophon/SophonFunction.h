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

#ifndef _SophonFUNCTION_H
#define _SophonFUNCTION_H

#include "glow/Backends/CompiledFunction.h"

#include <libbmodel/bmodel.hpp>
#include <memory>

namespace glow {

/// A Glow IR function compiled for Sophon.
class SophonFunction final : public CompiledFunction {

public:
  /// Ctor.
  explicit SophonFunction(std::unique_ptr<bmodel::Model> model);

  /// @name CompiledFunction interface
  ///@{
  ~SophonFunction() override;

  void execute(Context &ctx) override;

private:
  std::unique_ptr<bmodel::Model> model_;

};

}

#endif
