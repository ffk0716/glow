/*
 * bmnet/lib/Backends/Sophon/SophonFunction.h
 *
 * Copyright Bitmain Technologies Inc.
 * Written by:
 *   Wanwei CAI <wanwei.cai@bitmain.com>
 * Created Time: 2018-10-15 09:58
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

} // namespace glow

#endif
