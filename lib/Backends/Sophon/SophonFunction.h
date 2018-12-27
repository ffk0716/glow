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
#include <libbmruntime/bmruntime.h>
#include <libbmruntime/bmruntime_bmnet.h>
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

  /// Allocate Mutable buffers on device this includes Activations and
  /// Placeholders.
  void setupRuns() override;
  /// Copy Input Placeholder data to position.
  void beforeRun(const Context &ctx) override;
  /// Copy Outputs to Placeholders in \p ctx.
  void afterRun(const Context &ctx) override;
  /// Final cleanup, free all allocations.
  void tearDownRuns() override;

  void execute() override;

private:
  std::unique_ptr<bmodel::Model> model_;
  bmnet_t net;
  bmctx_t bmctx;
  bmnet_output_info_t output_info;
};

} // namespace glow

#endif
