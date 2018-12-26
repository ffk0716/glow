/*
 * bmnet/lib/Backends/Sophon/CommandLine.h
 *
 * Copyright Bitmain Technologies Inc.
 * Written by:
 *   Wanwei CAI <wanwei.cai@bitmain.com>
 * Created Time: 2018-10-15 09:54
 */

#ifndef _COMMANDLINE_H
#define _COMMANDLINE_H

#include "llvm/Support/CommandLine.h"

namespace glow {
extern llvm::cl::OptionCategory SophonBackendCat;
extern llvm::cl::opt<std::string> bmodelFileNameOpt;
extern llvm::cl::opt<bool> enableLayerGroupOpt;
} // namespace glow

#endif
