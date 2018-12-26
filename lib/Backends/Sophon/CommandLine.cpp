#include "CommandLine.h"

namespace glow {

llvm::cl::opt<std::string>
    bmodelFileNameOpt("bmodel", llvm::cl::desc("Specify file name to bmodel\n"),
                      llvm::cl::value_desc("bmodelPath"),
                      llvm::cl::init("tmp.bmodel"),
                      llvm::cl::cat(SophonBackendCat));
llvm::cl::opt<bool>
    enableLayerGroupOpt("enable-layer-group",
                        llvm::cl::desc("Enbale layer group optimization\n"),
                        llvm::cl::init(true), llvm::cl::cat(SophonBackendCat));

llvm::cl::OptionCategory SophonBackendCat("Sophon Backend Options");

} // namespace glow
