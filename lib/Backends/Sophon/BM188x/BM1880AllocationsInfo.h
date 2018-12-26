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
#pragma once

#include "Backends/Sophon/AllocationsInfo.h"
#include "Backends/Sophon/SophonTargetTransformInfo.h"
#include "glow/CodeGen/MemoryAllocator.h"
#include "llvm/ADT/DenseMap.h"

namespace glow {

/// Information about allocations for activations, constant weight variables
/// and mutable weight variables.
class BM1880AllocationsInfo : public AllocationsInfo {

private:
  void allocateNeuron(WeightVar *W, size_t &neuronOffset);

private:
  /// Different kinds of values that need to be allocated.
  enum class ValueKind { Global_Weight, Global_Neuron, Local_Memory };

  /// Maps Values in the module to their offsets.
  llvm::DenseMap<const Value *, uint64_t> allocatedAddressed_;
  /// Amount of memory to be allocated for constant WeightVars.
  size_t globa_weight_sizes_{0};
  /// Amount of memory to be allocated for mutable WeightVars.
  size_t global_neuron_sizes_{0};
  /// Amount of memory to be allocated for activations.
  size_t local_memory_sizes_{0};

  const Context *ctx_;
  const sophon::SophonTargetTransformInfo *TTI_;

public:
  BM1880AllocationsInfo();
  BM1880AllocationsInfo(const Context &ctx,
                        const sophon::SophonTargetTransformInfo *TTI);
  void allocateWeightVars(const IRFunction *F) override;
  void allocateActivations(const IRFunction *F) override;
  void allocateTensorViews(const IRFunction *F) override {
    llvm_unreachable("unsupported!");
  }
  /// Number all allocations and weight variables by assigning them unique
  /// numbers.
  void numberValues(const IRFunction *F) override { llvm_unreachable("TODO!"); }

public:
  llvm::DenseMap<const Value *, uint64_t> &getAllocatedAddress() override;
  void setActivationsMemSize(size_t v) override;
  size_t getActivationsMemSize() override;
};

} // namespace glow
