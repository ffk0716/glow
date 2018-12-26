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
#ifndef GLOW_BACKENDS_CPU_ALLOCATIONSINFO_H
#define GLOW_BACKENDS_CPU_ALLOCATIONSINFO_H

#include "llvm/ADT/DenseMap.h"

namespace glow {
class Value;
class IRFunction;
class Context;

/// Information about allocations for activations, constant weight variables
/// and mutable weight variables.
class AllocationsInfo {

public:
  virtual ~AllocationsInfo() = default;
  /// Assign offsets to all of the variables in the module \p M and to the
  /// placeholders. \p ctx is the context that maps the graph to the concrete
  /// execution environment for a specific function.
  /// If the \p absoluteAddr is true, simply reuse the addresses already used
  /// by the payloads of tensors corresponding to those WeightVars as offsets.
  /// This is useful in a JIT setup. If \p absoluteAddr is false, then all the
  /// WeightVars will get new offsets assigned.
  virtual void allocateWeightVars(const IRFunction *F) = 0;
  /// Assign offsets to all activations.
  /// No actual memory allocation is performed. All the allocations should be
  /// performed by the client based on the information provided by the
  /// AllocationsInfo.
  virtual void allocateActivations(const IRFunction *F) = 0;
  /// Assign offsets to all tensorviews.
  /// No memory allocation is performed. Sets up all offsets into already
  /// defined offsets for WeightVars and AllocActivations. Assumes the weight
  /// vars and alloc activations have already been added to allocatedAddressed_.
  virtual void allocateTensorViews(const IRFunction *F) = 0;
  /// Number all allocations and weight variables by assigning them unique
  /// numbers.
  virtual void numberValues(const IRFunction *F) = 0;

  virtual llvm::DenseMap<const Value *, uint64_t> &getAllocatedAddress() = 0;
  virtual void setActivationsMemSize(size_t v) = 0;
  virtual size_t getActivationsMemSize() = 0;
};

} // namespace glow
#endif // GLOW_BACKENDS_CPU_ALLOCATIONSINFO_H
