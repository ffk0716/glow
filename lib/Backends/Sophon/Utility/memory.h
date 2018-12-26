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

#include <cassert>
#include <cstddef>

namespace glow {
namespace sophon {

int idiv_round(int pNumerator, int pDenominator);
int align(int pNum, int pAlign);
size_t formula_4d_aligned(unsigned n, unsigned c, unsigned h, unsigned w,
                          unsigned npu_num, unsigned eu_num);
size_t formula_4d_nonaligned(unsigned n, unsigned c, unsigned h, unsigned w,
                             unsigned npu_num);

// Implicit W parameter Rule:
//   This is based on bmkernel implementation.
//   When W is not provided with 2D tensor, we use the algorithm to decide W.
struct SophonFCBiasDim {
  unsigned n, c, h, w;
  template <typename T> SophonFCBiasDim(T &vec) {
    n = c = h = w = 0;
    assert(vec.size() == 1);
    unsigned dim = vec[0];
    if (dim > 32) {
      w = 32;
    } else {
      w = 16;
    }
    n = 2;
    h = 1;
    c = glow::sophon::idiv_round(dim, w);
  }
};

struct SophonDim {
  unsigned n, c, h, w;
  template <typename T> SophonDim(const T &vec) {
    n = c = h = w = 0;
    size_t dim = vec.size();
    switch (dim) {
    case 4:
      n = vec[0];
      c = vec[1];
      h = vec[2];
      w = vec[3];
      break;
    case 2: {
      unsigned M = vec[0];
      unsigned N = vec[1];
      if (N > 32) {
        w = 32;
      } else {
        w = 16;
      }
      n = M;
      h = 1;
      c = glow::sophon::idiv_round(N, w);
    } break;
    case 1:
      n = 2;
      c = vec[0];
      h = 1;
      w = 1;
      break;
    default:
      assert(false && "Dimension is not between 1, 2, or 4");
      break;
    }
  }
};

} // namespace sophon
} // namespace glow
