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
#include "memory.h"
namespace glow {
namespace sophon {

int idiv_round(int pNumerator, int pDenominator) {
  return (pNumerator + pDenominator - 1) / pDenominator;
}

int align(int pNum, int pAlign) {
  int mask = pAlign - 1;
  return (pNum + mask) & ~mask;
}
size_t formula_4d_aligned(unsigned n, unsigned c, unsigned h, unsigned w,
                          unsigned npu_num, unsigned eu_num) {
  return idiv_round(c, npu_num) * n * align((h * w), eu_num);
}

size_t formula_4d_nonaligned(unsigned n, unsigned c, unsigned h, unsigned w,
                             unsigned npu_num) {
  return idiv_round(c, npu_num) * n * (h * w);
}
} // namespace sophon
} // namespace glow
