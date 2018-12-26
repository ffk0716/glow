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
