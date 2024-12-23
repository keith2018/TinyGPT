/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cstdint>

namespace TinyGPT {

class Blas {
 public:

  /**
   * c = a * b
   */
  static void gemm(float *c, const float *a, const float *b, uint32_t m, uint32_t k, uint32_t n);

  /**
   * c = a * b.T
   */
  static void gemmTrans(float *c, const float *a, const float *b, uint32_t m, uint32_t k, uint32_t n);
};

}
