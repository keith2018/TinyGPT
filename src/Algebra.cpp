/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Algebra.h"

#include "mkl.h"

namespace TinyGPT {

void Algebra::gemm(float *c, const float *a, const float *b, uint32_t m, uint32_t k, uint32_t n) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b, n, 0.f, c, n);
}

void Algebra::gemmTrans(float *c, const float *a, const float *b, uint32_t m, uint32_t k, uint32_t n) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a, k, b, k, 0.f, c, n);
}

}
