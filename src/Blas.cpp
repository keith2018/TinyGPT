/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Blas.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include "mkl.h"
#endif

namespace TinyGPT {

void Blas::gemm(float *c, const float *a, const float *b, uint32_t m,
                   uint32_t k, uint32_t n) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b,
              n, 0.f, c, n);
}

void Blas::gemmTrans(float *c, const float *a, const float *b, uint32_t m,
                        uint32_t k, uint32_t n) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a, k, b,
              k, 0.f, c, n);
}

} // namespace TinyGPT
