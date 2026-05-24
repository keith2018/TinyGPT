/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "Tensor/Tensor.h"
#include "Utils/Macros.h"

namespace tinygpt::kernel {

#define TINYGPT_DISPATCH_FLOAT_DTYPE(tensor, ...)   \
  do {                                              \
    switch ((tensor).dtype()) {                     \
      case ::tinytorch::DType::Float16: {           \
        using CudaT = __half;                       \
        __VA_ARGS__                                 \
      } break;                                      \
      case ::tinytorch::DType::BFloat16: {          \
        using CudaT = __nv_bfloat16;                \
        __VA_ARGS__                                 \
      } break;                                      \
      case ::tinytorch::DType::Float32: {           \
        using CudaT = float;                        \
        __VA_ARGS__                                 \
      } break;                                      \
      default:                                      \
        ASSERT(false && "unsupported float dtype"); \
    }                                               \
  } while (0)

}  // namespace tinygpt::kernel
