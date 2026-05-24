/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor/Tensor.h"

namespace tinygpt::kernel {

// Optimized GEMV for lm_head projection: output = input × weight^T
//
// Specialized for the decode case (M=1) where input is a single hidden-state
// vector and weight is the large vocabulary projection matrix.
//
// input  : [M, K]  where M is typically 1 (single token decode)
// weight : [N, K]  where N = vocabSize (large, e.g. 151936)
// output : [M, N]
//
// This kernel achieves higher memory bandwidth utilization than cuBLAS's
// generic gemv2T for the specific case of large N, moderate K, M=1.
// Falls back to cuBLAS for M > 1 or when K is not supported.
tinytorch::Tensor gemvLmHead(const tinytorch::Tensor& input, const tinytorch::Tensor& weight);

// General-purpose GEMV for Linear layers (no bias): output = input × weight^T
//
// Equivalent to gemvLmHead but intended for all Linear projections (qkv_proj,
// o_proj, gate_up_proj, down_proj) during M=1 decode. Provides significant
// speedup over cuBLAS GEMM which may pick suboptimal Tensor Core kernels
// (e.g. CUTLASS WMMA 16x16) when M=1.
//
// input  : [1, K]
// weight : [N, K]
// output : [1, N]
//
// Returns empty tensor if conditions not met (caller should fall back to GEMM).
tinytorch::Tensor gemvLinear(const tinytorch::Tensor& input, const tinytorch::Tensor& weight);

}  // namespace tinygpt::kernel
