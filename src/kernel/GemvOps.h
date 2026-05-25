/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor/Tensor.h"

namespace tinygpt::kernel {

// input  : [M, K]  where M is typically 1 (single token decode)
// weight : [N, K]  where N = vocabSize
// output : [M, N]
tinytorch::Tensor gemvLmHead(const tinytorch::Tensor& input, const tinytorch::Tensor& weight);

// input  : [1, K]
// weight : [N, K]
// output : [1, N]
tinytorch::Tensor gemvLinear(const tinytorch::Tensor& input, const tinytorch::Tensor& weight);

}  // namespace tinygpt::kernel
