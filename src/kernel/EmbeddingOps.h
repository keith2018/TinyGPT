/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor/Tensor.h"

namespace tinygpt::kernel {

// inputIds: [numTokens] int64 — token IDs to look up
// weight:   [vocabSize, hiddenSize] — embedding table
// output:   [numTokens, hiddenSize]
tinytorch::Tensor embeddingLookup(const tinytorch::Tensor& inputIds, const tinytorch::Tensor& weight);

}  // namespace tinygpt::kernel
