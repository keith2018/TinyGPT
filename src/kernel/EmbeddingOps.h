/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor/Tensor.h"

namespace tinygpt::kernel {

// Fast embedding lookup for single-token decode.
//
// The generic kIndex kernel used by TinyTorch's Embedding module takes ~74μs
// per token due to complex index computation overhead.  For the common decode
// case (1 token at a time), we need only copy a single row from the embedding
// table — a 2KB memcpy for hidden_size=1024 BF16.
//
// This specialized kernel completes in <3μs by using a single thread block
// with vectorized 128-bit loads to copy the row directly.
//
// inputIds: [numTokens] int64 — token IDs to look up
// weight:   [vocabSize, hiddenSize] — embedding table
// output:   [numTokens, hiddenSize]
//
// Falls back to empty tensor if conditions not met (caller should use generic path).
tinytorch::Tensor embeddingLookup(const tinytorch::Tensor& inputIds, const tinytorch::Tensor& weight);

}  // namespace tinygpt::kernel
