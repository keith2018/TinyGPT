/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "ParallelLinear.h"
#include "layer/Attention.h"
#include "layer/GatedMLP.h"

namespace tinygpt::distributed {

using TPAttention = tinytorch::nn::AttentionImpl<MergedColumnParallelLinear, RowParallelLinear>;
using TPAttentionWithQKNorm = tinytorch::nn::AttentionWithQKNormImpl<MergedColumnParallelLinear, RowParallelLinear>;
using TPGatedMLP = tinytorch::nn::GatedMLPImpl<MergedColumnParallelLinear, RowParallelLinear>;

using TPLmHead = VocabParallelLinear;

}  // namespace tinygpt::distributed
