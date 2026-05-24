/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor/Tensor.h"

namespace tinygpt::kernel {

// split-KV (paged attention)
// Minimum partition size for split-KV. Each partition handles at least this many KV tokens.
// Must be a multiple of kBc (64 for BF16/128/causal config).
inline constexpr int kSplitKvMinPartitionSize = 128;

// Maximum number of partitions (caps tmpO/tmpLse buffer size).
inline constexpr int kSplitKvMaxPartitions = 12;

// Legacy constant kept for backward compatibility with external code.
inline constexpr int kSplitKvPartitionSize = kSplitKvMinPartitionSize;

// query       : [totalTokensQ, numHeadsQ, headDim]
// kCachePool  : [numBlocks, numHeadsKV, pageSize, headDim]
// vCachePool  : [numBlocks, numHeadsKV, pageSize, headDim]
// cuSeqLensQ  : [batchSize + 1] int32
// cuSeqLensKV : [batchSize + 1] int32
// blockTable  : [batchSize, maxBlocksPerSeq] int32
// output      : [totalTokensQ, numHeadsQ, headDim]
tinytorch::Tensor flashAttentionPagedVarLen(const tinytorch::Tensor& query, const tinytorch::Tensor& kCachePool,
                                            const tinytorch::Tensor& vCachePool, const tinytorch::Tensor& cuSeqLensQ,
                                            const tinytorch::Tensor& cuSeqLensKV, const tinytorch::Tensor& blockTable,
                                            int maxSeqLenQ, int maxSeqLenKV, int pageSize, int maxBlocksPerSeq,
                                            bool isCausal = true, float* externalTmpO = nullptr,
                                            float* externalTmpLse = nullptr);

// key / value        : [numTokens, numKvHeads, headDim]
// keyCache / valCache: [numBlocks, numKvHeads, blockSize, headDim]
// slotMapping        : [numTokens] int32
void scatterKVToCache(const tinytorch::Tensor& key, const tinytorch::Tensor& value, tinytorch::Tensor& keyCache,
                      tinytorch::Tensor& valueCache, const tinytorch::Tensor& slotMapping, int blockSize);

// Fused RoPE(K) + ScatterKVToCache: applies rotary embedding to K in-place
// while writing K and V to the paged KV cache in a single kernel launch.
// This eliminates the intermediate rotated-K tensor and one kernel launch.
//
// key / value        : [numTokens, numKvHeads, headDim]  (K is NOT pre-rotated)
// keyCache / valCache: [numBlocks, numKvHeads, blockSize, headDim]
// slotMapping        : [numTokens] int32
// ropeCache          : [maxPos, headDim, 2] float32  (precomputed cos/sin)
// positions          : [numTokens] int64
void ropeScatterKVToCache(const tinytorch::Tensor& key, const tinytorch::Tensor& value, tinytorch::Tensor& keyCache,
                          tinytorch::Tensor& valueCache, const tinytorch::Tensor& slotMapping, int blockSize,
                          const tinytorch::Tensor& ropeCache, const tinytorch::Tensor& positions);

// Fused K-Norm + RoPE(K) + ScatterKVToCache: applies RMSNorm to K, then
// rotary embedding, then scattered write — all in a single kernel launch.
// Eliminates both the kNormSmall kernel for K and the intermediate tensors.
//
// key               : [numTokens, numKvHeads, headDim]  (raw, NOT normalized or rotated)
// value             : [numTokens, numKvHeads, headDim]
// keyCache/valCache : [numBlocks, numKvHeads, blockSize, headDim]
// slotMapping       : [numTokens] int32
// ropeCache         : [maxPos, headDim, 2] float32
// positions         : [numTokens] int64
// normWeight        : [headDim] — RMSNorm weight for K
// eps               : RMSNorm epsilon
void normRopeScatterKVToCache(const tinytorch::Tensor& key, const tinytorch::Tensor& value, tinytorch::Tensor& keyCache,
                              tinytorch::Tensor& valueCache, const tinytorch::Tensor& slotMapping, int blockSize,
                              const tinytorch::Tensor& ropeCache, const tinytorch::Tensor& positions,
                              const tinytorch::Tensor& normWeight, float eps);

}  // namespace tinygpt::kernel
