/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "Tensor.h"

namespace tinygpt {

class GPTModel;

struct PagedKVCacheConfig {
  int32_t blockSize = 16;
  int64_t maxSeqLen = 4096;
  int64_t numBlocks = 0;
  float memoryUtil = 0.85f;
  int64_t reserveBytes = 1024LL << 20;  // 1024 MiB
};

struct PagedKVCacheSizing {
  int64_t numBlocks;
  int32_t blockSize;
  int32_t maxBlocksPerSeq;  // ceilDiv(maxSeqLen, blockSize)
};

class PagedKVCache {
 public:
  using SeqId = uint64_t;

  PagedKVCache(int64_t numLayers, int64_t numKvHeads, int64_t headDim, const PagedKVCacheSizing& sizing,
               tinytorch::Options options);

  static PagedKVCacheSizing autoSize(const GPTModel& model, tinytorch::DType dtype, const PagedKVCacheConfig& cfg);

  SeqId allocate();
  void free(SeqId seqId);

  bool appendTokens(SeqId seqId, int32_t numTokens, std::vector<int32_t>& outSlots);

  int32_t seqLen(SeqId seqId) const;

  const std::vector<int32_t>& blocksOf(SeqId seqId) const;

  int64_t numFreeBlocks() const;
  int64_t numTotalBlocks() const { return numBlocks_; }

  int32_t blockSize() const { return blockSize_; }
  int32_t maxBlocksPerSeq() const { return maxBlocksPerSeq_; }
  int64_t numLayers() const { return numLayers_; }
  int64_t numKvHeads() const { return numKvHeads_; }
  int64_t headDim() const { return headDim_; }

  tinytorch::Tensor& kPool(size_t layerIdx) { return kPool_[layerIdx]; }
  tinytorch::Tensor& vPool(size_t layerIdx) { return vPool_[layerIdx]; }

  int64_t totalBytes() const;

 private:
  struct SeqState {
    std::vector<int32_t> blocks;
    int32_t seqLen = 0;
  };

  int64_t numBlocks_;
  int32_t blockSize_;
  int32_t maxBlocksPerSeq_;
  int64_t numLayers_;
  int64_t numKvHeads_;
  int64_t headDim_;

  std::vector<tinytorch::Tensor> kPool_;  // per-layer
  std::vector<tinytorch::Tensor> vPool_;

  mutable std::mutex mutex_;
  std::vector<int32_t> freeBlocks_;
  std::unordered_map<SeqId, SeqState> seqs_;
  SeqId nextSeqId_ = 1;
};

}  // namespace tinygpt
