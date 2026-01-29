/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Functions.h"

namespace tinygpt {

struct KVCacheStates {
  tinytorch::TensorPair kv;
  int64_t pastLength;
};

class KVCacheManager {
 public:
  void create(size_t numLayers) { cache_.resize(numLayers); }

  void reset() { cache_.clear(); }

  KVCacheStates append(size_t layerIdx, const tinytorch::TensorPair &kv, int64_t seqLenDim = 2) {
    ASSERT(layerIdx < cache_.size());
    auto &cached = cache_[layerIdx];
    int64_t pastLength = 0;

    if (cached.first.defined()) {
      ASSERT(cached.second.defined());

      // pastLength
      pastLength = cached.first.size(seqLenDim);

      // concat kv
      cached.first = tinytorch::function::concat({cached.first, kv.first}, seqLenDim);
      cached.second = tinytorch::function::concat({cached.second, kv.second}, seqLenDim);
    } else {
      cached = kv;
    }
    return {cached, pastLength};
  }

  int64_t pastLength(size_t layerIdx, int64_t seqLenDim = 2) const {
    ASSERT(layerIdx < cache_.size());
    auto &kv = cache_[layerIdx];
    if (kv.first.defined()) {
      return kv.first.size(seqLenDim);
    }
    return 0;
  }

 private:
  std::vector<tinytorch::TensorPair> cache_;
};

}  // namespace tinygpt
