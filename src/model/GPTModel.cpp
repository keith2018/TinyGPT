/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "GPTModel.h"

#include "Functions.h"

namespace tinygpt {

tinytorch::TensorPair KVCacheManager::append(size_t layerIdx, const tinytorch::TensorPair &kv) {
  ASSERT(layerIdx < cache_.size());
  auto &cached = cache_[layerIdx];
  if (cached.first.defined()) {
    ASSERT(cached.second.defined());
    cached.first = tinytorch::function::concat({cached.first, kv.first}, 2);
    cached.second = tinytorch::function::concat({cached.second, kv.second}, 2);
  } else {
    cached = kv;
  }
  return cached;
}

int64_t KVCacheManager::pastLength(size_t layerIdx) const {
  ASSERT(layerIdx < cache_.size());
  auto &kv = cache_[layerIdx];
  if (kv.first.defined()) {
    return kv.first.size(2);
  }
  return 0;
}

}  // namespace tinygpt