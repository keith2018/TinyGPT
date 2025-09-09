/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Modules.h"

namespace tinygpt {

enum class GPTModelType : int8_t {
  UNKNOWN = 0,
  GPT2,
  LLAMA,
  QWEN2,
  QWEN3,
  MISTRAL,
};

struct KVCacheStates {
  tinytorch::TensorPair kv;
  int64_t pastLength;
};

class KVCacheManager {
 public:
  void create(size_t numLayers) { cache_.resize(numLayers); }

  void reset() { cache_.clear(); }

  KVCacheStates append(size_t layerIdx, const tinytorch::TensorPair &kv);

  int64_t pastLength(size_t layerIdx) const;

 private:
  std::vector<tinytorch::TensorPair> cache_;
};

class GPTModel {
 public:
  virtual ~GPTModel() = default;

  virtual GPTModelType type() { return GPTModelType::UNKNOWN; }

  tinytorch::Tensor forward(const tinytorch::Tensor &inputIds) { return model()(inputIds); }

  void resetCache() { kvCache_.reset(); }

  virtual bool load(const std::string &path) = 0;
  virtual int64_t numLayers() = 0;
  virtual int64_t contextSize() = 0;
  virtual tinytorch::nn::Module &model() = 0;

 protected:
  void init() { kvCache_.create(numLayers()); }

  KVCacheManager kvCache_;
};

}  // namespace tinygpt
