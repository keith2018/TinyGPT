/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Functions.h"
#include "Modules.h"
#include "layer/Linear.h"

namespace tinytorch::nn {

struct AttentionConfig {
  int64_t hiddenSize = 0;
  int64_t numHeads = 0;
  int64_t headDim = 0;
  int64_t numKvHeads = 0;
  bool qkvBias = false;
  bool oBias = false;
};

class Attention : public Module {
 public:
  Attention(tinygpt::KVCacheManager *kvCache, size_t layerIdx, const AttentionConfig &config, RoPE &&rope,
            Options options = {})
      : kvCache_(kvCache),
        layerIdx_(layerIdx),
        numHeads_(config.numHeads),
        headDim_(config.headDim),
        numKvHeads_(config.numKvHeads),
        qDim_(config.numHeads * config.headDim),
        kvDim_(config.numKvHeads * config.headDim),
        qkvProj_(MergedLinear(config.hiddenSize, {qDim_, kvDim_, kvDim_}, config.qkvBias, options)),
        oProj_(Linear(qDim_, config.hiddenSize, config.oBias, options)),
        rope_(std::move(rope)) {
    ASSERT(config.numHeads % config.numKvHeads == 0);
    registerSubModules();
  }

  Attention(Attention &&other) noexcept
      : kvCache_(other.kvCache_),
        layerIdx_(other.layerIdx_),
        numHeads_(other.numHeads_),
        headDim_(other.headDim_),
        numKvHeads_(other.numKvHeads_),
        qDim_(other.qDim_),
        kvDim_(other.kvDim_),
        qkvProj_(std::move(other.qkvProj_)),
        oProj_(std::move(other.oProj_)),
        rope_(std::move(other.rope_)) {
    registerSubModules();
  }

  Attention &operator=(Attention &&) = delete;
  Attention(const Attention &) = delete;
  Attention &operator=(const Attention &) = delete;

 protected:
  void registerSubModules() {
    registerModules({
        {"q_proj", qkvProj_.moduleRefs(0)},
        {"k_proj", qkvProj_.moduleRefs(1)},
        {"v_proj", qkvProj_.moduleRefs(2)},
        {"o_proj", oProj_},
    });
  }

 public:
  Tensor forward(const Tensor &input) override {
    rope_.to(input.device());

    auto batchSize = input.shape(0);
    auto seqLen = input.shape(1);

    // qkv project
    auto [queries, keys, values] = projectQKV(input, batchSize, seqLen);

    // rope
    int64_t pastLength = kvCache_->pastLength(layerIdx_, 1);
    queries = rope_(queries, pastLength, QKVLayout::BSHD);
    keys = rope_(keys, pastLength, QKVLayout::BSHD);

    // attn
    auto attnOutput = computeAttention(queries, keys, values, batchSize, seqLen);
    ASSERT(attnOutput.defined());

    // o project
    return oProj_(attnOutput);
  }

 protected:
  virtual std::tuple<Tensor, Tensor, Tensor> projectQKV(const Tensor &input, int64_t batchSize, int64_t seqLen) {
    auto qkv = qkvProj_(input);
    auto qkvSplit = qkv.split({qDim_, kvDim_, kvDim_}, -1);
    auto queries = qkvSplit[0].view({batchSize, seqLen, numHeads_, headDim_});
    auto keys = qkvSplit[1].view({batchSize, seqLen, numKvHeads_, headDim_});
    auto values = qkvSplit[2].view({batchSize, seqLen, numKvHeads_, headDim_});
    return {queries, keys, values};
  }

  Tensor computeAttention(const Tensor &queries, const Tensor &keys, const Tensor &values, int64_t batchSize,
                          int64_t seqLen) {
    // BSHD: seqLenDim = 1
    auto kvStates = kvCache_->append(layerIdx_, {keys, values}, 1);

    bool isCausal = (kvStates.pastLength == 0);
    auto attnOutput = function::flashAttention(queries, kvStates.kv.first, kvStates.kv.second, isCausal);

    return attnOutput.reshape({batchSize, seqLen, qDim_});
  }

  tinygpt::KVCacheManager *kvCache_;
  size_t layerIdx_;
  int64_t numHeads_;
  int64_t headDim_;
  int64_t numKvHeads_;
  int64_t qDim_;
  int64_t kvDim_;

  MergedLinear qkvProj_;
  Linear oProj_;

  RoPE rope_;
};

class AttentionWithQKNorm : public Attention {
 public:
  AttentionWithQKNorm(tinygpt::KVCacheManager *kvCache, size_t layerIdx, const AttentionConfig &config, RoPE &&rope,
                      float rmsNormEps, Options options = {})
      : Attention(kvCache, layerIdx, config, std::move(rope), options),
        qNorm_(RMSNorm({config.headDim}, rmsNormEps, options)),
        kNorm_(RMSNorm({config.headDim}, rmsNormEps, options)) {
    registerQkNorm_Modules();
  }

  AttentionWithQKNorm(AttentionWithQKNorm &&other) noexcept
      : Attention(std::move(other)), qNorm_(std::move(other.qNorm_)), kNorm_(std::move(other.kNorm_)) {
    registerQkNorm_Modules();
  }

  AttentionWithQKNorm &operator=(AttentionWithQKNorm &&) = delete;
  AttentionWithQKNorm(const AttentionWithQKNorm &) = delete;
  AttentionWithQKNorm &operator=(const AttentionWithQKNorm &) = delete;

 private:
  void registerQkNorm_Modules() {
    registerModules({
        {"q_norm", qNorm_},
        {"k_norm", kNorm_},
    });
  }

 protected:
  std::tuple<Tensor, Tensor, Tensor> projectQKV(const Tensor &input, int64_t batchSize, int64_t seqLen) override {
    auto qkv = qkvProj_(input);
    auto qkvSplit = qkv.split({qDim_, kvDim_, kvDim_}, -1);
    auto queries = qNorm_(qkvSplit[0].view({batchSize, seqLen, numHeads_, headDim_}));
    auto keys = kNorm_(qkvSplit[1].view({batchSize, seqLen, numKvHeads_, headDim_}));
    auto values = qkvSplit[2].view({batchSize, seqLen, numKvHeads_, headDim_});
    return {queries, keys, values};
  }

  RMSNorm qNorm_;
  RMSNorm kNorm_;
};

}  // namespace tinytorch::nn
