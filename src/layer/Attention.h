/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Modules.h"
#include "engine/ForwardContext.h"
#include "engine/PagedKVCache.h"
#include "kernel/AttentionOps.h"
#include "layer/Linear.h"
#include "layer/RoPE.h"

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
  Attention(size_t layerIdx, const AttentionConfig &config, RoPE &&rope, Options options = {})
      : layerIdx_(layerIdx),
        numHeads_(config.numHeads),
        headDim_(config.headDim),
        numKvHeads_(config.numKvHeads),
        qDim_(config.numHeads * config.headDim),
        kvDim_(config.numKvHeads * config.headDim),
        qkvProj_(MergedLinear(config.hiddenSize, {qDim_, kvDim_, kvDim_}, config.qkvBias, options)),
        oProj_(GemvLinear(qDim_, config.hiddenSize, config.oBias, options)),
        rope_(std::move(rope)) {
    ASSERT(config.numHeads % config.numKvHeads == 0);
    registerSubModules();
  }

  Attention(Attention &&other) noexcept
      : layerIdx_(other.layerIdx_),
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
    auto *ctx = tinygpt::ForwardContext::current();
    ASSERT(ctx != nullptr && ctx->pagedCache != nullptr);

    rope_.to(input.device());

    auto total = input.size(0);

    auto [queries, keys, values] = projectQKV(input, total);

    // rope for Q
    queries = rope_.apply(queries, ctx->positions);

    // fused RoPE(K) + scatter K,V to paged cache
    auto &kPool = ctx->pagedCache->kPool(layerIdx_);
    auto &vPool = ctx->pagedCache->vPool(layerIdx_);
    tinygpt::kernel::ropeScatterKVToCache(keys, values, kPool, vPool, ctx->slotMapping, ctx->pageSize, rope_.cache(),
                                          ctx->positions);

    // paged flash attention
    auto attnOutput = tinygpt::kernel::flashAttentionPagedVarLen(
        queries, kPool, vPool, ctx->cuSeqLensQ, ctx->cuSeqLensKV, ctx->blockTable, ctx->maxSeqLenQ, ctx->maxSeqLenKV,
        ctx->pageSize, ctx->maxBlocksPerSeq, /*isCausal=*/true, ctx->tmpO, ctx->tmpLse);
    ASSERT(attnOutput.defined());

    // output projection (packed)
    return oProj_(attnOutput.reshape({total, qDim_}));
  }

 protected:
  virtual std::tuple<Tensor, Tensor, Tensor> projectQKV(const Tensor &input, int64_t totalTokens) {
    auto qkv = qkvProj_(input);
    auto qkvSplit = qkv.split({qDim_, kvDim_, kvDim_}, -1);
    auto queries = qkvSplit[0].view({totalTokens, numHeads_, headDim_});
    auto keys = qkvSplit[1].view({totalTokens, numKvHeads_, headDim_});
    auto values = qkvSplit[2].view({totalTokens, numKvHeads_, headDim_});
    return {queries, keys, values};
  }

  size_t layerIdx_;
  int64_t numHeads_;
  int64_t headDim_;
  int64_t numKvHeads_;
  int64_t qDim_;
  int64_t kvDim_;

  MergedLinear qkvProj_;
  GemvLinear oProj_;

  RoPE rope_;
};

class AttentionWithQKNorm : public Attention {
 public:
  AttentionWithQKNorm(size_t layerIdx, const AttentionConfig &config, RoPE &&rope, float rmsNormEps,
                      Options options = {})
      : Attention(layerIdx, config, std::move(rope), options),
        rmsNormEps_(rmsNormEps),
        qNorm_(RMSNorm({config.headDim}, rmsNormEps, options)),
        kNorm_(RMSNorm({config.headDim}, rmsNormEps, options)) {
    registerQkNorm_Modules();
  }

  AttentionWithQKNorm(AttentionWithQKNorm &&other) noexcept
      : Attention(std::move(other)),
        rmsNormEps_(other.rmsNormEps_),
        qNorm_(std::move(other.qNorm_)),
        kNorm_(std::move(other.kNorm_)) {
    registerQkNorm_Modules();
  }

  AttentionWithQKNorm &operator=(AttentionWithQKNorm &&) = delete;
  AttentionWithQKNorm(const AttentionWithQKNorm &) = delete;
  AttentionWithQKNorm &operator=(const AttentionWithQKNorm &) = delete;

  // fuse K-Norm + RoPE(K) + ScatterKV into one kernel
  Tensor forward(const Tensor &input) override {
    auto *ctx = tinygpt::ForwardContext::current();
    ASSERT(ctx != nullptr && ctx->pagedCache != nullptr);

    rope_.to(input.device());

    auto total = input.size(0);

    auto qkv = qkvProj_(input);
    auto qkvSplit = qkv.split({qDim_, kvDim_, kvDim_}, -1);
    auto queries = qkvSplit[0].view({total, numHeads_, headDim_});
    auto keys = qkvSplit[1].view({total, numKvHeads_, headDim_});
    auto values = qkvSplit[2].view({total, numKvHeads_, headDim_});

    queries = qNorm_(queries);
    queries = rope_.apply(queries, ctx->positions);

    // fused K-norm + RoPE(K) + scatter KV
    auto &kPool = ctx->pagedCache->kPool(layerIdx_);
    auto &vPool = ctx->pagedCache->vPool(layerIdx_);
    tinygpt::kernel::normRopeScatterKVToCache(keys, values, kPool, vPool, ctx->slotMapping, ctx->pageSize,
                                              rope_.cache(), ctx->positions, kNorm_.weight(), rmsNormEps_);

    // paged flash attention
    auto attnOutput = tinygpt::kernel::flashAttentionPagedVarLen(
        queries, kPool, vPool, ctx->cuSeqLensQ, ctx->cuSeqLensKV, ctx->blockTable, ctx->maxSeqLenQ, ctx->maxSeqLenKV,
        ctx->pageSize, ctx->maxBlocksPerSeq, /*isCausal=*/true, ctx->tmpO, ctx->tmpLse);
    ASSERT(attnOutput.defined());

    return oProj_(attnOutput.reshape({total, qDim_}));
  }

 private:
  void registerQkNorm_Modules() {
    registerModules({
        {"q_norm", qNorm_},
        {"k_norm", kNorm_},
    });
  }

 protected:
  float rmsNormEps_;
  RMSNorm qNorm_;
  RMSNorm kNorm_;
};

}  // namespace tinytorch::nn
