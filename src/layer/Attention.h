/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Modules.h"
#include "distributed/Communicator.h"
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

template <typename QKVProjT = MergedLinear, typename OProjT = GemvLinear>
class AttentionImpl : public Module {
 public:
  AttentionImpl(size_t layerIdx, const AttentionConfig &config, RoPE &&rope, Options options = {})
      : layerIdx_(layerIdx),
        numHeads_(localCount(config.numHeads)),
        headDim_(config.headDim),
        numKvHeads_(localCount(config.numKvHeads)),
        qDim_(numHeads_ * config.headDim),
        kvDim_(numKvHeads_ * config.headDim),
        qkvProj_(QKVProjT(
            config.hiddenSize,
            {config.numHeads * config.headDim, config.numKvHeads * config.headDim, config.numKvHeads * config.headDim},
            config.qkvBias, options)),
        oProj_(OProjT(config.numHeads * config.headDim, config.hiddenSize, config.oBias, options)),
        rope_(std::move(rope)) {
    ASSERT(config.numHeads % config.numKvHeads == 0);
    registerSubModules();
  }

  AttentionImpl(AttentionImpl &&other) noexcept
      : layerIdx_(other.layerIdx_),
        numHeads_(other.numHeads_),
        headDim_(other.headDim_),
        numKvHeads_(other.numKvHeads_),
        qDim_(other.qDim_),
        kvDim_(other.kvDim_),
        qkvProj_(std::move(other.qkvProj_)),
        oProj_(std::move(other.oProj_)),
        rope_(std::move(other.rope_)) {
    this->subModules_.clear();
    registerSubModules();
  }

  AttentionImpl &operator=(AttentionImpl &&) = delete;
  AttentionImpl(const AttentionImpl &) = delete;
  AttentionImpl &operator=(const AttentionImpl &) = delete;

 protected:
  void registerSubModules() {
    this->registerModules({
        {"q_proj", qkvProj_.moduleRefs(0)},
        {"k_proj", qkvProj_.moduleRefs(1)},
        {"v_proj", qkvProj_.moduleRefs(2)},
        {"o_proj", oProj_},
    });
  }

  // local head count = full / worldSize
  static int64_t localCount(int64_t full) {
    int ws = tinygpt::distributed::Communicator::tp().worldSize();
    ASSERT(full % ws == 0);
    return full / ws;
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

  QKVProjT qkvProj_;
  OProjT oProj_;

  RoPE rope_;
};

template <typename QKVProjT = MergedLinear, typename OProjT = GemvLinear>
class AttentionWithQKNormImpl : public AttentionImpl<QKVProjT, OProjT> {
 public:
  using Base = AttentionImpl<QKVProjT, OProjT>;

  AttentionWithQKNormImpl(size_t layerIdx, const AttentionConfig &config, RoPE &&rope, float rmsNormEps,
                          Options options = {})
      : Base(layerIdx, config, std::move(rope), options),
        rmsNormEps_(rmsNormEps),
        qNorm_(RMSNorm({config.headDim}, rmsNormEps, options)),
        kNorm_(RMSNorm({config.headDim}, rmsNormEps, options)) {
    registerQkNorm_Modules();
  }

  AttentionWithQKNormImpl(AttentionWithQKNormImpl &&other) noexcept
      : Base(std::move(other)),
        rmsNormEps_(other.rmsNormEps_),
        qNorm_(std::move(other.qNorm_)),
        kNorm_(std::move(other.kNorm_)) {
    registerQkNorm_Modules();
  }
  AttentionWithQKNormImpl &operator=(AttentionWithQKNormImpl &&) = delete;
  AttentionWithQKNormImpl(const AttentionWithQKNormImpl &) = delete;
  AttentionWithQKNormImpl &operator=(const AttentionWithQKNormImpl &) = delete;

  // fuse K-Norm + RoPE(K) + ScatterKV into one kernel
  Tensor forward(const Tensor &input) override {
    auto *ctx = tinygpt::ForwardContext::current();
    ASSERT(ctx != nullptr && ctx->pagedCache != nullptr);

    this->rope_.to(input.device());

    auto total = input.size(0);

    auto qkv = this->qkvProj_(input);
    auto qkvSplit = qkv.split({this->qDim_, this->kvDim_, this->kvDim_}, -1);
    auto queries = qkvSplit[0].view({total, this->numHeads_, this->headDim_});
    auto keys = qkvSplit[1].view({total, this->numKvHeads_, this->headDim_});
    auto values = qkvSplit[2].view({total, this->numKvHeads_, this->headDim_});

    queries = qNorm_(queries);
    queries = this->rope_.apply(queries, ctx->positions);

    // fused K-norm + RoPE(K) + scatter KV
    auto &kPool = ctx->pagedCache->kPool(this->layerIdx_);
    auto &vPool = ctx->pagedCache->vPool(this->layerIdx_);
    tinygpt::kernel::normRopeScatterKVToCache(keys, values, kPool, vPool, ctx->slotMapping, ctx->pageSize,
                                              this->rope_.cache(), ctx->positions, kNorm_.weight(), rmsNormEps_);

    // paged flash attention
    auto attnOutput = tinygpt::kernel::flashAttentionPagedVarLen(
        queries, kPool, vPool, ctx->cuSeqLensQ, ctx->cuSeqLensKV, ctx->blockTable, ctx->maxSeqLenQ, ctx->maxSeqLenKV,
        ctx->pageSize, ctx->maxBlocksPerSeq, /*isCausal=*/true, ctx->tmpO, ctx->tmpLse);
    ASSERT(attnOutput.defined());

    return this->oProj_(attnOutput.reshape({total, this->qDim_}));
  }

 private:
  void registerQkNorm_Modules() {
    this->registerModules({
        {"q_norm", qNorm_},
        {"k_norm", kNorm_},
    });
  }

 protected:
  float rmsNormEps_;
  RMSNorm qNorm_;
  RMSNorm kNorm_;
};

// default aliases (single-gpu)
using Attention = AttentionImpl<MergedLinear, GemvLinear>;
using AttentionWithQKNorm = AttentionWithQKNormImpl<MergedLinear, GemvLinear>;

}  // namespace tinytorch::nn
