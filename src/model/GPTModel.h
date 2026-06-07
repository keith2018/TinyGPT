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
#include "kernel/EmbeddingOps.h"
#include "kernel/GemvOps.h"
#include "layer/DecoderLayer.h"
#include "layer/GatedMLP.h"
#include "layer/Linear.h"
#include "util/SafeTensors.h"

namespace tinytorch::nn {

template <typename AttnType, typename MLPType, typename LMHeadT = LmHeadLinear>
class CausalLM : public Module {
 public:
  using DecoderLayerType = DecoderLayer<AttnType, MLPType>;

  template <typename AttnFactory, typename MLPFactory>
  CausalLM(int64_t vocabSize, int64_t hiddenSize, int64_t numLayers, float rmsNormEps, bool tieWordEmbeddings,
           Options options, AttnFactory &&attnFactory, MLPFactory &&mlpFactory)
      : rmsNormEps_(rmsNormEps),
        embedTokens_(Embedding(vocabSize, hiddenSize, options)),
        layers_(ModuleList()),
        norm_(RMSNorm({hiddenSize}, rmsNormEps, options)),
        lmHead_(LMHeadT(hiddenSize, vocabSize, false, options)) {
    for (int64_t i = 0; i < numLayers; i++) {
      auto attn = attnFactory(i);
      auto mlp = mlpFactory(i);
      auto inputLn = RMSNorm({hiddenSize}, rmsNormEps, options);
      auto postAttnLn = RMSNorm({hiddenSize}, rmsNormEps, options);
      layers_.template emplaceBack<DecoderLayerType>(std::move(attn), std::move(mlp), std::move(inputLn),
                                                     std::move(postAttnLn));
    }

    if (tieWordEmbeddings) {
      // tie not compatible with vocab parallel: lmHead_ is sharded but
      // embedTokens_ is full, so storage shapes mismatch. fall back to
      // single-gpu lm_head when tying is required, or use a non-tied model.
      ASSERT(!tinygpt::distributed::Communicator::tp().enabled() &&
             "tied word embeddings not supported under tensor parallel");
      lmHead_.weight() = embedTokens_.weight();
    }

    registerModules({
        {"model.embed_tokens", embedTokens_},
        {"model.layers", layers_},
        {"model.norm", norm_},
        {"lm_head", lmHead_},
    });
  }

  // inputIds: [totalTokens] int64 -> output: [selectedTokens, vocabSize]
  Tensor forward(const Tensor &inputIds) override {
    // fast embedding lookup for decode (small token counts)
    Tensor hiddenStates;
    auto fastEmbed = tinygpt::kernel::embeddingLookup(inputIds, embedTokens_.weight());
    if (fastEmbed.defined()) {
      hiddenStates = fastEmbed;
    } else {
      hiddenStates = embedTokens_(inputIds);  // [totalTokens, hidden]
    }
    Tensor residual;

    for (auto &layer : layers_) {
      auto *typedLayer = static_cast<DecoderLayerType *>(layer.get());
      std::tie(hiddenStates, residual) = typedLayer->forward(std::move(hiddenStates), std::move(residual));
    }

    // final fused residual-add + RMSNorm
    function::fusedAddRmsNorm(hiddenStates, residual, norm_.weight(), rmsNormEps_);

    // select last-token rows before lm_head
    auto *ctx = tinygpt::ForwardContext::current();
    if (ctx && ctx->lastTokenIndices.defined()) {
      hiddenStates = function::indexSelect(hiddenStates, 0, ctx->lastTokenIndices);
    }

    // lm_head: gemv fast-path for M==1 is internal to the linear class;
    // VocabParallelLinear additionally allGathers logits across ranks.
    return lmHead_(hiddenStates);
  }

 protected:
  float rmsNormEps_;
  Embedding embedTokens_;
  ModuleList layers_;
  RMSNorm norm_;
  LMHeadT lmHead_;
};

}  // namespace tinytorch::nn

namespace tinygpt {

enum class GPTModelType : int8_t {
  UNKNOWN = 0,
  LLAMA,
  QWEN2,
  QWEN3,
  MISTRAL,
};

class GPTModel {
 public:
  struct ModelDims {
    int64_t numLayers = 0;
    int64_t contextSize = 0;
    int64_t numHeads = 0;
    int64_t numKvHeads = 0;
    int64_t headDim = 0;
    int64_t hiddenSize = 0;
  };

  virtual ~GPTModel() = default;

  virtual GPTModelType type() const { return GPTModelType::UNKNOWN; }

  tinytorch::Tensor forward(const tinytorch::Tensor &inputIds) { return model()(inputIds); }

  PagedKVCache *pagedCache() const { return pagedCache_; }
  void setPagedCache(PagedKVCache *cache) { pagedCache_ = cache; }

  virtual bool load(const std::string &path) { return SafeTensors::load(model(), path, false); }

  int64_t numLayers() const { return dims_.numLayers; }
  int64_t contextSize() const { return dims_.contextSize; }
  int64_t numHeads() const { return dims_.numHeads; }
  int64_t numKvHeads() const { return dims_.numKvHeads; }
  int64_t headDim() const { return dims_.headDim; }
  int64_t hiddenSize() const { return dims_.hiddenSize; }
  tinytorch::Device device() const { return device_; }

  virtual tinytorch::nn::Module &model() = 0;

 protected:
  GPTModel(ModelDims dims, tinytorch::Device device) : dims_(dims), device_(device) {}

  ModelDims dims_;
  tinytorch::Device device_;
  PagedKVCache *pagedCache_ = nullptr;
};

}  // namespace tinygpt
