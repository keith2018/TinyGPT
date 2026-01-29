/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Modules.h"
#include "engine/CacheManager.h"
#include "layer/Attention.h"
#include "layer/DecoderLayer.h"
#include "layer/GatedMLP.h"
#include "util/SafeTensors.h"

namespace tinytorch::nn {

template <typename AttnType, typename MLPType>
class CausalLM : public Module {
 public:
  using DecoderLayerType = DecoderLayer<AttnType, MLPType>;

  template <typename AttnFactory, typename MLPFactory>
  CausalLM(int64_t vocabSize, int64_t hiddenSize, int64_t numLayers, float rmsNormEps, bool tieWordEmbeddings,
           Options options, AttnFactory &&attnFactory, MLPFactory &&mlpFactory)
      : embedTokens_(Embedding(vocabSize, hiddenSize, options)),
        layers_(ModuleList()),
        norm_(RMSNorm({hiddenSize}, rmsNormEps, options)),
        lmHead_(Linear(hiddenSize, vocabSize, false, options)) {
    for (int i = 0; i < numLayers; i++) {
      auto attn = attnFactory(i);
      auto mlp = mlpFactory(i);
      auto inputLn = RMSNorm({hiddenSize}, rmsNormEps, options);
      auto postAttnLn = RMSNorm({hiddenSize}, rmsNormEps, options);
      layers_.template emplaceBack<DecoderLayerType>(std::move(attn), std::move(mlp), std::move(inputLn),
                                                     std::move(postAttnLn));
    }

    if (tieWordEmbeddings) {
      lmHead_.weight() = embedTokens_.weight();
    }

    registerModules({
        {"model.embed_tokens", embedTokens_},
        {"model.layers", layers_},
        {"model.norm", norm_},
        {"lm_head", lmHead_},
    });
  }

  Tensor forward(const Tensor &inputIds) override {
    auto x = embedTokens_(inputIds);
    for (auto &layer : layers_) {
      x = layer->forward(x);
    }
    x = norm_(x);
    return lmHead_(x);
  }

 protected:
  Embedding embedTokens_;
  ModuleList layers_;
  RMSNorm norm_;
  Linear lmHead_;
};

}  // namespace tinytorch::nn

namespace tinygpt {

enum class GPTModelType : int8_t {
  UNKNOWN = 0,
  GPT2,
  LLAMA,
  QWEN2,
  QWEN3,
  MISTRAL,
};

class GPTModel {
 public:
  virtual ~GPTModel() = default;

  virtual GPTModelType type() { return GPTModelType::UNKNOWN; }

  tinytorch::Tensor forward(const tinytorch::Tensor &inputIds) { return model()(inputIds); }

  KVCacheManager &kvCache() { return kvCache_; }
  const KVCacheManager &kvCache() const { return kvCache_; }

  void resetCache() {
    kvCache_.reset();
    kvCache_.create(numLayers());
  }

  virtual bool load(const std::string &path) { return SafeTensors::load(model(), path, false); }
  virtual int64_t numLayers() = 0;
  virtual int64_t contextSize() = 0;
  virtual tinytorch::nn::Module &model() = 0;
  virtual tinytorch::Device device() const = 0;

 protected:
  void init() { kvCache_.create(numLayers()); }

  KVCacheManager kvCache_;
};

}  // namespace tinygpt
