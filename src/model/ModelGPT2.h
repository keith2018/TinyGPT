/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Functions.h"
#include "GPTModel.h"
#include "Modules.h"
#include "huggingface/ModelConfig.h"
#include "util/SafeTensors.h"

namespace tinygpt {

namespace gpt2 {

namespace tt = tinytorch;

using Config = huggingface::model::GPT2Config;

class Conv1D : public tt::nn::Module {
 public:
  Conv1D(int64_t outFeatures, int64_t inFeatures, tt::Options options = {}) {
    weight = tt::Tensor::empty({inFeatures, outFeatures}, options);
    bias = tt::Tensor::empty({outFeatures}, options);
  }

  tt::Tensor forward(const tt::Tensor &input) override {
    tt::SizeVector outputSize(input.shape());
    outputSize.back() = bias.size(0);
    auto linearOutput = input.view({-1, input.size(-1)}).matmul(weight) + bias;
    linearOutput = linearOutput.view(outputSize);
    return linearOutput;
  }

  std::vector<std::pair<std::string, tt::TensorPtr>> namedParameters_() override {
    return {{"weight", &weight}, {"bias", &bias}};
  }

  tt::Tensor weight;
  tt::Tensor bias;
};

class GPT2Attention : public tt::nn::Module {
 public:
  static constexpr int64_t kNumQKVProjections = 3;  // Query, Key, Value

  GPT2Attention(const Config &config, KVCacheManager *kvCache, size_t layerIdx, tt::Options options = {})
      : kvCache(kvCache),
        layerIdx(layerIdx),
        numHeads(config.nHead),
        headDim(config.nEmbd / config.nHead),
        cAttn(Conv1D(kNumQKVProjections * config.nEmbd, config.nEmbd, options)),
        cProj(Conv1D(config.nEmbd, config.nEmbd, options)) {
    ASSERT(config.nEmbd % config.nHead == 0);
    registerModules({
        {"c_attn", cAttn},
        {"c_proj", cProj},
    });
  }

  tt::Tensor forward(const tt::Tensor &input) override {
    auto batchSize = input.size(0), seqLen = input.size(1), channels = input.size(2);
    auto qkv = tt::function::split(cAttn(input), channels, 2);
    auto query = qkv[0];
    auto key = qkv[1];
    auto value = qkv[2];

    query = query.view({batchSize, seqLen, numHeads, headDim}).transpose(1, 2);
    key = key.view({batchSize, seqLen, numHeads, headDim}).transpose(1, 2);
    value = value.view({batchSize, seqLen, numHeads, headDim}).transpose(1, 2);

    // update kv cache
    auto kvStates = kvCache->append(layerIdx, {key, value});

    bool isCausal = (kvStates.pastLength == 0);
    auto attnOutput = tt::function::sdpAttention(query, kvStates.kv.first, kvStates.kv.second, isCausal);

    attnOutput = attnOutput.transpose(1, 2).view({batchSize, seqLen, channels});
    attnOutput = cProj(attnOutput);
    return attnOutput;
  }

  KVCacheManager *kvCache;
  size_t layerIdx;

  int64_t numHeads;
  int64_t headDim;

  Conv1D cAttn;
  Conv1D cProj;
};

class GPT2MLP : public tt::nn::Module {
 public:
  static constexpr int64_t kMLPExpansionRatio = 4;  // Standard GPT-2 MLP expansion ratio

  explicit GPT2MLP(const Config &config, tt::Options options = {})
      : cFc(Conv1D(kMLPExpansionRatio * config.nEmbd, config.nEmbd, options)),
        cProj(Conv1D(config.nEmbd, kMLPExpansionRatio * config.nEmbd, options)) {
    registerModules({
        {"c_fc", cFc},
        {"c_proj", cProj},
        {"act", act},
    });
  }

  tt::Tensor forward(const tt::Tensor &input) override { return cProj(act(cFc(input))); }

  Conv1D cFc;
  Conv1D cProj;
  tt::nn::Gelu act;
};

class GPT2Block : public tt::nn::Module {
 public:
  GPT2Block(const Config &config, KVCacheManager *kvCache, size_t layerIndex, tt::Options options = {})
      : ln1(tt::nn::LayerNorm({config.nEmbd}, config.layerNormEpsilon, true, options)),
        attn(GPT2Attention(config, kvCache, layerIndex, options)),
        ln2(tt::nn::LayerNorm({config.nEmbd}, config.layerNormEpsilon, true, options)),
        mlp(GPT2MLP(config, options)) {
    registerModules({
        {"ln_1", ln1},
        {"attn", attn},
        {"ln_2", ln2},
        {"mlp", mlp},
    });
  }

  tt::Tensor forward(const tt::Tensor &input) override {
    auto x = input;
    x = x + attn(ln1(x));
    x = x + mlp(ln2(x));
    return x;
  }

  tt::nn::LayerNorm ln1;
  GPT2Attention attn;
  tt::nn::LayerNorm ln2;
  GPT2MLP mlp;
};

class GPT2Model : public tt::nn::Module {
 public:
  GPT2Model(const Config &config, KVCacheManager *kvCache, tt::Options options = {})
      : kvCache(kvCache),
        wte(tt::nn::Embedding(config.vocabSize, config.nEmbd, options)),
        wpe(tt::nn::Embedding(config.nPositions, config.nEmbd, options)),
        h(tt::nn::ModuleList()),
        lnF(tt::nn::LayerNorm({config.nEmbd}, config.layerNormEpsilon, true, options)) {
    for (auto i = 0; i < config.nLayer; i++) {
      h.emplaceBack<GPT2Block>(config, kvCache, i, options);
    }
    registerModules({
        {"wte", wte},
        {"wpe", wpe},
        {"h", h},
        {"ln_f", lnF},
    });
  }

  tt::Tensor forward(const tt::Tensor &inputIds) override {
    auto seqLen = inputIds.size(1);
    int64_t pastLength = kvCache->pastLength(0);
    auto pos = tt::Tensor::arange<int64_t>(pastLength, pastLength + seqLen, 1, inputIds.options()).unsqueeze(0);

    auto x = wte(inputIds) + wpe(pos);
    for (auto &layer : h) {
      x = layer->forward(x);
    }
    x = lnF(x);
    return x;
  }

  KVCacheManager *kvCache;

  tt::nn::Embedding wte;
  tt::nn::Embedding wpe;
  tt::nn::ModuleList h;
  tt::nn::LayerNorm lnF;
};

class GPT2LMHeadModel : public tt::nn::Module {
 public:
  explicit GPT2LMHeadModel(const Config &config, KVCacheManager *kvCache, tt::Options options = {})
      : kvCache(kvCache),
        transformer(GPT2Model(config, kvCache, options)),
        lmHead(tt::nn::Linear(config.nEmbd, config.vocabSize, false, options)) {
    lmHead.weight() = transformer.wte.weight();
    registerModules({
        {"transformer", transformer},
        {"lm_head", lmHead},
    });
  }

  tinytorch::Tensor forward(const tt::Tensor &inputIds) override {
    auto x = transformer(inputIds);
    auto logits = lmHead(x);
    return logits;
  }

  KVCacheManager *kvCache;

  GPT2Model transformer;
  tt::nn::Linear lmHead;
};

}  // namespace gpt2

class ModelGPT2 : public GPTModel {
 public:
  ModelGPT2(const huggingface::model::GPT2Config &config, tinytorch::Device device)
      : config_(config),
        device_(device),
        model_(std::make_unique<gpt2::GPT2LMHeadModel>(config_, &kvCache_,
                                                       tinytorch::Options(device, config.torchDtype))) {
    init();
  }

  ~ModelGPT2() override = default;

  GPTModelType type() override { return GPTModelType::GPT2; }

  bool load(const std::string &path) override { return SafeTensors::load(model_->transformer, path, false); }

  int64_t numLayers() override { return config_.nLayer; }

  int64_t contextSize() override { return config_.nCtx; }

  tinytorch::nn::Module &model() override { return *model_; }

  tinytorch::Device device() const override { return device_; }

 private:
  const huggingface::model::GPT2Config &config_;
  tinytorch::Device device_;
  std::unique_ptr<gpt2::GPT2LMHeadModel> model_;
};

}  // namespace tinygpt
