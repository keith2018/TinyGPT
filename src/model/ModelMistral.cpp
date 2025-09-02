/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ModelMistral.h"

#include "Functions.h"
#include "Modules.h"
#include "SafeTensors.h"
#include "huggingface/ModelConfig.h"

namespace tt = tinytorch;

namespace tinygpt {

namespace mistral {

using Config = huggingface::model::MistralConfig;

class MistralMLP : public tt::nn::Module {
 public:
  explicit MistralMLP(const Config &config, tt::Options options = {})
      : gate_proj(tt::nn::Linear(config.hiddenSize, config.intermediateSize, false, options)),
        up_proj(tt::nn::Linear(config.hiddenSize, config.intermediateSize, false, options)),
        down_proj(tt::nn::Linear(config.intermediateSize, config.hiddenSize, false, options)) {
    registerModules({
        {"gate_proj", gate_proj},
        {"up_proj", up_proj},
        {"down_proj", down_proj},
    });
  }

  tt::Tensor forward(const tt::Tensor &input) override {
    auto x1 = gate_proj(input);
    auto x2 = up_proj(input);
    auto x = tt::function::silu(x1) * x2;
    return down_proj(x);
  }

  tt::nn::Linear gate_proj;
  tt::nn::Linear up_proj;
  tt::nn::Linear down_proj;
};

class MistralAttention : public tt::nn::Module {
 public:
  MistralAttention(KVCacheManager &kvCache, size_t layerIdx, int64_t dIn, int64_t dOut, int64_t numHeads,
                   int64_t numKvGroups, tt::nn::RoPE &rope, tt::Options options = {})
      : kvCache_(kvCache),
        layerIdx_(layerIdx),
        dOut_(dOut),
        numHeads_(numHeads),
        headDim_(dOut / numHeads),
        numKvGroups_(numKvGroups),
        groupSize_(numHeads / numKvGroups),
        k_proj(tt::nn::Linear(dIn, numKvGroups * headDim_, false, options)),
        v_proj(tt::nn::Linear(dIn, numKvGroups * headDim_, false, options)),
        q_proj(tt::nn::Linear(dIn, dOut, false, options)),
        o_proj(tt::nn::Linear(dOut, dOut, false, options)),
        rope(rope) {
    ASSERT(dOut % numHeads == 0);
    ASSERT(numHeads % numKvGroups == 0);

    registerModules({
        {"k_proj", k_proj},
        {"v_proj", v_proj},
        {"q_proj", q_proj},
        {"o_proj", o_proj},
    });
  }

  tt::Tensor forward(const tt::Tensor &input) override {
    auto batchSize = input.shape(0);
    auto numTokens = input.shape(1);

    auto queries = q_proj(input);
    auto keys = k_proj(input);
    auto values = v_proj(input);

    queries = queries.view({batchSize, numTokens, numHeads_, headDim_}).transpose(1, 2);
    keys = keys.view({batchSize, numTokens, numKvGroups_, headDim_}).transpose(1, 2);
    values = values.view({batchSize, numTokens, numKvGroups_, headDim_}).transpose(1, 2);

    int64_t pastLength = kvCache_.pastLength(layerIdx_);

    queries = rope(queries, pastLength);
    keys = rope(keys, pastLength);

    // update kv cache
    auto kvStates = kvCache_.append(layerIdx_, {keys, values});

    keys = tt::function::repeatInterleave(kvStates.kv.first, groupSize_, 1);
    values = tt::function::repeatInterleave(kvStates.kv.second, groupSize_, 1);

    bool isCausal = (kvStates.pastLength == 0);
    auto attnOutput = tt::function::sdpAttention(queries, keys, values, isCausal);

    attnOutput = attnOutput.transpose(1, 2).reshape({batchSize, numTokens, dOut_});
    attnOutput = o_proj(attnOutput);
    return attnOutput;
  }

  KVCacheManager &kvCache_;
  size_t layerIdx_;
  int64_t dOut_;
  int64_t numHeads_;
  int64_t headDim_;
  int64_t numKvGroups_;
  int64_t groupSize_;

  tt::nn::Linear k_proj;
  tt::nn::Linear v_proj;
  tt::nn::Linear q_proj;
  tt::nn::Linear o_proj;

  tt::nn::RoPE &rope;
};

class MistralDecoderLayer : public tt::nn::Module {
 public:
  MistralDecoderLayer(const Config &config, KVCacheManager &kvCache, size_t layerIdx, tt::nn::RoPE &rope,
                      tt::Options options = {})
      : self_attn(MistralAttention(kvCache, layerIdx, config.hiddenSize, config.hiddenSize, config.numAttentionHeads,
                                   config.numKeyValueHeads, rope, options)),
        mlp(MistralMLP(config, options)),
        input_layernorm(tt::nn::RMSNorm({config.hiddenSize}, config.rmsNormEps, options)),
        post_attention_layernorm(tt::nn::RMSNorm({config.hiddenSize}, config.rmsNormEps, options)) {
    registerModules({
        {"self_attn", self_attn},
        {"mlp", mlp},
        {"input_layernorm", input_layernorm},
        {"post_attention_layernorm", post_attention_layernorm},
    });
  }

  tt::Tensor forward(const tt::Tensor &input) override {
    auto x = input;
    x = x + self_attn(input_layernorm(x));
    x = x + mlp(post_attention_layernorm(x));
    return x;
  }

  MistralAttention self_attn;
  MistralMLP mlp;
  tt::nn::RMSNorm input_layernorm;
  tt::nn::RMSNorm post_attention_layernorm;
};

class MistralModel : public tt::nn::Module {
 public:
  explicit MistralModel(const Config &config, KVCacheManager &kvCache, tt::Options options = {})
      : embed_tokens(tt::nn::Embedding(config.vocabSize, config.hiddenSize, options)),
        layers(tt::nn::ModuleList()),
        norm(tt::nn::RMSNorm({config.hiddenSize}, config.rmsNormEps, options)),
        rope(config.hiddenSize / config.numAttentionHeads, config.maxPositionEmbeddings, config.ropeTheta, std::nullopt,
             options) {
    for (auto i = 0; i < config.numHiddenLayers; i++) {
      layers.emplaceBack<MistralDecoderLayer>(config, kvCache, i, rope, options);
    }
    registerModules({
        {"embed_tokens", embed_tokens},
        {"layers", layers},
        {"norm", norm},
    });
  }

  tt::Tensor forward(const tt::Tensor &inputIds) override {
    rope.to(inputIds.device());

    auto x = embed_tokens(inputIds);
    for (auto &layer : layers) {
      x = layer->forward(x);
    }
    x = norm(x);
    return x;
  }

  tt::nn::Embedding embed_tokens;
  tt::nn::ModuleList layers;
  tt::nn::RMSNorm norm;

  tt::nn::RoPE rope;
};

class MistralForCausalLM : public tt::nn::Module {
 public:
  MistralForCausalLM(const Config &config, KVCacheManager &kvCache, tt::Options options = {})
      : model(MistralModel(config, kvCache, options)),
        lm_head(tt::nn::Linear(config.hiddenSize, config.vocabSize, false, options)) {
    if (config.tieWordEmbeddings) {
      // shared weights
      lm_head.weight() = model.embed_tokens.weight();
    }
    registerModules({
        {"model", model},
        {"lm_head", lm_head},
    });
  }

  tinytorch::Tensor forward(const tt::Tensor &inputIds) override {
    auto x = model(inputIds);
    auto logits = lm_head(x);
    return logits;
  }

  MistralModel model;
  tt::nn::Linear lm_head;
};

}  // namespace mistral

ModelMistral::ModelMistral(const huggingface::model::MistralConfig &config, tt::Device device)
    : config_(config),
      model_(std::make_unique<mistral::MistralForCausalLM>(config_, kvCache_, tt::Options(device, config.torchDtype))) {
  init();
}

ModelMistral::~ModelMistral() = default;

bool ModelMistral::load(const std::string &path) { return SafeTensors::load(*model_, path, false); }

int64_t ModelMistral::numLayers() { return config_.numHiddenLayers; }

tt::nn::Module &ModelMistral::model() { return *model_; }

}  // namespace tinygpt
