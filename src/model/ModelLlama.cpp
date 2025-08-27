/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ModelLlama.h"

#include "Functions.h"
#include "Modules.h"
#include "SafeTensors.h"
#include "huggingface/ModelConfig.h"

namespace tt = tinytorch;

namespace tinygpt {

namespace llama {

using Config = huggingface::model::LlamaConfig;

static tt::RopeScalingConfig cvtRopeScaling(const Config &config) {
  return {config.ropeScaling.factor, config.ropeScaling.highFreqFactor, config.ropeScaling.lowFreqFactor,
          config.ropeScaling.originalMaxPositionEmbeddings};
}

static int64_t getContextSize(const Config &config) { return config.ropeScaling.originalMaxPositionEmbeddings; }

class FeedForward : public tt::nn::Module {
 public:
  explicit FeedForward(const Config &config, tt::Options options = {})
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
    auto xFc1 = gate_proj(input);
    auto xFc2 = up_proj(input);
    auto temp = tt::function::silu(xFc1) * xFc2;
    return down_proj(temp);
  }

  tt::nn::Linear gate_proj;
  tt::nn::Linear up_proj;
  tt::nn::Linear down_proj;
};

class GroupedQueryAttention : public tt::nn::Module {
 public:
  GroupedQueryAttention(KVCacheManager &kvCache, size_t layerIdx, int64_t dIn, int64_t dOut, int64_t numHeads,
                        int64_t numKvGroups, tt::Tensor &mask, tt::nn::RoPE &rope, tt::Options options = {})
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
        mask(mask),
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

    queries = rope.forward(queries, pastLength);
    keys = rope.forward(keys, pastLength);

    // update kv cache
    auto kv = kvCache_.append(layerIdx_, {keys, values});

    keys = tt::function::repeatInterleave(kv.first, groupSize_, 1);
    values = tt::function::repeatInterleave(kv.second, groupSize_, 1);

    tt::Tensor attnMask;
    if (pastLength == 0) {
      if (mask.dim() == 3) {
        attnMask = ~(tt::function::narrow(tt::function::narrow(mask, 1, 0, numTokens), 2, 0, numTokens));
      } else {
        attnMask = ~(tt::function::narrow(tt::function::narrow(mask, 0, 0, numTokens), 1, 0, numTokens));
      }
    } else {
      attnMask = {};
    }

    auto contextVec = tt::function::sdpAttention(queries, keys, values, false, attnMask);
    contextVec = contextVec.transpose(1, 2).reshape({batchSize, numTokens, dOut_});
    contextVec = o_proj(contextVec);
    return contextVec;
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

  tt::Tensor &mask;
  tt::nn::RoPE &rope;
};

class TransformerBlock : public tt::nn::Module {
 public:
  TransformerBlock(const Config &config, KVCacheManager &kvCache, size_t layerIdx, tt::Tensor &mask, tt::nn::RoPE &rope,
                   tt::Options options = {})
      : self_attn(GroupedQueryAttention(kvCache, layerIdx, config.hiddenSize, config.hiddenSize,
                                        config.numAttentionHeads, config.numKeyValueHeads, mask, rope, options)),
        mlp(FeedForward(config, options)),
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
    auto shortcut = input;
    auto temp = input_layernorm(input);
    temp = self_attn.forward(temp);
    temp += shortcut;

    shortcut = temp;
    temp = post_attention_layernorm(temp);
    temp = mlp(temp);
    temp += shortcut;
    return temp;
  }

  GroupedQueryAttention self_attn;
  FeedForward mlp;
  tt::nn::RMSNorm input_layernorm;
  tt::nn::RMSNorm post_attention_layernorm;
};

class LlamaModel : public tt::nn::Module {
 public:
  explicit LlamaModel(const Config &config, KVCacheManager &kvCache, tt::Options options = {})
      : embed_tokens(tt::nn::Embedding(config.vocabSize, config.hiddenSize, options)),
        layers(tt::nn::ModuleList()),
        norm(tt::nn::RMSNorm({config.hiddenSize}, config.rmsNormEps, options)),
        mask(initMask(config, options)),
        rope(config.hiddenSize / config.numAttentionHeads, getContextSize(config), config.ropeTheta,
             cvtRopeScaling(config), options) {
    for (auto i = 0; i < config.numHiddenLayers; i++) {
      layers.emplaceBack<TransformerBlock>(config, kvCache, i, mask, rope, options);
    }
    registerModules({
        {"embed_tokens", embed_tokens},
        {"layers", layers},
        {"norm", norm},
    });
  }

  static tt::Tensor initMask(const Config &config, const tt::Options &options) {
    auto contextSize = getContextSize(config);
    return tt::Tensor::ones({contextSize, contextSize}, tt::Options(options.device_, tt::DType::Bool)).triu(1);
  }

  tt::Tensor forward(const tt::Tensor &inputIds) override {
    mask = mask.to(inputIds.device());
    rope.to(inputIds.device());

    auto tokenEmbeds = embed_tokens(inputIds);
    auto temp = tokenEmbeds;

    for (auto &layer : layers) {
      temp = layer->forward(temp);
    }
    temp = norm(temp);
    return temp;
  }

  tt::nn::Embedding embed_tokens;
  tt::nn::ModuleList layers;
  tt::nn::RMSNorm norm;

  tt::Tensor mask;
  tt::nn::RoPE rope;
};

class LlamaHeadModel : public tt::nn::Module {
 public:
  LlamaHeadModel(const Config &config, KVCacheManager &kvCache, tt::Options options = {})
      : model(LlamaModel(config, kvCache, options)),
        out_head(tt::nn::Linear(config.hiddenSize, config.vocabSize, false, options)) {
    // shared weights
    out_head.weight() = model.embed_tokens.weight();
    registerModules({
        {"model", model},
        {"out_head", out_head},
    });
  }

  tinytorch::Tensor forward(const tt::Tensor &inputIds) override {
    auto hiddenStates = model.forward(inputIds);
    auto logits = out_head(hiddenStates);
    return logits;
  }

  LlamaModel model;
  tt::nn::Linear out_head;
};

}  // namespace llama

LlamaForCausalLM::LlamaForCausalLM(const huggingface::model::LlamaConfig &config, tt::Device device)
    : config_(config),
      model_(std::make_unique<llama::LlamaHeadModel>(config_, kvCache_, tt::Options(device, config.torchDtype))) {
  init();
}

LlamaForCausalLM::~LlamaForCausalLM() = default;

bool LlamaForCausalLM::load(const std::string &path) { return SafeTensors::load(*model_, path, false); }

int64_t LlamaForCausalLM::numLayers() { return config_.numHiddenLayers; }

tt::nn::Module &LlamaForCausalLM::model() { return *model_; }

}  // namespace tinygpt
