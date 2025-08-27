/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ModelGPT2.h"

#include "Functions.h"
#include "Modules.h"
#include "SafeTensors.h"
#include "huggingface/ModelConfig.h"

namespace tt = tinytorch;

namespace tinygpt {

namespace gpt2 {

using Config = huggingface::model::GPT2Config;

class Conv1D : public tt::nn::Module {
 public:
  Conv1D(int64_t outFeatures, int64_t inFeatures, tt::Options options = {}) {
    weight_ = tt::Tensor::empty({inFeatures, outFeatures}, options);
    bias_ = tt::Tensor::empty({outFeatures}, options);
  }

  tt::Tensor forward(const tt::Tensor &input) override {
    tt::SizeVector outputSize(input.shape());
    outputSize.back() = bias_.size(0);
    auto temp = input.view({-1, input.size(-1)}).matmul(weight_) + bias_;
    temp = temp.view(outputSize);
    return temp;
  }

  std::vector<std::pair<std::string, tt::TensorPtr>> namedParameters_() override {
    return {{"weight", &weight_}, {"bias", &bias_}};
  }

  tt::Tensor weight_;
  tt::Tensor bias_;
};

class GPT2MHA : public tt::nn::Module {
 public:
  GPT2MHA(const Config &config, KVCacheManager &kvCache, size_t layerIndex, tt::Options options = {})
      : kvCache_(kvCache),
        layerIndex_(layerIndex),
        numHeads_(config.nHead),
        headDim_(config.nEmbd / config.nHead),
        c_attn(Conv1D(3 * config.nEmbd, config.nEmbd, options)),
        c_proj(Conv1D(config.nEmbd, config.nEmbd, options)) {
    ASSERT(config.nEmbd % config.nHead == 0);
    registerModules({
        {"c_attn", c_attn},
        {"c_proj", c_proj},
    });
  }

  tt::Tensor forward(const tt::Tensor &input) override {
    auto batchSize = input.size(0), seqLen = input.size(1), channels = input.size(2);
    auto qkv = tt::function::split(c_attn(input), channels, 2);
    auto query = qkv[0], key = qkv[1], value = qkv[2];

    query = query.view({batchSize, seqLen, numHeads_, headDim_}).transpose(1, 2);
    key = key.view({batchSize, seqLen, numHeads_, headDim_}).transpose(1, 2);
    value = value.view({batchSize, seqLen, numHeads_, headDim_}).transpose(1, 2);

    // update kv cache
    auto kv = kvCache_.append(layerIndex_, {key, value});

    bool isCausal = (kvCache_.pastLength(layerIndex_) == 0);
    auto attnOutput = tt::function::sdpAttention(query, kv.first, kv.second, isCausal);

    attnOutput = attnOutput.transpose(1, 2).view({batchSize, seqLen, channels});
    attnOutput = c_proj(attnOutput);
    return attnOutput;
  }

  KVCacheManager &kvCache_;
  size_t layerIndex_;

  int64_t numHeads_;
  int64_t headDim_;

  Conv1D c_attn;
  Conv1D c_proj;
};

class GPT2MLP : public tt::nn::Module {
 public:
  explicit GPT2MLP(const Config &config, tt::Options options = {})
      : c_fc(Conv1D(4 * config.nEmbd, config.nEmbd, options)), c_proj(Conv1D(config.nEmbd, 4 * config.nEmbd, options)) {
    registerModules({
        {"c_fc", c_fc},
        {"c_proj", c_proj},
        {"act", act},
    });
  }

  tt::Tensor forward(const tt::Tensor &input) override { return c_proj(act(c_fc(input))); }

  Conv1D c_fc;
  Conv1D c_proj;
  tt::nn::Gelu act;
};

class GPT2Block : public tt::nn::Module {
 public:
  GPT2Block(const Config &config, KVCacheManager &kvCache, size_t layerIndex, tt::Options options = {})
      : ln_1(tt::nn::LayerNorm({config.nEmbd}, config.layerNormEpsilon, true, options)),
        attn(GPT2MHA(config, kvCache, layerIndex, options)),
        ln_2(tt::nn::LayerNorm({config.nEmbd}, config.layerNormEpsilon, true, options)),
        mlp(GPT2MLP(config, options)) {
    registerModules({
        {"ln_1", ln_1},
        {"attn", attn},
        {"ln_2", ln_2},
        {"mlp", mlp},
    });
  }

  tt::Tensor forward(const tt::Tensor &input) override {
    auto attnOutput = attn.forward(ln_1(input));
    auto temp = input + attnOutput;
    temp = temp + mlp(ln_2(temp));
    return temp;
  }

  tt::nn::LayerNorm ln_1;
  GPT2MHA attn;
  tt::nn::LayerNorm ln_2;
  GPT2MLP mlp;
};

class GPT2Model : public tt::nn::Module {
 public:
  GPT2Model(const Config &config, KVCacheManager &kvCache, tt::Options options = {})
      : kvCache_(kvCache),
        wte(tt::nn::Embedding(config.vocabSize, config.nEmbd, options)),
        wpe(tt::nn::Embedding(config.nPositions, config.nEmbd, options)),
        h(tt::nn::ModuleList()),
        ln_f(tt::nn::LayerNorm({config.nEmbd}, config.layerNormEpsilon, true, options)) {
    for (auto i = 0; i < config.nLayer; i++) {
      h.emplaceBack<GPT2Block>(config, kvCache, i, options);
    }
    registerModules({
        {"wte", wte},
        {"wpe", wpe},
        {"h", h},
        {"ln_f", ln_f},
    });
  }

  tt::Tensor forward(const tt::Tensor &inputIds) override {
    auto seqLen = inputIds.size(1);
    int64_t pastLength = kvCache_.pastLength(0);

    auto pos = tt::Tensor::arange<int64_t>(pastLength, pastLength + seqLen, 1, inputIds.options()).unsqueeze(0);

    auto hiddenStates = wte(inputIds) + wpe(pos);

    for (auto &layer : h) {
      hiddenStates = layer->forward(hiddenStates);
    }

    hiddenStates = ln_f(hiddenStates);
    return hiddenStates;
  }

  KVCacheManager &kvCache_;

  tt::nn::Embedding wte;
  tt::nn::Embedding wpe;
  tt::nn::ModuleList h;
  tt::nn::LayerNorm ln_f;
};

class GPT2HeadModel : public tt::nn::Module {
 public:
  explicit GPT2HeadModel(const Config &config, KVCacheManager &kvCache, tt::Options options = {})
      : kvCache_(kvCache),
        transformer(GPT2Model(config, kvCache, options)),
        out_head(tt::nn::Linear(config.nEmbd, config.vocabSize, false, options)) {
    // shared weights
    out_head.weight() = transformer.wte.weight();
    registerModules({
        {"transformer", transformer},
        {"out_head", out_head},
    });
  }

  tinytorch::Tensor forward(const tt::Tensor &inputIds) override {
    auto hiddenStates = transformer.forward(inputIds);
    auto logits = out_head(hiddenStates);
    return logits;
  }

  KVCacheManager &kvCache_;

  GPT2Model transformer;
  tt::nn::Linear out_head;
};

}  // namespace gpt2

GPT2LMHeadModel::GPT2LMHeadModel(const gpt2::Config &config, tt::Device device)
    : config_(config),
      model_(std::make_unique<gpt2::GPT2HeadModel>(config_, kvCache_, tt::Options(device, config.torchDtype))) {
  init();
}

GPT2LMHeadModel::~GPT2LMHeadModel() = default;

bool GPT2LMHeadModel::load(const std::string &path) { return SafeTensors::load(model_->transformer, path, false); }

int64_t GPT2LMHeadModel::numLayers() { return config_.nLayer; }

tt::nn::Module &GPT2LMHeadModel::model() { return *model_; }

}  // namespace tinygpt
