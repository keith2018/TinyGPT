/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Functions.h"
#include "Initializer.h"
#include "Module.h"
#include "Sampler.h"

using namespace tinytorch;

namespace tinygpt::gpt2 {

struct GPT2Config {
  int64_t vocab_size = 50257, n_positions = 1024, n_embd = 768, n_layer = 12, n_head = 12;
};

class Conv1D : public nn::Module {
 public:
  Conv1D(int64_t outFeatures, int64_t inFeatures, Options options = {}) {
    weight_ = Tensor::empty({inFeatures, outFeatures}, options);
    bias_ = Tensor::empty({outFeatures}, options);
  }

  Tensor forward(const Tensor &x) override {
    SizeVector size_out(x.shape());
    size_out.back() = bias_.size(0);
    auto y = x.view({-1, x.size(-1)}).matmul(weight_) + bias_;
    y = y.view(size_out.view());
    return y;
  }

  void resetParameters() override {
    nn::Initializer::normal(weight_, 0.f, 0.02f);
    nn::Initializer::zeros(bias_);
  }

  std::vector<std::pair<std::string, TensorPtr>> namedParameters_() override {
    return {{"weight", &weight_}, {"bias", &bias_}};
  }

  Tensor weight_;
  Tensor bias_;
};

class GPT2MHA : public nn::Module {
 public:
  explicit GPT2MHA(const GPT2Config &config, Options options = {})
      : n_head(config.n_head),
        head_dim(config.n_embd / config.n_head),
        c_attn(Conv1D(3 * config.n_embd, config.n_embd, options)),
        c_proj(Conv1D(config.n_embd, config.n_embd, options)) {
    ASSERT(config.n_embd % config.n_head == 0);
    registerModules({
        {"c_attn", c_attn},
        {"c_proj", c_proj},
    });
  }

  Tensor forward(const Tensor &x) override {
    auto B = x.size(0), T = x.size(1), C = x.size(2);
    auto qkv = function::split(c_attn(x), C, 2);
    ASSERT(qkv.size() == 3);
    auto &q = qkv[0], &k = qkv[1], &v = qkv[2];
    q = q.view({B, T, n_head, head_dim}).transpose(1, 2);
    k = k.view({B, T, n_head, head_dim}).transpose(1, 2);
    v = v.view({B, T, n_head, head_dim}).transpose(1, 2);
    auto y = function::sdpAttention(q, k, v, true);
    y = y.transpose(1, 2).view({B, T, C});
    y = c_proj(y);
    return y;
  }

  int64_t n_head;
  int64_t head_dim;
  Conv1D c_attn;
  Conv1D c_proj;
};

class GPT2MLP : public nn::Module {
 public:
  explicit GPT2MLP(const GPT2Config &config, Options options = {})
      : c_fc(Conv1D(4 * config.n_embd, config.n_embd, options)),
        c_proj(Conv1D(config.n_embd, 4 * config.n_embd, options)) {
    registerModules({
        {"c_fc", c_fc},
        {"c_proj", c_proj},
        {"act", act},
    });
  }

  Tensor forward(const Tensor &x) override { return c_proj(act(c_fc(x))); }

  Conv1D c_fc;
  Conv1D c_proj;
  nn::Gelu act;
};

class GPT2Block : public nn::Module {
 public:
  explicit GPT2Block(const GPT2Config &config, Options options = {})
      : ln_1(nn::LayerNorm({config.n_embd}, 1e-5, true, options)),
        attn(GPT2MHA(config, options)),
        ln_2(nn::LayerNorm({config.n_embd}, 1e-5, true, options)),
        mlp(GPT2MLP(config, options)) {
    registerModules({
        {"ln_1", ln_1},
        {"attn", attn},
        {"ln_2", ln_2},
        {"mlp", mlp},
    });
  }

  Tensor forward(const Tensor &x) override {
    auto y = x + attn(ln_1(x));
    y = y + mlp(ln_2(y));
    return y;
  }

  nn::LayerNorm ln_1;
  GPT2MHA attn;
  nn::LayerNorm ln_2;
  GPT2MLP mlp;
};

class GPT2Model : public nn::Module {
 public:
  explicit GPT2Model(const GPT2Config &config, Options options = {})
      : wte(nn::Embedding(config.vocab_size, config.n_embd, options)),
        wpe(nn::Embedding(config.n_positions, config.n_embd, options)),
        drop(nn::Dropout(0.1f)),
        h(nn::ModuleList()),
        ln_f(nn::LayerNorm({config.n_embd}, 1e-5, true, options)) {
    for (auto i = 0; i < config.n_layer; i++) {
      h.emplaceBack<GPT2Block>(config, options);
    }

    registerModules({
        {"wte", wte},
        {"wpe", wpe},
        {"drop", drop},
        {"h", h},
        {"ln_f", ln_f},
    });
  }

  Tensor forward(const Tensor &input_ids) override {
    auto T = input_ids.size(1);
    auto pos = Tensor::arange<int64_t>(0, T, 1, options::device(input_ids.device())).unsqueeze(0);
    auto x = wte(input_ids) + wpe(pos);
    x = drop(x);
    for (auto &block : h) {
      x = block->forward(x);
    }
    x = ln_f(x);
    return x;
  }

  nn::Embedding wte;
  nn::Embedding wpe;
  nn::Dropout drop;
  nn::ModuleList h;
  nn::LayerNorm ln_f;
};

class GPT2LMHeadModel : public nn::Module {
 public:
  explicit GPT2LMHeadModel(const GPT2Config &config, Options options = {})
      : transformer(GPT2Model(config, options)), lm_head(nn::Linear(config.n_embd, config.vocab_size, false, options)) {
    // share weights
    lm_head.weight() = transformer.wte.weight();
    registerModules({
        {"transformer", transformer},
        {"lm_head", lm_head},
    });
  }

  Tensor forward(const Tensor &input_ids) override {
    auto x = transformer(input_ids);
    auto logits = lm_head(x);
    // TODO loss
    return logits;
  }

  GPT2Model transformer;
  nn::Linear lm_head;
};

Tensor generate(const GPT2Config &config, nn::Module &model, Sampler &sampler, Tensor input_ids,
                int64_t max_new_tokens = 20);

}  // namespace tinygpt::gpt2
