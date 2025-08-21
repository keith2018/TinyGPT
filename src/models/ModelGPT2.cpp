/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ModelGPT2.h"

#include "Functions.h"
#include "Modules.h"
#include "SafeTensors.h"

namespace tt = tinytorch;

namespace tinygpt {
namespace gpt2 {

struct GPT2Config {
  int64_t vocab_size = 50257;
  int64_t n_positions = 1024;
  int64_t n_embd = 768;
  int64_t n_layer = 12;
  int64_t n_head = 12;
};

class Conv1D : public tt::nn::Module {
 public:
  Conv1D(int64_t outFeatures, int64_t inFeatures, tt::Options options = {}) {
    weight_ = tt::Tensor::empty({inFeatures, outFeatures}, options);
    bias_ = tt::Tensor::empty({outFeatures}, options);
  }

  tt::Tensor forward(const tt::Tensor &x) override {
    tt::SizeVector size_out(x.shape());
    size_out.back() = bias_.size(0);
    auto xx = x.view({-1, x.size(-1)}).matmul(weight_) + bias_;
    xx = xx.view(size_out.view());
    return xx;
  }

  std::vector<std::pair<std::string, tt::TensorPtr>> namedParameters_() override {
    return {{"weight", &weight_}, {"bias", &bias_}};
  }

  tt::Tensor weight_;
  tt::Tensor bias_;
};

class GPT2MHA : public tt::nn::Module {
 public:
  explicit GPT2MHA(const GPT2Config &config, tt::Options options = {})
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

  tt::Tensor forward(const tt::Tensor &x) override {
    auto B = x.size(0), T = x.size(1), C = x.size(2);
    auto qkv = tt::function::split(c_attn(x), C, 2);
    ASSERT(qkv.size() == 3);
    auto &q = qkv[0], &k = qkv[1], &v = qkv[2];
    q = q.view({B, T, n_head, head_dim}).transpose(1, 2);
    k = k.view({B, T, n_head, head_dim}).transpose(1, 2);
    v = v.view({B, T, n_head, head_dim}).transpose(1, 2);
    auto xx = tt::function::sdpAttention(q, k, v, true);
    xx = xx.transpose(1, 2).view({B, T, C});
    xx = c_proj(xx);
    return xx;
  }

  int64_t n_head;
  int64_t head_dim;
  Conv1D c_attn;
  Conv1D c_proj;
};

class GPT2MLP : public tt::nn::Module {
 public:
  explicit GPT2MLP(const GPT2Config &config, tt::Options options = {})
      : c_fc(Conv1D(4 * config.n_embd, config.n_embd, options)),
        c_proj(Conv1D(config.n_embd, 4 * config.n_embd, options)) {
    registerModules({
        {"c_fc", c_fc},
        {"c_proj", c_proj},
        {"act", act},
    });
  }

  tt::Tensor forward(const tt::Tensor &x) override { return c_proj(act(c_fc(x))); }

  Conv1D c_fc;
  Conv1D c_proj;
  tt::nn::Gelu act;
};

class GPT2Block : public tt::nn::Module {
 public:
  explicit GPT2Block(const GPT2Config &config, tt::Options options = {})
      : ln_1(tt::nn::LayerNorm({config.n_embd}, 1e-5, true, options)),
        attn(GPT2MHA(config, options)),
        ln_2(tt::nn::LayerNorm({config.n_embd}, 1e-5, true, options)),
        mlp(GPT2MLP(config, options)) {
    registerModules({
        {"ln_1", ln_1},
        {"attn", attn},
        {"ln_2", ln_2},
        {"mlp", mlp},
    });
  }

  tt::Tensor forward(const tt::Tensor &x) override {
    auto xx = x + attn(ln_1(x));
    xx = xx + mlp(ln_2(xx));
    return xx;
  }

  tt::nn::LayerNorm ln_1;
  GPT2MHA attn;
  tt::nn::LayerNorm ln_2;
  GPT2MLP mlp;
};

class GPT2Model : public tt::nn::Module {
 public:
  explicit GPT2Model(const GPT2Config &config, tt::Options options = {})
      : wte(tt::nn::Embedding(config.vocab_size, config.n_embd, options)),
        wpe(tt::nn::Embedding(config.n_positions, config.n_embd, options)),
        drop(tt::nn::Dropout(0.1f)),
        h(tt::nn::ModuleList()),
        ln_f(tt::nn::LayerNorm({config.n_embd}, 1e-5, true, options)) {
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

  tt::Tensor forward(const tt::Tensor &input_ids) override {
    auto T = input_ids.size(1);
    auto pos = tt::Tensor::arange<int64_t>(0, T, 1, tt::options::device(input_ids.device())).unsqueeze(0);
    auto x = wte(input_ids) + wpe(pos);
    x = drop(x);
    for (auto &block : h) {
      x = block->forward(x);
    }
    x = ln_f(x);
    return x;
  }

  tt::nn::Embedding wte;
  tt::nn::Embedding wpe;
  tt::nn::Dropout drop;
  tt::nn::ModuleList h;
  tt::nn::LayerNorm ln_f;
};

class GPT2LMHeadModel : public tt::nn::Module {
 public:
  explicit GPT2LMHeadModel(const GPT2Config &config, tt::Options options = {})
      : transformer(GPT2Model(config, options)),
        out_head(tt::nn::Linear(config.n_embd, config.vocab_size, false, options)) {
    // share weights
    out_head.weight() = transformer.wte.weight();
    registerModules({
        {"transformer", transformer},
        {"out_head", out_head},
    });
  }

  tt::Tensor forward(const tt::Tensor &input_ids) override {
    auto x = transformer(input_ids);
    auto logits = out_head(x);
    return logits;
  }

  GPT2Model transformer;
  tt::nn::Linear out_head;
};

}  // namespace gpt2

ModelGPT2::ModelGPT2(tt::Device device)
    : config_(std::make_unique<gpt2::GPT2Config>()),
      model_(std::make_unique<gpt2::GPT2LMHeadModel>(*config_, tt::Options(device, tt::DType::Float32))) {}

ModelGPT2::~ModelGPT2() = default;

bool ModelGPT2::load(const std::string &path) { return SafeTensors::load(model_->transformer, path, false); }

int64_t ModelGPT2::contextSize() { return config_->n_positions; }

tt::nn::Module &ModelGPT2::model() { return *model_; }

}  // namespace tinygpt
