/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ModelLlama32.h"

#include "Functions.h"
#include "Modules.h"
#include "SafeTensors.h"

namespace tt = tinytorch;

namespace tinygpt {
namespace llama32 {

struct Llama32Config {
  int64_t vocab_size = 128256;
  int64_t context_length = 8192;
  int64_t orig_context_length = 131072;
  int64_t emb_dim = 2048;
  int64_t n_heads = 32;
  int64_t n_layers = 16;
  int64_t hidden_dim = 8192;
  int64_t n_kv_groups = 8;
  float rope_base = 500000.f;

  struct {
    float factor = 32.f;
    float highFreqFactor = 1.f;
    float lowFreqFactor = 4.f;
    int64_t originalContextLength = 8192;
  } rope_freq;

  Llama32Config(GPTModelSize size) {
    if (size == GPTModelSize::SIZE_3B) {
      emb_dim = 3072;
      n_heads = 24;
      n_layers = 28;
    }
    rescaleTheta();
  }

  void rescaleTheta() {
    if (orig_context_length != context_length) {
      float scaling_factor = static_cast<float>(context_length) / static_cast<float>(orig_context_length);
      rope_base = rope_base * scaling_factor;
    }
  }

  tt::RopeScalingConfig ropeScaling() const {
    return {rope_freq.factor, rope_freq.highFreqFactor, rope_freq.lowFreqFactor, rope_freq.originalContextLength};
  }
};

class FeedForward : public tt::nn::Module {
 public:
  explicit FeedForward(const Llama32Config &config, tt::Options options = {})
      : gate_proj(tt::nn::Linear(config.emb_dim, config.hidden_dim, false, options)),
        up_proj(tt::nn::Linear(config.emb_dim, config.hidden_dim, false, options)),
        down_proj(tt::nn::Linear(config.hidden_dim, config.emb_dim, false, options)) {
    registerModules({
        {"gate_proj", gate_proj},
        {"up_proj", up_proj},
        {"down_proj", down_proj},
    });
  }

  tt::Tensor forward(const tt::Tensor &x) override {
    auto x_fc1 = gate_proj(x);
    auto x_fc2 = up_proj(x);
    auto xx = tt::function::silu(x_fc1) * x_fc2;
    return down_proj(xx);
  }

  tt::nn::Linear gate_proj;
  tt::nn::Linear up_proj;
  tt::nn::Linear down_proj;
};

class GroupedQueryAttention : public tt::nn::Module {
 public:
  GroupedQueryAttention(int64_t d_in, int64_t d_out, int64_t num_heads, int64_t num_kv_groups, tt::Tensor &mask,
                        tt::nn::RoPE &rope, tt::Options options = {})
      : d_out(d_out),
        num_heads(num_heads),
        head_dim(d_out / num_heads),
        num_kv_groups(num_kv_groups),
        group_size(num_heads / num_kv_groups),
        k_proj(tt::nn::Linear(d_in, num_kv_groups * head_dim, false, options)),
        v_proj(tt::nn::Linear(d_in, num_kv_groups * head_dim, false, options)),
        q_proj(tt::nn::Linear(d_in, d_out, false, options)),
        o_proj(tt::nn::Linear(d_out, d_out, false, options)),
        mask(mask),
        rope(rope) {
    ASSERT(d_out % num_heads == 0);
    ASSERT(num_heads % num_kv_groups == 0);

    registerModules({
        {"k_proj", k_proj},
        {"v_proj", v_proj},
        {"q_proj", q_proj},
        {"o_proj", o_proj},
    });
  }

  tt::Tensor forward(const tt::Tensor &x) override {
    auto b = x.shape(0), num_tokens = x.shape(1);
    auto queries = q_proj(x);
    auto keys = k_proj(x);
    auto values = v_proj(x);

    queries = queries.view({b, num_tokens, num_heads, head_dim});
    keys = keys.view({b, num_tokens, num_kv_groups, head_dim});
    values = values.view({b, num_tokens, num_kv_groups, head_dim});

    queries = queries.transpose(1, 2);
    keys = keys.transpose(1, 2);
    values = values.transpose(1, 2);

    keys = rope(keys);
    queries = rope(queries);

    keys = tt::function::repeatInterleave(keys, group_size, 1);
    values = tt::function::repeatInterleave(values, group_size, 1);

    tt::Tensor attn_mask;
    if (mask.dim() == 3) {
      attn_mask = ~(tt::function::narrow(tt::function::narrow(mask, 1, 0, num_tokens), 2, 0, num_tokens));
    } else {
      attn_mask = ~(tt::function::narrow(tt::function::narrow(mask, 0, 0, num_tokens), 1, 0, num_tokens));
    }

    auto context_vec = tt::function::sdpAttention(queries, keys, values, false, attn_mask);
    context_vec = context_vec.transpose(1, 2).reshape({b, num_tokens, d_out});
    context_vec = o_proj(context_vec);
    return context_vec;
  }

  int64_t d_out;
  int64_t num_heads;
  int64_t head_dim;
  int64_t num_kv_groups;
  int64_t group_size;

  tt::nn::Linear k_proj;
  tt::nn::Linear v_proj;
  tt::nn::Linear q_proj;
  tt::nn::Linear o_proj;

  tt::Tensor &mask;
  tt::nn::RoPE &rope;
};

class TransformerBlock : public tt::nn::Module {
 public:
  TransformerBlock(const Llama32Config &config, tt::Tensor &mask, tt::nn::RoPE &rope, tt::Options options = {})
      : self_attn(GroupedQueryAttention(config.emb_dim, config.emb_dim, config.n_heads, config.n_kv_groups, mask, rope,
                                        options)),
        mlp(FeedForward(config, options)),
        input_layernorm(tt::nn::RMSNorm({config.emb_dim}, 1e-5, options)),
        post_attention_layernorm(tt::nn::RMSNorm({config.emb_dim}, 1e-5, options)) {
    registerModules({
        {"self_attn", self_attn},
        {"mlp", mlp},
        {"input_layernorm", input_layernorm},
        {"post_attention_layernorm", post_attention_layernorm},
    });
  }

  tt::Tensor forward(const tt::Tensor &x) override {
    auto shortcut = x;
    auto xx = input_layernorm(x);
    xx = self_attn(xx);
    xx += shortcut;

    shortcut = xx;
    xx = post_attention_layernorm(xx);
    xx = mlp(xx);
    xx += shortcut;
    return xx;
  }

  GroupedQueryAttention self_attn;
  FeedForward mlp;
  tt::nn::RMSNorm input_layernorm;
  tt::nn::RMSNorm post_attention_layernorm;
};

class Llama32Model : public tt::nn::Module {
 public:
  explicit Llama32Model(const Llama32Config &config, tt::Options options = {})
      : embed_tokens(tt::nn::Embedding(config.vocab_size, config.emb_dim, options)),
        layers(tt::nn::ModuleList()),
        norm(tt::nn::RMSNorm({config.emb_dim}, 1e-5, options)),
        mask(initMask(config, options)),
        rope(config.emb_dim / config.n_heads, config.context_length, config.rope_base, config.ropeScaling(), options) {
    for (auto i = 0; i < config.n_layers; i++) {
      layers.emplaceBack<TransformerBlock>(config, mask, rope, options);
    }
    registerModules({
        {"embed_tokens", embed_tokens},
        {"layers", layers},
        {"norm", norm},
    });
  }

  static tt::Tensor initMask(const Llama32Config &config, const tt::Options &options) {
    return tt::Tensor::ones({config.context_length, config.context_length},
                            tt::Options(options.device_, tt::DType::Bool))
        .triu(1);
  }

  tt::Tensor forward(const tt::Tensor &in_idx) override {
    mask = mask.to(in_idx.device());
    rope.to(in_idx.device());

    auto tok_embeds = embed_tokens(in_idx);
    auto x = tok_embeds;

    for (auto &layer : layers) {
      x = layer->forward(x);
    }
    x = norm(x);
    return x;
  }

  tt::nn::Embedding embed_tokens;
  tt::nn::ModuleList layers;
  tt::nn::RMSNorm norm;

  tt::Tensor mask;
  tt::nn::RoPE rope;
};

class Llama32LMHeadModel : public tt::nn::Module {
 public:
  explicit Llama32LMHeadModel(const Llama32Config &config, tt::Options options = {})
      : model(Llama32Model(config, options)),
        out_head(tt::nn::Linear(config.emb_dim, config.vocab_size, false, options)) {
    // share weights
    out_head.weight() = model.embed_tokens.weight();
    registerModules({
        {"model", model},
        {"out_head", out_head},
    });
  }

  tt::Tensor forward(const tt::Tensor &input_ids) override {
    auto x = model(input_ids);
    auto logits = out_head(x);
    return logits;
  }

  Llama32Model model;
  tt::nn::Linear out_head;
};

}  // namespace llama32

ModelLlama32::ModelLlama32(tt::Device device, GPTModelSize size)
    : size_(size),
      config_(std::make_unique<llama32::Llama32Config>(size)),
      model_(std::make_unique<llama32::Llama32LMHeadModel>(*config_, tt::Options(device, tt::DType::BFloat16))) {}

ModelLlama32::~ModelLlama32() = default;

bool ModelLlama32::load(const std::string &path) { return SafeTensors::load(*model_, path, false); }

int64_t ModelLlama32::contextSize() { return config_->context_length; }

tt::nn::Module &ModelLlama32::model() { return *model_; }

}  // namespace tinygpt
