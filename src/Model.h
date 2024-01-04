/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor.h"
#include "json11.hpp"

#include <functional>

namespace TinyGPT {

struct Conv1D {
  Tensor w;
  Tensor b;
};

struct LayerNorm {
  Tensor g;
  Tensor b;
};

struct TransformerBlock {
  struct {
    Conv1D c_attn;
    Conv1D c_proj;
  } attn;
  LayerNorm ln_1;
  LayerNorm ln_2;
  struct {
    Conv1D c_fc;
    Conv1D c_proj;
  } mlp;
};

struct GPT2 {
  struct HParams {
    uint32_t n_vocab;
    uint32_t n_ctx;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_layer;
  } hparams;

  struct Params {
    Tensor wpe;
    Tensor wte;
    LayerNorm ln_f;
    std::vector<TransformerBlock> blocks;
  } params;
};

typedef std::vector<Tensor> KVCache;

class Model {
 public:
  static Tensor gelu(const Tensor &x);
  static Tensor softmax(const Tensor &x);
  static Tensor layerNorm(const Tensor &x, const Tensor &g, const Tensor &b, float eps = 1e-5);
  static Tensor linear(const Tensor &x, const Tensor &w, const Tensor &b);

  static Tensor feadForward(const Tensor &x, const Conv1D &fc, const Conv1D &proj);
  static Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &mask);
  static Tensor multiHeadAttention(const Tensor &x, const Conv1D &attn, const Conv1D &proj, uint32_t head, KVCache &cache);
  static Tensor transformerBlock(const Tensor &x, const TransformerBlock &block, uint32_t head, KVCache &cache);

  static Tensor gpt2(const std::vector<int32_t> &inputs, const GPT2::Params &params, uint32_t head, std::vector<KVCache> &cache);

 public:
  static bool loadModelGPT2(GPT2 &gpt2, const char *hparams, const char *modelDict);
  static void generate(std::vector<int32_t> &tokens, const GPT2::Params &params, uint32_t head,
                       uint32_t maxTokens, const std::function<bool(int32_t token)> &callback);

 private:
  static void loadTensor(Tensor &ret, std::fstream &fin, const json11::Json &json);
  static void loadConv1D(Conv1D &ret, std::fstream &fin, const json11::Json &json);
  static void loadLayerNorm(LayerNorm &ret, std::fstream &fin, const json11::Json &json);
  static void loadTransformerBlock(TransformerBlock &ret, std::fstream &fin, const json11::Json &json);
  static Shape getShape(const json11::Json &json);
};

}
