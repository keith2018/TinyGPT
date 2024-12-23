/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Model.h"
#include "FileUtils.h"
#include "Timer.h"

#include <cmath>

#define MATH_PI 3.1415926535f

namespace TinyGPT {

Tensor Model::gelu(const Tensor &x) {
  return 0.5f * x * (1.f + Tensor::tanh(std::sqrt(2.f / MATH_PI) * (x + 0.044715f * x * x * x)));
}

Tensor Model::softmax(const Tensor &x) {
  Tensor expX = Tensor::exp(x - Tensor::max(x, -1, true));
  return expX / Tensor::sum(expX, -1, true);
}

Tensor Model::layerNorm(const Tensor &x, const Tensor &g, const Tensor &b, float eps) {
  auto mean = Tensor::mean(x, -1, true);
  auto variance = Tensor::var(x, -1, true);

  // normalize x to have mean=0 and var=1 over last axis
  auto norm = (x - mean) / Tensor::sqrt(variance + eps);

  // scale and offset with gamma/beta params
  return g * norm + b;
}

Tensor Model::linear(const Tensor &x, const Tensor &w, const Tensor &b) {
  return Tensor::matmul(x, w) + b;
}

Tensor Model::feadForward(const Tensor &x, const Conv1D &fc, const Conv1D &proj) {
  // project up
  auto a = gelu(linear(x, fc.w, fc.b));

  // project back down
  return linear(a, proj.w, proj.b);
}

Tensor Model::attention(const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &mask) {
  auto x = Tensor::matmulTrans(q, k);
  x /= std::sqrt((float) q.shape().back());
  x += mask;
  return Tensor::matmul(softmax(x), v);
}

Tensor Model::multiHeadAttention(const Tensor &x, const Conv1D &attn, const Conv1D &proj, uint32_t head, KVCache &cache) {
  // qkv projection
  auto xx = linear(x, attn.w, attn.b);

  // split into qkv
  auto qkv = Tensor::split(xx, 3, -1);

  bool useCache = !cache.empty();
  if (useCache) {
    std::vector<Tensor> kStack = {cache[0], qkv[1]};
    std::vector<Tensor> vStack = {cache[1], qkv[2]};
    qkv[1] = Tensor::vstack({begin(kStack), end(kStack)});
    qkv[2] = Tensor::vstack({begin(vStack), end(vStack)});
  }

  cache = {qkv[1], qkv[2]};

  // split into heads
  std::vector<std::vector<Tensor>> qkvHeads;
  qkvHeads.reserve(qkv.size());
  for (auto &elem : qkv) {
    qkvHeads.emplace_back(elem.split((uint32_t) head, -1));
  }

  // causal mask to hide future inputs from being attended to
  Tensor causalMask;
  if (useCache) {
    causalMask = Tensor::zeros({1, qkv[1].shape()[0]});
  } else {
    causalMask = (1 - Tensor::tri(xx.shape()[0])) * -1e10;
  }

  // perform attention over each head
  std::vector<Tensor> outHeads;
  outHeads.reserve(qkvHeads.size());
  for (uint32_t i = 0; i < head; i++) {
    outHeads.emplace_back(attention(qkvHeads[0][i], qkvHeads[1][i], qkvHeads[2][i], causalMask));
  }

  // merge heads
  xx = Tensor::hstack({begin(outHeads), end(outHeads)});

  // out projection
  xx = linear(xx, proj.w, proj.b);

  return xx;
}

Tensor Model::transformerBlock(const Tensor &x, const TransformerBlock &block, uint32_t head, KVCache &cache) {
  // multi-head causal self attention
  auto norm = layerNorm(x, block.ln_1.g, block.ln_1.b);
  auto xx = x + multiHeadAttention(norm, block.attn.c_attn, block.attn.c_proj, head, cache);

  // position-wise feed forward network
  norm = layerNorm(xx, block.ln_2.g, block.ln_2.b);
  xx += feadForward(norm, block.mlp.c_fc, block.mlp.c_proj);
  return xx;
}

Tensor Model::gpt2(const std::vector<int32_t> &inputs, const GPT2::Params &params, uint32_t head, std::vector<KVCache> &cache) {
  bool useCache = !cache.empty();

  // token + positional embeddings
  Tensor x;
  if (useCache) {
    std::vector<int32_t> wteIdx = {inputs.back()};
    std::vector<int32_t> wpeIdx = {(int32_t) inputs.size() - 1};
    x = params.wte.index(wteIdx) + params.wpe.index(wpeIdx);
  } else {
    cache.resize(params.blocks.size());
    x = params.wte.index(inputs) + params.wpe.index(Tensor::range(0, (int32_t) inputs.size()));
  }

  // forward pass through n_layer transformer blocks
  for (int i = 0; i < params.blocks.size(); i++) {
    x = transformerBlock(x, params.blocks[i], head, cache[i]);
  }

  // projection to vocab
  x = layerNorm(x, params.ln_f.g, params.ln_f.b);
  return Tensor::matmulTrans(x, params.wte);
}

void Model::generate(std::vector<int32_t> &tokens, const GPT2::Params &params, uint32_t head,
                     uint32_t maxTokens, const std::function<bool(int32_t token)> &callback) {
  std::vector<KVCache> cache;
  // auto-regressive decode loop
  for (uint32_t tokenCnt = 0; tokenCnt < maxTokens; tokenCnt++) {
    // model forward pass
    auto logits = gpt2(tokens, params, head, cache);
    // greedy sampling
    auto nextId = (int32_t) Tensor::argmax(logits.index(std::vector<int32_t>{-1}));
    // append prediction to input
    tokens.emplace_back(nextId);

    if (callback) {
      bool stop = callback(nextId);
      if (stop) {
        break;
      }
    }
  }
}

bool Model::loadModelGPT2(GPT2 &gpt2, const char *hparams, const char *modelDict) {
  FUNCTION_TIMED();
  // load hparams.json
  const auto hparamsJson = FileUtils::parseJson(hparams);
  if (hparamsJson.is_null()) {
    LOGE("parse file failed: %s", hparams);
    return false;
  }
  gpt2.hparams.n_vocab = hparamsJson["n_vocab"].int_value();
  gpt2.hparams.n_ctx = hparamsJson["n_ctx"].int_value();
  gpt2.hparams.n_embd = hparamsJson["n_embd"].int_value();
  gpt2.hparams.n_head = hparamsJson["n_head"].int_value();
  gpt2.hparams.n_layer = hparamsJson["n_layer"].int_value();

  // load model_dict.json
  const auto modelDictJson = FileUtils::parseJson(modelDict);
  if (modelDictJson.is_null()) {
    LOGE("parse file failed: %s", modelDict);
    return false;
  }

  // load model_file.data
  auto dataPath = modelDictJson["file_path"].string_value();
  std::fstream dataFile(dataPath, std::ios::in | std::ios::binary);
  if (!dataFile.is_open()) {
    LOGE("open file failed: %s", dataPath.c_str());
    return {};
  }

  // check file size
  dataFile.seekg(0, std::ios::end);
  auto fileSize = (uint32_t) dataFile.tellg();
  auto expectSize = modelDictJson["file_size"].int_value();
  if (fileSize != expectSize) {
    LOGE("model file size invalid! expect: %lld, actual: %lld", expectSize, fileSize);
    return {};
  }
  dataFile.seekg(0, std::ios::beg);

  auto &model = modelDictJson["model_index"];

  loadTensor(gpt2.params.wpe, dataFile, model["wpe"]);
  loadTensor(gpt2.params.wte, dataFile, model["wte"]);
  loadLayerNorm(gpt2.params.ln_f, dataFile, model["ln_f"]);
  gpt2.params.blocks.resize(gpt2.hparams.n_layer);
  for (uint32_t i = 0; i < gpt2.hparams.n_layer; i++) {
    loadTransformerBlock(gpt2.params.blocks[i], dataFile, model["blocks"][i]);
  }

  return true;
}

void Model::loadTensor(Tensor &ret, std::fstream &fin, const json11::Json &json) {
  ret = Tensor::shape(getShape(json));
  uint32_t pos = json["pos"].int_value();
  fin.seekg(pos, std::ios::beg);
  fin.read((char *) &ret[0], std::streamsize(sizeof(float) * ret.size()));
}

void Model::loadConv1D(Conv1D &ret, std::fstream &fin, const json11::Json &json) {
  loadTensor(ret.w, fin, json["w"]);
  loadTensor(ret.b, fin, json["b"]);
}

void Model::loadLayerNorm(LayerNorm &ret, std::fstream &fin, const json11::Json &json) {
  loadTensor(ret.g, fin, json["g"]);
  loadTensor(ret.b, fin, json["b"]);
}

void Model::loadTransformerBlock(TransformerBlock &ret, std::fstream &fin, const json11::Json &json) {
  loadConv1D(ret.attn.c_attn, fin, json["attn"]["c_attn"]);
  loadConv1D(ret.attn.c_proj, fin, json["attn"]["c_proj"]);
  loadLayerNorm(ret.ln_1, fin, json["ln_1"]);
  loadLayerNorm(ret.ln_2, fin, json["ln_2"]);
  loadConv1D(ret.mlp.c_fc, fin, json["mlp"]["c_fc"]);
  loadConv1D(ret.mlp.c_proj, fin, json["mlp"]["c_proj"]);
}

Shape Model::getShape(const json11::Json &json) {
  auto &items = json["shape"].array_items();
  Shape ret;
  ret.reserve(items.size());
  for (auto &it : items) {
    ret.push_back(it.int_value());
  }
  return ret;
}

}
