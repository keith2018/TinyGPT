/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Model.h"

#include <cmath>

#include "FileUtils.h"
#include "Functions.h"
#include "Operations.h"
#include "Utils/Timer.h"

using namespace tinytorch;

namespace tinygpt {

Device Model::device_ = Device(DeviceType::CUDA, 0);

Tensor Model::gelu(const Tensor& x) { return op::gelu(x); }

Tensor Model::softmax(const Tensor& x) { return op::softmax(x, -1); }

Tensor Model::layerNorm(const Tensor& x, const Tensor& g, const Tensor& b, float eps) {
  return op::layerNorm(x, g, b, eps);
}

Tensor Model::linear(const Tensor& x, const Tensor& w, const Tensor& b) { return op::matmul(x, w) + b; }

Tensor Model::feadForward(const Tensor& x, const Conv1D& fc, const Conv1D& proj) {
  // project up
  auto a = gelu(linear(x, fc.w, fc.b));

  // project back down
  return linear(a, proj.w, proj.b);
}

Tensor Model::attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& mask) {
  auto x = op::matmulTrans(q, k, false, true);
  x /= std::sqrt((float)q.shape().back());
  x += mask;
  return op::matmul(softmax(x), v);
}

Tensor Model::multiHeadAttention(const Tensor& x, const Conv1D& attn, const Conv1D& proj, uint32_t head,
                                 KVCache& cache) {
  // qkv projection
  auto xx = linear(x, attn.w, attn.b);

  // split into qkv
  auto qkv = op::split(xx, xx.shape().back() / 3, -1);

  bool useCache = !cache.empty();
  if (useCache) {
    std::vector<Tensor> kStack = {cache[0], qkv[1]};
    std::vector<Tensor> vStack = {cache[1], qkv[2]};
    qkv[1] = op::vstack(ArrayView(kStack));
    qkv[2] = op::vstack(ArrayView(vStack));
  }

  cache = {qkv[1], qkv[2]};

  // split into heads
  std::vector<std::vector<Tensor>> qkvHeads;
  qkvHeads.reserve(qkv.size());
  for (auto& elem : qkv) {
    qkvHeads.emplace_back(op::split(elem, elem.shape().back() / static_cast<int32_t>(head), -1));
  }

  // causal mask to hide future inputs from being attended to
  Tensor causalMask;
  if (useCache) {
    causalMask = Tensor::zeros({1, qkv[1].shape()[0]}, options::device(device_));
  } else {
    auto& n = xx.shape()[0];
    causalMask = (1 - op::tril(Tensor::ones({n, n}, options::device(device_)), 0)) * -1e10f;
  }

  // perform attention over each head
  std::vector<Tensor> outHeads;
  outHeads.reserve(qkvHeads.size());
  for (uint32_t i = 0; i < head; i++) {
    outHeads.emplace_back(attention(qkvHeads[0][i], qkvHeads[1][i], qkvHeads[2][i], causalMask));
  }

  // merge heads
  xx = op::hstack(ArrayView(outHeads));

  // out projection
  xx = linear(xx, proj.w, proj.b);

  return xx;
}

Tensor Model::transformerBlock(const Tensor& x, const TransformerBlock& block, uint32_t head, KVCache& cache) {
  // multi-head causal self attention
  auto norm = layerNorm(x, block.ln_1.g, block.ln_1.b);
  auto xx = x + multiHeadAttention(norm, block.attn.c_attn, block.attn.c_proj, head, cache);

  // position-wise feed forward network
  norm = layerNorm(xx, block.ln_2.g, block.ln_2.b);
  xx += feadForward(norm, block.mlp.c_fc, block.mlp.c_proj);
  return xx;
}

Tensor Model::gpt2(const std::vector<int64_t>& inputs, const GPT2::Params& params, uint32_t head,
                   std::vector<KVCache>& cache) {
  bool useCache = !cache.empty();

  Options idxOptions = options::device(device_).dtype(DType::Int64);
  // token + positional embeddings
  Tensor x;
  if (useCache) {
    auto wteIdx = Tensor(std::vector<int64_t>{static_cast<int64_t>(inputs.back())}, idxOptions);
    auto wpeIdx = Tensor(std::vector<int64_t>{static_cast<int64_t>(inputs.size() - 1)}, idxOptions);
    x = op::indexAdvance(params.wte, ArrayView<Tensor>{wteIdx}) +
        op::indexAdvance(params.wpe, ArrayView<Tensor>{wpeIdx});
  } else {
    cache.resize(params.blocks.size());
    auto wteIdx = Tensor(inputs, idxOptions);
    auto wpeIdx = Tensor::arange<int64_t>(0, static_cast<int64_t>(inputs.size()), 1.f, idxOptions);
    x = op::indexAdvance(params.wte, ArrayView<Tensor>{wteIdx}) +
        op::indexAdvance(params.wpe, ArrayView<Tensor>{wpeIdx});
  }

  // forward pass through n_layer transformer blocks
  for (size_t i = 0; i < params.blocks.size(); i++) {
    x = transformerBlock(x, params.blocks[i], head, cache[i]);
  }

  // projection to vocab
  x = layerNorm(x, params.ln_f.g, params.ln_f.b);
  return op::matmulTrans(x, params.wte, false, true);
}

void Model::generate(std::vector<int64_t>& tokens, const GPT2::Params& params, uint32_t head, uint32_t maxTokens,
                     const std::function<bool(int64_t token)>& callback) {
  tinytorch::NoGradGuard guard;
  std::vector<KVCache> cache;
  // auto-regressive decode loop
  for (uint32_t tokenCnt = 0; tokenCnt < maxTokens; tokenCnt++) {
    // model forward pass
    auto logits = gpt2(tokens, params, head, cache);
    // greedy sampling
    auto nextId = op::argmax(op::index(logits, IntArrayView{-1})).item<int64_t>();
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

bool Model::loadModelGPT2(GPT2& gpt2, const char* hparams, const char* modelDict) {
  FUNCTION_TIMED();
  // load hparams.json
  const auto hparamsJson = FileUtils::parseJson(hparams);
  if (hparamsJson.IsNull() || !hparamsJson.IsObject()) {
    LOGE("parse file failed: %s", hparams);
    return false;
  }
  gpt2.hparams.n_vocab = hparamsJson["n_vocab"].GetInt();
  gpt2.hparams.n_ctx = hparamsJson["n_ctx"].GetInt();
  gpt2.hparams.n_embd = hparamsJson["n_embd"].GetInt();
  gpt2.hparams.n_head = hparamsJson["n_head"].GetInt();
  gpt2.hparams.n_layer = hparamsJson["n_layer"].GetInt();

  // load model_dict.json
  const auto modelDictJson = FileUtils::parseJson(modelDict);
  if (modelDictJson.IsNull() || !modelDictJson.IsObject()) {
    LOGE("parse file failed: %s", modelDict);
    return false;
  }

  // load model_file.data
  if (!modelDictJson.HasMember("file_path") || !modelDictJson["file_path"].IsString()) {
    LOGE("model_dict missing file_path");
    return false;
  }
  std::string dataPath = modelDictJson["file_path"].GetString();
  std::fstream dataFile(dataPath, std::ios::in | std::ios::binary);
  if (!dataFile.is_open()) {
    LOGE("open file failed: %s", dataPath.c_str());
    return false;
  }

  // check file size
  dataFile.seekg(0, std::ios::end);
  auto fileSize = (uint32_t)dataFile.tellg();
  if (!modelDictJson.HasMember("file_size") || !modelDictJson["file_size"].IsInt()) {
    LOGE("model_dict missing file_size");
    return false;
  }
  auto expectSize = modelDictJson["file_size"].GetUint();
  if (fileSize != expectSize) {
    LOGE("model file size invalid! expect: %d, actual: %d", expectSize, fileSize);
    return false;
  }
  dataFile.seekg(0, std::ios::beg);

  if (!modelDictJson.HasMember("model_index") || !modelDictJson["model_index"].IsObject()) {
    LOGE("model_dict missing model_index");
    return false;
  }
  const auto& model = modelDictJson["model_index"];

  loadTensor(gpt2.params.wpe, dataFile, model["wpe"]);
  loadTensor(gpt2.params.wte, dataFile, model["wte"]);
  loadLayerNorm(gpt2.params.ln_f, dataFile, model["ln_f"]);
  gpt2.params.blocks.resize(gpt2.hparams.n_layer);
  if (!model.HasMember("blocks") || !model["blocks"].IsArray()) {
    LOGE("model_index missing blocks");
    return false;
  }
  const auto& blocks = model["blocks"];
  for (uint32_t i = 0; i < gpt2.hparams.n_layer; i++) {
    loadTransformerBlock(gpt2.params.blocks[i], dataFile, blocks[i]);
  }

  return true;
}

void Model::loadTensor(Tensor& ret, std::fstream& fin, const rapidjson::Value& json) {
  ret = Tensor::empty(getShape(json).view(), options::device(DeviceType::CPU));
  uint32_t pos = json["pos"].GetUint();
  fin.seekg(pos, std::ios::beg);
  fin.read((char*)ret.dataPtr<void>(), std::streamsize(sizeof(float) * ret.numel()));

  ret = ret.to(device_);
}

void Model::loadConv1D(Conv1D& ret, std::fstream& fin, const rapidjson::Value& json) {
  loadTensor(ret.w, fin, json["w"]);
  loadTensor(ret.b, fin, json["b"]);
}

void Model::loadLayerNorm(LayerNorm& ret, std::fstream& fin, const rapidjson::Value& json) {
  loadTensor(ret.g, fin, json["g"]);
  loadTensor(ret.b, fin, json["b"]);
}

void Model::loadTransformerBlock(TransformerBlock& ret, std::fstream& fin, const rapidjson::Value& json) {
  loadConv1D(ret.attn.c_attn, fin, json["attn"]["c_attn"]);
  loadConv1D(ret.attn.c_proj, fin, json["attn"]["c_proj"]);
  loadLayerNorm(ret.ln_1, fin, json["ln_1"]);
  loadLayerNorm(ret.ln_2, fin, json["ln_2"]);
  loadConv1D(ret.mlp.c_fc, fin, json["mlp"]["c_fc"]);
  loadConv1D(ret.mlp.c_proj, fin, json["mlp"]["c_proj"]);
}

SizeVector Model::getShape(const rapidjson::Value& json) {
  const auto& items = json["shape"].GetArray();
  SizeVector ret;
  ret.reserve(items.Size());
  for (auto& it : items) {
    ret.pushBack(it.GetInt());
  }
  return ret;
}

}  // namespace tinygpt