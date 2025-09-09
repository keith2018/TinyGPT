/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "GPTModel.h"

namespace tinygpt {

namespace huggingface::model {
struct QwenConfig;
}  // namespace huggingface::model

namespace qwen3 {
class Qwen3ForCausalLM;
}  // namespace qwen3

class ModelQwen3 : public GPTModel {
 public:
  ModelQwen3(const huggingface::model::QwenConfig &config, tinytorch::Device device);
  ~ModelQwen3() override;

  GPTModelType type() override { return GPTModelType::QWEN3; }

  bool load(const std::string &path) override;
  int64_t numLayers() override;
  int64_t contextSize() override;
  tinytorch::nn::Module &model() override;

 private:
  const huggingface::model::QwenConfig &config_;
  std::unique_ptr<qwen3::Qwen3ForCausalLM> model_;
};

}  // namespace tinygpt
