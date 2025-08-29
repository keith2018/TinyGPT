/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "GPTModel.h"

namespace tinygpt {

namespace huggingface::model {
struct Qwen2Config;
}  // namespace huggingface::model

namespace qwen2 {
class Qwen2ForCausalLM;
}  // namespace qwen2

class ModelQwen2 : public GPTModel {
 public:
  ModelQwen2(const huggingface::model::Qwen2Config &config, tinytorch::Device device);
  ~ModelQwen2() override;

  GPTModelType type() override { return GPTModelType::QWEN2; }

  bool load(const std::string &path) override;
  int64_t numLayers() override;
  tinytorch::nn::Module &model() override;

 private:
  const huggingface::model::Qwen2Config &config_;
  std::unique_ptr<qwen2::Qwen2ForCausalLM> model_;
};

}  // namespace tinygpt
