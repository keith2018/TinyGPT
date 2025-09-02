/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "GPTModel.h"

namespace tinygpt {

namespace huggingface::model {
struct MistralConfig;
}  // namespace huggingface::model

namespace mistral {
class MistralForCausalLM;
}  // namespace mistral

class ModelMistral : public GPTModel {
 public:
  ModelMistral(const huggingface::model::MistralConfig &config, tinytorch::Device device);
  ~ModelMistral() override;

  GPTModelType type() override { return GPTModelType::MISTRAL; }

  bool load(const std::string &path) override;
  int64_t numLayers() override;
  tinytorch::nn::Module &model() override;

 private:
  const huggingface::model::MistralConfig &config_;
  std::unique_ptr<mistral::MistralForCausalLM> model_;
};

}  // namespace tinygpt
