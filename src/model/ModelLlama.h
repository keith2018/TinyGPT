/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "GPTModel.h"

namespace tinygpt {

namespace huggingface::model {
struct LlamaConfig;
}  // namespace huggingface::model

namespace llama {
class LlamaHeadModel;
}  // namespace llama

class LlamaForCausalLM : public GPTModel {
 public:
  LlamaForCausalLM(const huggingface::model::LlamaConfig &config, tinytorch::Device device);
  ~LlamaForCausalLM() override;

  GPTModelType type() override { return GPTModelType::LLAMA; }

  bool load(const std::string &path) override;
  int64_t numLayers() override;
  tinytorch::nn::Module &model() override;

 private:
  const huggingface::model::LlamaConfig &config_;
  std::unique_ptr<llama::LlamaHeadModel> model_;
};

}  // namespace tinygpt
