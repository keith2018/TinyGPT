/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "GPTModel.h"

namespace tinygpt {

namespace huggingface::model {
struct GPT2Config;
}  // namespace huggingface::model

namespace gpt2 {
class GPT2LMHeadModel;
}  // namespace gpt2

class ModelGPT2 : public GPTModel {
 public:
  ModelGPT2(const huggingface::model::GPT2Config &config, tinytorch::Device device);
  ~ModelGPT2() override;

  GPTModelType type() override { return GPTModelType::GPT2; }

  bool load(const std::string &path) override;
  int64_t numLayers() override;
  int64_t contextSize() override;
  tinytorch::nn::Module &model() override;

 private:
  const huggingface::model::GPT2Config &config_;
  std::unique_ptr<gpt2::GPT2LMHeadModel> model_;
};

}  // namespace tinygpt
