/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "GPTModel.h"

namespace tinygpt {

namespace gpt2 {
struct GPT2Config;
class GPT2LMHeadModel;
}  // namespace gpt2

class ModelGPT2 : public GPTModel {
 public:
  explicit ModelGPT2(tinytorch::Device device);
  ~ModelGPT2() override;

  GPTModelType type() override { return GPTModelType::GPT2; }

  bool load(const std::string &path) override;
  int64_t contextSize() override;
  tinytorch::nn::Module &model() override;

 private:
  std::unique_ptr<gpt2::GPT2Config> config_;
  std::unique_ptr<gpt2::GPT2LMHeadModel> model_;
};

}  // namespace tinygpt
