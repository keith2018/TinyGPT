/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "ModelConfig.h"
#include "Sampler.h"
#include "model/GPTModel.h"
#include "tokenizer/Tokenizer.h"

namespace tinygpt::huggingface {

struct GPTContext {
  std::unique_ptr<model::ModelConfig> modelConfig;
  std::unique_ptr<model::GenerationConfig> generationConfig;
  std::unique_ptr<tokenizer::Tokenizer> tokenizer;
  std::unique_ptr<GPTModel> model;
};

class ModelLoader {
 public:
  bool load(const std::string &dir, tinytorch::Device device, tinytorch::DType dtype);

  GPTContext &&getContext() { return std::move(context_); }

 private:
  GPTContext context_;
};

}  // namespace tinygpt::huggingface
