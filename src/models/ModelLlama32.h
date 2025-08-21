/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "GPTModel.h"

namespace tinygpt {

namespace llama32 {
struct Llama32Config;
class Llama32LMHeadModel;
}  // namespace llama32

class ModelLlama32 : public GPTModel {
 public:
  ModelLlama32(tinytorch::Device device, GPTModelSize size);
  ~ModelLlama32() override;

  GPTModelType type() override { return GPTModelType::LLAMA32; }

  bool load(const std::string &path) override;
  int64_t contextSize() override;
  tinytorch::nn::Module &model() override;

 private:
  GPTModelSize size_;
  std::unique_ptr<llama32::Llama32Config> config_;
  std::unique_ptr<llama32::Llama32LMHeadModel> model_;
};

}  // namespace tinygpt
