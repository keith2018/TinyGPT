/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Modules.h"

namespace tinygpt {

enum class GPTModelType : int8_t {
  UNKNOWN = 0,
  GPT2 = 1,
  LLAMA32 = 2,
};

enum class GPTModelSize : int8_t {
  UNKNOWN = 0,
  SIZE_1B = 2,
  SIZE_3B = 3,
};

inline std::string toString(GPTModelSize size) {
  switch (size) {
    case GPTModelSize::SIZE_1B:
      return "SIZE_1B";
    case GPTModelSize::SIZE_3B:
      return "SIZE_3B";
    default:
      return "UNKNOWN";
  }
}

class GPTModel {
 public:
  virtual ~GPTModel() = default;

  virtual GPTModelType type() { return GPTModelType::UNKNOWN; }
  tinytorch::Tensor forward(const tinytorch::Tensor &input) { return model()(input); }

  virtual bool load(const std::string &path) = 0;
  virtual int64_t contextSize() = 0;
  virtual tinytorch::nn::Module &model() = 0;
};

}  // namespace tinygpt
