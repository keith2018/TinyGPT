/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor.h"

namespace tinygpt {

struct SamplerConfig {
  std::optional<float> temperature;
  std::optional<int64_t> topK;
  std::optional<float> topP;

  // NOLINTNEXTLINE(google-explicit-constructor)
  SamplerConfig(std::optional<float> t = std::nullopt, std::optional<int> k = std::nullopt,
                std::optional<float> p = std::nullopt)
      : temperature(t), topK(k), topP(p) {}
};

class Sampler {
 public:
  explicit Sampler(const SamplerConfig& config) : config_(config) {}
  virtual ~Sampler() = default;

  // logits: [batch, vocab_size]
  virtual tinytorch::Tensor sample(const tinytorch::Tensor& logits);

 protected:
  SamplerConfig config_;
};

}  // namespace tinygpt
