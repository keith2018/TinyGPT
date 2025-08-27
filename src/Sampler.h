/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor.h"

namespace tinygpt {

struct SamplerConfig {
  float temperature;
  int64_t topK;
  float topP;

  // NOLINTNEXTLINE(google-explicit-constructor)
  SamplerConfig(float t = 1.f, int64_t k = 0, float p = 0.f) : temperature(t), topK(k), topP(p) {}
};

class Sampler {
 public:
  explicit Sampler(const SamplerConfig& config);
  virtual ~Sampler() = default;

  // logits: [batch, vocab_size]
  virtual tinytorch::Tensor sample(const tinytorch::Tensor& logits);

 protected:
  SamplerConfig config_;

  bool setTemperature_;
  bool setTopK_;
  bool setTopP_;

  bool doSample_;
};

}  // namespace tinygpt
