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
  float minP;

  // NOLINTNEXTLINE(google-explicit-constructor)
  SamplerConfig(float t = 0.f, int64_t k = 0, float tp = 1.f, float mp = 0.f)
      : temperature(t), topK(k), topP(tp), minP(mp) {}
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
  bool setMinP_;

  bool doSample_;
};

}  // namespace tinygpt
