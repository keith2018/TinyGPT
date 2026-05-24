/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "kernel/SamplerOps.h"

namespace tinygpt {

struct SamplerConfig {
  float temperature = 0.f;
  int32_t topK = 0;
  float topP = 1.f;
  float minP = 0.f;

  int64_t seed = -1;

  // NOLINTNEXTLINE(google-explicit-constructor)
  SamplerConfig(float t = 0.f, int32_t k = 0, float tp = 1.f, float mp = 0.f, int64_t s = -1)
      : temperature(t), topK(k), topP(tp), minP(mp), seed(s) {}
};

kernel::SamplingParams toKernelParams(const SamplerConfig& cfg);

class Sampler {
 public:
  explicit Sampler(const SamplerConfig& config);
  ~Sampler() = default;

  const SamplerConfig& config() const { return config_; }
  const kernel::SamplingParams& params() const { return params_; }
  bool doSample() const { return doSample_; }

 private:
  SamplerConfig config_;
  kernel::SamplingParams params_;
  bool doSample_;
};

}  // namespace tinygpt
