/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cstdint>

#include "Tensor/Tensor.h"

namespace tinygpt::kernel {

inline constexpr int kMaxTopK = 1024;
inline constexpr int kDefaultImplicitTopK = 256;
inline constexpr int64_t kGlobalSeed = -1;

struct SamplingParams {
  int64_t seed = kGlobalSeed;

  float temperature = 0.f;
  int32_t topK = 0;
  float topP = 1.f;
  float minP = 0.f;

  SamplingParams() = default;

  // NOLINTNEXTLINE(google-explicit-constructor)
  SamplingParams(float t, int32_t k = 0, float tp = 1.f, float mp = 0.f, int64_t s = kGlobalSeed)
      : seed(s), temperature(t), topK(k), topP(tp), minP(mp) {}

  static SamplingParams greedy() { return SamplingParams{}; }

  void normalize() {
    if (topK < 0) topK = 0;
    if (topP >= 1.f || topP <= 0.f) topP = 1.f;
    if (minP < 0.f) minP = 0.f;
  }

  friend bool operator==(const SamplingParams& a, const SamplingParams& b) {
    return a.seed == b.seed && a.temperature == b.temperature && a.topK == b.topK && a.topP == b.topP &&
           a.minP == b.minP;
  }
  friend bool operator!=(const SamplingParams& a, const SamplingParams& b) { return !(a == b); }
};

inline bool isGreedy(const SamplingParams& p) {
  if (p.temperature <= 0.f) {
    return true;
  }
  const bool useTemp = p.temperature != 1.f;
  const bool useTopK = p.topK > 0;
  const bool useTopP = p.topP < 1.f && p.topP > 0.f;
  const bool useMinP = p.minP > 0.f;
  return !(useTemp || useTopK || useTopP || useMinP);
}

// logits: [batch, vocab]
tinytorch::Tensor fusedSample(const tinytorch::Tensor& logits, const SamplingParams* paramsHost, int32_t batch,
                              uint64_t globalSeed, uint64_t globalSeq);

tinytorch::Tensor fusedSample(const tinytorch::Tensor& logits, const SamplingParams& params, uint64_t globalSeed,
                              uint64_t globalSeq);

void fusedSampleGraphable(const tinytorch::Tensor& logits, tinytorch::Tensor& output, const SamplingParams& params,
                          uint64_t globalSeed, const uint64_t* devGlobalSeqPtr);

}  // namespace tinygpt::kernel
