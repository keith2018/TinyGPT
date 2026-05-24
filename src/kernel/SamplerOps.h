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

  float temperature = 1.f;
  int32_t topK = 0;
  float topP = 1.f;
  float minP = 0.f;

  static SamplingParams greedy() {
    SamplingParams p;
    p.temperature = 0.f;
    return p;
  }

  friend bool operator==(const SamplingParams& a, const SamplingParams& b) {
    return a.seed == b.seed && a.temperature == b.temperature && a.topK == b.topK && a.topP == b.topP &&
           a.minP == b.minP;
  }
  friend bool operator!=(const SamplingParams& a, const SamplingParams& b) { return !(a == b); }
};

inline bool isGreedy(const SamplingParams& p) {
  const bool useTemp = p.temperature > 0.f && p.temperature != 1.f;
  const bool useTopK = p.topK > 0;
  const bool useTopP = p.topP < 1.f && p.topP > 0.f;
  const bool useMinP = p.minP > 0.f;
  if (p.temperature <= 0.f) {
    return true;
  }
  return !(useTemp || useTopK || useTopP || useMinP);
}

// logits: [batch, vocab]
tinytorch::Tensor fusedSample(const tinytorch::Tensor& logits, const SamplingParams* paramsHost, int32_t batch,
                              uint64_t globalSeed, uint64_t globalSeq);

tinytorch::Tensor fusedSample(const tinytorch::Tensor& logits, const SamplingParams& params, uint64_t globalSeed,
                              uint64_t globalSeq);

// Graph-capturable variant: reads globalSeq from a device pointer so that the
// value can be updated between CUDA Graph replays without re-capturing.
// output: pre-allocated [batch, 1] Int64 tensor (addresses baked into graph)
void fusedSampleGraphable(const tinytorch::Tensor& logits, tinytorch::Tensor& output, const SamplingParams& params,
                          uint64_t globalSeed, const uint64_t* devGlobalSeqPtr);

}  // namespace tinygpt::kernel
