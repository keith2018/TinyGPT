/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Sampler.h"

namespace tinygpt {

kernel::SamplingParams toKernelParams(const SamplerConfig& cfg) {
  kernel::SamplingParams p;
  p.temperature = cfg.temperature;
  p.topK = (cfg.topK > 0) ? cfg.topK : 0;
  p.topP = (cfg.topP < 1.f && cfg.topP > 0.f) ? cfg.topP : 1.f;
  p.minP = (cfg.minP > 0.f) ? cfg.minP : 0.f;
  p.seed = cfg.seed;
  return p;
}

Sampler::Sampler(const SamplerConfig& config)
    : config_(config), params_(toKernelParams(config)), doSample_(!kernel::isGreedy(params_)) {}

}  // namespace tinygpt