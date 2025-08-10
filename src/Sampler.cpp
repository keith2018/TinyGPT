/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Sampler.h"

#include "Functions.h"

namespace tinygpt {

using namespace tinytorch;

Tensor Sampler::sample(const Tensor& logits) {
  ASSERT(logits.dim() == 2);  // [batch, vocab_size]

  bool setTemperature = config_.temperature && config_.temperature != 1.f;
  bool setTopK = config_.topK && config_.topK > 0;
  bool setTopP = config_.topP && config_.topP > 0.f;

  if (!setTemperature && !setTopK && !setTopP) {
    // greedy
    return function::argmax(logits, -1, true);
  }

  Tensor l = logits;

  // temperature
  if (setTemperature) {
    l = l / *config_.temperature;
  }

  // top k
  if (setTopK) {
    auto topK = std::min(*config_.topK, logits.size(-1));
    auto [topkLogits, topkIndices] = function::topk(l, topK, -1);

    l.fill_(-std::numeric_limits<float>::infinity());
    l.scatter_(-1, topkIndices, topkLogits);
  }

  // top p
  if (setTopP) {
    auto [sortedLogits, sortedIndices] = function::sort(l, -1, true);
    auto probs = function::softmax(sortedLogits, -1);
    auto cumulativeProbs = function::cumsum(probs, -1);
    auto sortedMask = cumulativeProbs <= *config_.topP;

    auto firstIndices = Tensor::zeros({sortedMask.size(0), 1}, sortedMask.options().indices());
    auto firstMask = function::scatter(Tensor::zerosLike(sortedMask, sortedMask.options()), -1, firstIndices,
                                       Tensor::fullLike(firstIndices, true, sortedMask.options()));
    sortedMask = sortedMask | firstMask;

    // TODO sortedMask.indexPut_({Slice(), 0}, true);

    sortedLogits.fillMasked_(~sortedMask, -std::numeric_limits<float>::infinity());

    l.fill_(-std::numeric_limits<float>::infinity());
    l.scatter_(-1, sortedIndices, sortedLogits);
  }

  // multinomial
  auto probs = function::softmax(l, -1);
  return function::multinomial(probs, 1);
}

}  // namespace tinygpt