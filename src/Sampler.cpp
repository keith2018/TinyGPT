/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Sampler.h"

#include "Functions.h"

namespace tt = tinytorch;

namespace tinygpt {

tt::Tensor Sampler::sample(const tt::Tensor& logits) {
  ASSERT(logits.dim() == 2);  // [batch, vocab_size]

  bool setTemperature = config_.temperature && config_.temperature != 1.f;
  bool setTopK = config_.topK && config_.topK > 0;
  bool setTopP = config_.topP && config_.topP > 0.f;

  if (!setTemperature && !setTopK && !setTopP) {
    // greedy
    return tt::function::argmax(logits, -1, true);
  }

  tt::Tensor l = logits;

  // temperature
  if (setTemperature) {
    l = l / *config_.temperature;
  }

  // top k
  if (setTopK) {
    auto topK = std::min(*config_.topK, logits.size(-1));
    auto [topkLogits, topkIndices] = tt::function::topk(l, topK, -1);

    l.fill_(-std::numeric_limits<float>::infinity());
    l.scatter_(-1, topkIndices, topkLogits);
  }

  // top p
  if (setTopP) {
    auto [sortedLogits, sortedIndices] = tt::function::sort(l, -1, true);
    auto probs = tt::function::softmax(sortedLogits, -1);
    auto cumulativeProbs = tt::function::cumsum(probs, -1);
    auto sortedMask = cumulativeProbs <= *config_.topP;

    auto firstIndices = tt::Tensor::zeros({sortedMask.size(0), 1}, sortedMask.options().indices());
    auto firstMask = tt::function::scatter(tt::Tensor::zerosLike(sortedMask, sortedMask.options()), -1, firstIndices,
                                           tt::Tensor::fullLike(firstIndices, true, sortedMask.options()));
    sortedMask = sortedMask | firstMask;

    // TODO sortedMask.indexPut_({Slice(), 0}, true);

    sortedLogits.fillMasked_(~sortedMask, -std::numeric_limits<float>::infinity());

    l.fill_(-std::numeric_limits<float>::infinity());
    l.scatter_(-1, sortedIndices, sortedLogits);
  }

  // multinomial
  auto probs = tt::function::softmax(l, -1);
  return tt::function::multinomial(probs, 1);
}

}  // namespace tinygpt