/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "GPT2.h"

namespace tinygpt::gpt2 {

Tensor generate(const GPT2Config &config, nn::Module &model, Sampler &sampler, Tensor input_ids,
                int64_t max_new_tokens) {
  model.eval();
  NoGradGuard guard;
  for (auto i = 0; i < max_new_tokens; i++) {
    auto max_len = config.n_positions;
    auto seq_len = input_ids.size(1);
    Tensor input_ids_cond;
    if (seq_len > max_len) {
      input_ids_cond = function::narrow(input_ids, 1, seq_len - max_len, max_len);
    } else {
      input_ids_cond = input_ids;
    }
    auto logits = model(input_ids_cond);
    logits = function::narrow(logits, 1, logits.size(1) - 1, 1).squeeze(1);

    auto next_token = sampler.sample(logits);
    input_ids = function::concat(ArrayView<Tensor>{input_ids, next_token}, 1);
  }
  return input_ids;
}

}  // namespace tinygpt::gpt2