/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Modules.h"

namespace tinytorch::nn {

template <typename AttentionType, typename MLPType>
class DecoderLayer : public Module {
 public:
  DecoderLayer(AttentionType &&selfAttn, MLPType &&mlp, RMSNorm &&inputLayerNorm, RMSNorm &&postAttnLayerNorm)
      : selfAttn_(std::move(selfAttn)),
        mlp_(std::move(mlp)),
        inputLayerNorm_(std::move(inputLayerNorm)),
        postAttnLayerNorm_(std::move(postAttnLayerNorm)) {
    registerSubModules();
  }

  DecoderLayer(DecoderLayer &&other) noexcept
      : Module(std::move(other)),
        selfAttn_(std::move(other.selfAttn_)),
        mlp_(std::move(other.mlp_)),
        inputLayerNorm_(std::move(other.inputLayerNorm_)),
        postAttnLayerNorm_(std::move(other.postAttnLayerNorm_)) {
    subModules_.clear();
    registerSubModules();
  }

  DecoderLayer(const DecoderLayer &) = delete;
  DecoderLayer &operator=(const DecoderLayer &) = delete;
  DecoderLayer &operator=(DecoderLayer &&) = delete;

  Tensor forward(const Tensor &input) override {
    auto x = input;
    x = x + selfAttn_(inputLayerNorm_(x));
    x = x + mlp_(postAttnLayerNorm_(x));
    return x;
  }

 private:
  void registerSubModules() {
    registerModules({
        {"self_attn", this->selfAttn_},
        {"mlp", this->mlp_},
        {"input_layernorm", this->inputLayerNorm_},
        {"post_attention_layernorm", this->postAttnLayerNorm_},
    });
  }

  AttentionType selfAttn_;
  MLPType mlp_;
  RMSNorm inputLayerNorm_;
  RMSNorm postAttnLayerNorm_;
};

}  // namespace tinytorch::nn
