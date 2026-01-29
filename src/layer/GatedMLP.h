/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Modules.h"
#include "layer/Activation.h"
#include "layer/Linear.h"

namespace tinytorch::nn {

class GatedMLP : public Module {
 public:
  GatedMLP(int64_t inputSize, int64_t outputSize, Options options = {})
      : gateUpProj_(MergedLinear(inputSize, {outputSize, outputSize}, false, options)),
        downProj_(Linear(outputSize, inputSize, false, options)),
        actFn_(SiLUMul()) {
    registerSubModules();
  }

  GatedMLP(GatedMLP &&other) noexcept
      : Module(std::move(other)),
        gateUpProj_(std::move(other.gateUpProj_)),
        downProj_(std::move(other.downProj_)),
        actFn_(std::move(other.actFn_)) {
    subModules_.clear();
    registerSubModules();
  }

  GatedMLP(const GatedMLP &) = delete;
  GatedMLP &operator=(const GatedMLP &) = delete;
  GatedMLP &operator=(GatedMLP &&) = delete;

  Tensor forward(const Tensor &input) override {
    auto x = gateUpProj_(input);
    x = actFn_(x);
    return downProj_(x);
  }

 private:
  void registerSubModules() {
    registerModules({
        {"gate_proj", gateUpProj_.moduleRefs(0)},
        {"up_proj", gateUpProj_.moduleRefs(1)},
        {"down_proj", downProj_},
    });
  }

  MergedLinear gateUpProj_;
  Linear downProj_;
  SiLUMul actFn_;
};

}  // namespace tinytorch::nn
