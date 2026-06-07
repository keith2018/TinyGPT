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

template <typename GateUpProjT = MergedLinear, typename DownProjT = GemvLinear>
class GatedMLPImpl : public Module {
 public:
  GatedMLPImpl(int64_t inputSize, int64_t outputSize, Options options = {})
      : gateUpProj_(GateUpProjT(inputSize, {outputSize, outputSize}, false, options)),
        downProj_(DownProjT(outputSize, inputSize, false, options)),
        actFn_(SiLUMul()) {
    registerSubModules();
  }

  GatedMLPImpl(GatedMLPImpl &&other) noexcept
      : Module(other),
        gateUpProj_(std::move(other.gateUpProj_)),
        downProj_(std::move(other.downProj_)),
        actFn_(std::move(other.actFn_)) {
    subModules_.clear();
    registerSubModules();
  }

  GatedMLPImpl(const GatedMLPImpl &) = delete;
  GatedMLPImpl &operator=(const GatedMLPImpl &) = delete;
  GatedMLPImpl &operator=(GatedMLPImpl &&) = delete;

  Tensor forward(const Tensor &input) override {
    auto x = gateUpProj_(input);
    x = actFn_(x);
    return downProj_(x);
  }

 private:
  void registerSubModules() {
    this->registerModules({
        {"gate_proj", gateUpProj_.moduleRefs(0)},
        {"up_proj", gateUpProj_.moduleRefs(1)},
        {"down_proj", downProj_},
    });
  }

  GateUpProjT gateUpProj_;
  DownProjT downProj_;
  SiLUMul actFn_;
};

using GatedMLP = GatedMLPImpl<MergedLinear, GemvLinear>;

}  // namespace tinytorch::nn
