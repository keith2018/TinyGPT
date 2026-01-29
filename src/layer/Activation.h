/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Functions.h"
#include "Modules.h"

namespace tinytorch::nn {

class SiLUMul : public Module {
 public:
  Tensor forward(const Tensor &x) override { return function::siluMul(x); }
};

}  // namespace tinytorch::nn
