/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Module.h"

namespace tinygpt {

class SafeTensors {
 public:
  static std::string toTypeString(tinytorch::DType type);
  static tinytorch::DType fromTypeString(const std::string& s);

  static bool save(tinytorch::nn::Module& module, const std::string& path);
  static bool load(tinytorch::nn::Module& module, const std::string& path);
};

}  // namespace tinygpt
