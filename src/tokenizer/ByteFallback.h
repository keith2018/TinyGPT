/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Base.h"

namespace tinygpt::tokenizer {

class ByteFallback : public Component {
 public:
  ComponentType getType() override { return ComponentType::BYTE_FALLBACK; }
  std::vector<std::string> decode(const std::vector<std::string>& pieces) override;

 private:
  static bool isValidUtf8(const std::vector<uint8_t>& bytes);
};

}  // namespace tinygpt::tokenizer
