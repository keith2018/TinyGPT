/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Base.h"

namespace tinygpt::tokenizer {

class Strip : public Component {
 public:
  Strip(std::string_view content, int32_t start, int32_t stop);

  ComponentType getType() override { return ComponentType::STRIP; }
  std::vector<std::string> decode(const std::vector<std::string> &pieces) override;

 private:
  std::string content_;
  size_t start_;
  size_t stop_;
};

}  // namespace tinygpt::tokenizer
