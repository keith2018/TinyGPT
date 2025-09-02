/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Base.h"

namespace tinygpt::tokenizer {

class Fuse : public Component {
 public:
  ComponentType getType() override { return ComponentType::FUSE; }
  std::vector<std::string> decode(const std::vector<std::string> &pieces) override;
};

}  // namespace tinygpt::tokenizer
