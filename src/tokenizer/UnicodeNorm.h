/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Base.h"

namespace tinygpt::tokenizer {

class NFC : public Component {
 public:
  ComponentType getType() override { return ComponentType::NFC; }
  std::string normalize(std::string_view text) override;
};

class NFD : public Component {
 public:
  ComponentType getType() override { return ComponentType::NFD; }
  std::string normalize(std::string_view text) override;
};

class NFKC : public Component {
 public:
  ComponentType getType() override { return ComponentType::NFKC; }
  std::string normalize(std::string_view text) override;
};

class NFKD : public Component {
 public:
  ComponentType getType() override { return ComponentType::NFKD; }
  std::string normalize(std::string_view text) override;
};

}  // namespace tinygpt::tokenizer
