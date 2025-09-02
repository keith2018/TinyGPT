/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Base.h"
#include "Regex.h"

namespace tinygpt::tokenizer {

class Replace : public Component {
 public:
  Replace(std::string_view patternStr, std::string_view patternRegex, std::string_view content);

  ComponentType getType() override { return ComponentType::REPLACE; }
  std::string normalize(std::string_view text) override;
  std::vector<std::string> decode(const std::vector<std::string>& pieces) override;

 private:
  std::string replaceAll(std::string_view text) const;
  const Regex& getRegex() const;

  std::string patternString_;
  std::string patternRegex_;
  std::string content_;

  mutable std::unique_ptr<Regex> regex_;
  mutable bool regexInitialized_ = false;
};

}  // namespace tinygpt::tokenizer
