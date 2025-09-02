/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Base.h"

namespace tinygpt::tokenizer {

constexpr const char *const kPrependSchemeAlways = "always";
constexpr const char *const kPrependSchemeFirst = "first";
constexpr const char *const kPrependSchemeNever = "never";

class Metaspace : public Component {
 public:
  explicit Metaspace(std::string_view replacement, std::string_view prependScheme = kPrependSchemeAlways,
                     bool split = true);

  ComponentType getType() override { return ComponentType::METASPACE; }
  StringPieces preTokenize(const StringPieces &text) override;
  std::vector<std::string> decode(const std::vector<std::string> &pieces) override;

 private:
  std::string replacement_;
  std::string prependScheme_;
  bool split_;
};

}  // namespace tinygpt::tokenizer
