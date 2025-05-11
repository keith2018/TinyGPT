/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <memory>

#include "Base.h"

namespace tinygpt::tokenizer {

class Regex {
  class Impl;

 public:
  explicit Regex(std::string_view pattern);
  ~Regex();

  bool valid() const;
  void matchAll(std::vector<Range> &ret, std::string_view str) const;
  static std::string quoteMeta(std::string_view unquoted);

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace tinygpt::tokenizer
