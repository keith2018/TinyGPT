/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <string_view>
#include <vector>

#include "Base.h"

namespace tinygpt::tokenizer {

class Split : public Component {
 public:
  Split(std::string_view pattern, SplitDelimiterBehavior behavior, bool invert = false);

  ComponentType getType() override { return ComponentType::SPLIT; }

  PreTokenizedString preTokenize(std::string_view text) override;

  static std::string adjustRegexPattern(const std::string &inputRegex);
  static std::vector<Range> split(std::string_view str, const re2::RE2 &matcher,
                                  SplitDelimiterBehavior behavior = SplitDelimiterBehavior::ISOLATED);

 private:
  static std::vector<Range> match(std::string_view str, const re2::RE2 &matcher);

  static void splitRemoved(std::vector<Range> &results, std::vector<Range> &matches, size_t originSize);
  static void splitIsolated(std::vector<Range> &results, std::vector<Range> &matches, size_t originSize);
  static void splitMergePrevious(std::vector<Range> &results, std::vector<Range> &matches, size_t originSize);
  static void splitMergeNext(std::vector<Range> &results, std::vector<Range> &matches, size_t originSize);
  static void splitContiguous(std::vector<Range> &results, std::vector<Range> &matches, size_t originSize);

  std::string pattern_;
  SplitDelimiterBehavior behavior_;
  bool invert_;

  std::vector<std::unique_ptr<re2::RE2>> matchers_;
  bool patternValid_;
};

}  // namespace tinygpt::tokenizer
