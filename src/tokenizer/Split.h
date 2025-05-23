/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <string_view>
#include <vector>

#include "Base.h"
#include "Regex.h"

namespace tinygpt::tokenizer {

enum class SplitDelimiterBehavior {
  UNKNOWN = 0,
  REMOVED,
  ISOLATED,
  MERGED_WITH_PREVIOUS,
  MERGED_WITH_NEXT,
  CONTIGUOUS
};

class Split : public Component {
 public:
  Split(std::string_view pattern, SplitDelimiterBehavior behavior, bool invert = false);

  ComponentType getType() override { return ComponentType::SPLIT; }

  StringPieces preTokenize(const StringPieces &text) override;

  static std::vector<Range> split(std::string_view str, const Regex &matcher,
                                  SplitDelimiterBehavior behavior = SplitDelimiterBehavior::ISOLATED);

 private:
  static std::vector<Range> match(std::string_view str, const Regex &matcher);

  static void splitRemoved(std::vector<Range> &results, std::vector<Range> &matches, size_t originSize);
  static void splitIsolated(std::vector<Range> &results, std::vector<Range> &matches, size_t originSize);
  static void splitMergePrevious(std::vector<Range> &results, std::vector<Range> &matches, size_t originSize);
  static void splitMergeNext(std::vector<Range> &results, std::vector<Range> &matches, size_t originSize);
  static void splitContiguous(std::vector<Range> &results, std::vector<Range> &matches, size_t originSize);

  std::string pattern_;
  SplitDelimiterBehavior behavior_;
  bool invert_;

  std::unique_ptr<Regex> matcher_;
  bool patternValid_;
};

}  // namespace tinygpt::tokenizer
