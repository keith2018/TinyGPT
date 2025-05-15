/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Split.h"

namespace tinygpt::tokenizer {

Split::Split(std::string_view pattern, SplitDelimiterBehavior behavior, bool invert)
    : pattern_(pattern), behavior_(behavior), invert_(invert) {
  matcher_ = std::make_unique<Regex>(pattern);
  patternValid_ = matcher_->valid();

  if (invert) {
    LOGE("error: invert mode not support");
    // TODO support "invert" mode
  }
}

StringPieces Split::preTokenize(const StringPieces &text) {
  if (!patternValid_) {
    return {};
  }

  StringPieces ret;
  ret.backStr = text.backStr;
  for (auto &r : text.pieces) {
    auto splits = split({ret.backStr.data() + r.first, r.second - r.first}, *matcher_, behavior_);
    ret.pieces.insert(ret.pieces.end(), splits.begin(), splits.end());
  }
  return ret;
}

std::vector<Range> Split::split(std::string_view str, const Regex &matcher, SplitDelimiterBehavior behavior) {
  std::vector<Range> matches = match(str, matcher);
  std::vector<Range> splits;
  splits.reserve(matches.size());

  switch (behavior) {
    case SplitDelimiterBehavior::REMOVED:
      splitRemoved(splits, matches, str.size());
      break;
    case SplitDelimiterBehavior::ISOLATED:
      splitIsolated(splits, matches, str.size());
      break;
    case SplitDelimiterBehavior::MERGED_WITH_PREVIOUS:
      splitMergePrevious(splits, matches, str.size());
      break;
    case SplitDelimiterBehavior::MERGED_WITH_NEXT:
      splitMergeNext(splits, matches, str.size());
      break;
    case SplitDelimiterBehavior::CONTIGUOUS:
      splitContiguous(splits, matches, str.size());
      break;
    default:
      LOGE("PreTokenizerSplit: behavior not support: %d", behavior);
      break;
  }
  return splits;
}

std::vector<Range> Split::match(std::string_view str, const Regex &matcher) {
  std::vector<Range> matches;
  matches.reserve(str.size() / 2);
  matcher.matchAll(matches, str);
  return matches;
}

void Split::splitRemoved(std::vector<Range> &results, std::vector<Range> &matches, size_t originSize) {
  size_t pos = 0;
  for (auto &match : matches) {
    if (match.first > pos) {
      results.emplace_back(pos, match.first);
    }
    pos = match.second;
  }

  if (originSize > pos) {
    results.emplace_back(pos, originSize);
  }
}

void Split::splitIsolated(std::vector<Range> &results, std::vector<Range> &matches, size_t originSize) {
  size_t pos = 0;
  for (auto &match : matches) {
    if (match.first > pos) {
      results.emplace_back(pos, match.first);
    }
    results.emplace_back(match.first, match.second);
    pos = match.second;
  }

  if (originSize > pos) {
    results.emplace_back(pos, originSize);
  }
}

void Split::splitMergePrevious(std::vector<Range> &results, std::vector<Range> &matches, size_t originSize) {
  size_t pos = 0;
  for (auto &match : matches) {
    if (match.first > pos) {
      results.emplace_back(pos, match.second);
    } else {
      results.emplace_back(match.first, match.second);
    }
    pos = match.second;
  }

  if (originSize > pos) {
    results.emplace_back(pos, originSize);
  }
}

void Split::splitMergeNext(std::vector<Range> &results, std::vector<Range> &matches, size_t originSize) {
  Range lastMatch = {0, 0};
  size_t pos = 0;
  for (auto &match : matches) {
    if (match.first > pos) {
      if (pos == lastMatch.second && !results.empty()) {
        results.back() = {lastMatch.first, match.first};
      } else {
        results.emplace_back(pos, match.first);
      }
    }
    results.emplace_back(match.first, match.second);
    pos = match.second;
    lastMatch = match;
  }

  if (originSize > pos) {
    if (pos == lastMatch.second && !results.empty()) {
      results.back() = {lastMatch.first, originSize};
    } else {
      results.emplace_back(pos, originSize);
    }
  }
}

void Split::splitContiguous(std::vector<Range> &results, std::vector<Range> &matches, size_t originSize) {
  size_t pos = 0;
  for (auto &match : matches) {
    if (match.first > pos) {
      results.emplace_back(pos, match.first);
    }

    if (match.first == pos && !results.empty()) {
      results.back().second = match.second;
    } else {
      results.emplace_back(match.first, match.second);
    }
    pos = match.second;
  }

  if (originSize > pos) {
    results.emplace_back(pos, originSize);
  }
}

}  // namespace tinygpt::tokenizer
