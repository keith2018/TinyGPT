/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Split.h"

#include <regex>
#include <thread>

namespace tinygpt::tokenizer {

Split::Split(std::string_view pattern, SplitDelimiterBehavior behavior, bool invert)
    : pattern_(pattern), behavior_(behavior), invert_(invert) {
  auto regexPat = adjustRegexPattern(pattern_);
  matchers_.reserve(NUM_MAX_THREAD);
  for (uint32_t i = 0; i < NUM_MAX_THREAD; i++) {
    matchers_.emplace_back(std::make_unique<re2::RE2>(regexPat));
  }
  patternValid_ = matchers_[0]->ok();

  if (invert) {
    LOGE("error: invert mode not support");
    // TODO support "invert" mode
  }
}

PreTokenizedString Split::preTokenize(std::string_view text) {
  if (!patternValid_) {
    return {};
  }
  const auto tId = std::hash<std::thread::id>{}(std::this_thread::get_id());
  auto &matcher = *matchers_[tId % NUM_MAX_THREAD];

  PreTokenizedString ret;
  ret.backStr = text;
  ret.pieces = split(text, matcher, behavior_);
  return ret;
}

std::string Split::adjustRegexPattern(const std::string &inputRegex) {
  // match \s+(?!\S)
  static std::regex lookaheadPattern(R"(\\s\+\(\?!\\S\))");
  // replace with \s+$
  return std::regex_replace(inputRegex, lookaheadPattern, R"(\s+$)");
}

std::vector<Range> Split::split(std::string_view str, const re2::RE2 &matcher, SplitDelimiterBehavior behavior) {
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

std::vector<Range> Split::match(std::string_view str, const re2::RE2 &matcher) {
  std::vector<Range> matches;
  matches.reserve(str.size());

  re2::StringPiece input(str.data(), str.size());
  re2::StringPiece match;

  size_t matchStart = 0;
  size_t matchEnd = 0;

  while (matcher.Match(input, matchEnd, str.size(), re2::RE2::UNANCHORED, &match, 1)) {
    matchStart = match.data() - str.data();
    matchEnd = matchStart + match.size();
    matches.emplace_back(matchStart, matchEnd);
  }
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
