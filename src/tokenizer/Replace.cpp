/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Replace.h"

namespace tinygpt::tokenizer {

// Ref: https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/normalizers/replace.rs
Replace::Replace(std::string_view patternStr, std::string_view patternRegex, std::string_view content)
    : patternString_(patternStr), patternRegex_(patternRegex), content_(content) {}

const Regex& Replace::getRegex() const {
  if (!regexInitialized_) {
    if (patternRegex_.empty()) {
      regex_ = std::make_unique<Regex>(Regex::quoteMeta(patternString_));
    } else {
      regex_ = std::make_unique<Regex>(patternRegex_);
    }
    regexInitialized_ = true;
  }
  return *regex_;
}

std::string Replace::replaceAll(std::string_view text) const {
  const Regex& regex = getRegex();
  if (!regex.valid()) {
    return std::string(text);
  }

  std::vector<Range> matches;
  regex.matchAll(matches, text);

  if (matches.empty()) {
    return std::string(text);
  }

  size_t resultLength = text.length();
  for (const auto& range : matches) {
    resultLength -= (range.second - range.first);
  }
  resultLength += matches.size() * content_.length();

  std::string result;
  result.reserve(resultLength);

  size_t lastEnd = 0;
  for (const auto& range : matches) {
    result.append(text.substr(lastEnd, range.first - lastEnd));
    result.append(content_);
    lastEnd = range.second;
  }

  if (lastEnd < text.length()) {
    result.append(text.substr(lastEnd));
  }

  return result;
}

std::string Replace::normalize(std::string_view text) { return replaceAll(text); }

std::vector<std::string> Replace::decode(const std::vector<std::string>& pieces) {
  std::vector<std::string> retPieces;
  retPieces.reserve(pieces.size());

  for (const auto& token : pieces) {
    retPieces.push_back(replaceAll(token));
  }

  return retPieces;
}

}  // namespace tinygpt::tokenizer