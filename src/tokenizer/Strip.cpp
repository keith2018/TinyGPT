/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Strip.h"

namespace tinygpt::tokenizer {

// Ref: https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/decoders/strip.rs
Strip::Strip(std::string_view content, int32_t start, int32_t stop) : content_(content), start_(start), stop_(stop) {
  ASSERT(start >= 0);
  ASSERT(stop >= 0);
  ASSERT(content_.size() == 1);
}

std::vector<std::string> Strip::decode(const std::vector<std::string>& pieces) {
  std::vector<std::string> retPieces;
  retPieces.reserve(pieces.size());
  char stripChar = content_[0];

  for (const auto& piece : pieces) {
    size_t startCut = 0;
    while (startCut < piece.length() && startCut < start_ && piece[startCut] == stripChar) {
      startCut++;
    }

    size_t stopCut = piece.length();
    while (stopCut > startCut && (piece.length() - stopCut) < stop_ && piece[stopCut - 1] == stripChar) {
      stopCut--;
    }

    if (stopCut <= startCut) {
      retPieces.emplace_back("");
    } else {
      retPieces.emplace_back(std::string_view(piece).substr(startCut, stopCut - startCut));
    }
  }
  return retPieces;
}

}  // namespace tinygpt::tokenizer