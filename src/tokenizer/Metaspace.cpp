/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Metaspace.h"

namespace tinygpt::tokenizer {

// Ref: https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/metaspace.rs
Metaspace::Metaspace(std::string_view replacement, std::string_view prependScheme, bool split)
    : replacement_(replacement), prependScheme_(prependScheme), split_(split) {}

StringPieces Metaspace::preTokenize(const StringPieces &text) {
  StringPieces result;
  std::string processedText;

  for (const auto &range : text.pieces) {
    std::string pieceText;
    pieceText.reserve(range.second - range.first + replacement_.size());
    for (size_t i = range.first; i < range.second; i++) {
      char c = text.backStr[i];
      if (c == ' ') {
        pieceText.append(replacement_);
      } else {
        pieceText.push_back(c);
      }
    }

    if (prependScheme_ == kPrependSchemeAlways) {
      pieceText.insert(0, replacement_);
    } else if (prependScheme_ == kPrependSchemeFirst) {
      if (range.first == 0) {
        pieceText.insert(0, replacement_);
      }
    }

    size_t startPos = processedText.size();
    processedText.append(pieceText);

    if (split_) {
      std::vector<size_t> splitPositions;
      for (size_t i = 0; i + replacement_.size() <= pieceText.size(); i++) {
        if (pieceText.compare(i, replacement_.size(), replacement_) == 0) {
          splitPositions.push_back(startPos + i);
          i += replacement_.size() - 1;
        }
      }

      if (!splitPositions.empty()) {
        size_t lastPos = startPos;
        for (size_t pos : splitPositions) {
          if (pos > lastPos) {
            result.pieces.emplace_back(lastPos, pos + replacement_.size());
          }
          lastPos = pos + replacement_.size();
        }

        if (lastPos < startPos + pieceText.size()) {
          result.pieces.emplace_back(lastPos, startPos + pieceText.size());
        }
      } else {
        result.pieces.emplace_back(startPos, startPos + pieceText.size());
      }
    } else {
      result.pieces.emplace_back(startPos, startPos + pieceText.size());
    }
  }

  result.backStr = std::move(processedText);
  return result;
}

std::vector<std::string> Metaspace::decode(const std::vector<std::string> &pieces) {
  std::vector<std::string> retPieces;
  retPieces.reserve(pieces.size());

  for (size_t i = 0; i < pieces.size(); i++) {
    const auto &token = pieces[i];
    std::string decoded;
    decoded.reserve(token.size());

    size_t pos = 0;
    while (pos < token.size()) {
      if (pos + replacement_.size() <= token.size() && token.compare(pos, replacement_.size(), replacement_) == 0) {
        if (!(i == 0 && pos == 0 && prependScheme_ != kPrependSchemeNever)) {
          decoded.push_back(' ');
        }
        pos += replacement_.size();
      } else {
        decoded.push_back(token[pos]);
        pos++;
      }
    }

    retPieces.push_back(std::move(decoded));
  }

  return retPieces;
}

}  // namespace tinygpt::tokenizer