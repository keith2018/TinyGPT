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
    std::string pieceText = text.backStr.substr(range.first, range.second - range.first);
    std::string newPieceText;
    for (char c : pieceText) {
      if (c == ' ') {
        newPieceText += replacement_;
      } else {
        newPieceText += c;
      }
    }
    pieceText = newPieceText;

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
      for (size_t i = 0; i <= pieceText.size() - replacement_.size(); i++) {
        if (pieceText.substr(i, replacement_.size()) == replacement_) {
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

  result.backStr = processedText;
  return result;
}

std::vector<std::string> Metaspace::decode(const std::vector<std::string> &pieces) {
  std::vector<std::string> retPieces;
  retPieces.reserve(pieces.size());

  for (size_t i = 0; i < pieces.size(); i++) {
    const auto &token = pieces[i];
    std::string decoded;

    size_t pos = 0;
    while (pos < token.length()) {
      if (pos + replacement_.size() <= token.length() && token.substr(pos, replacement_.size()) == replacement_) {
        if (!(i == 0 && pos == 0 && prependScheme_ != kPrependSchemeNever)) {
          decoded += ' ';
        }
        pos += replacement_.size();
      } else {
        decoded += token[pos];
        pos++;
      }
    }

    retPieces.push_back(decoded);
  }

  return retPieces;
}

}  // namespace tinygpt::tokenizer