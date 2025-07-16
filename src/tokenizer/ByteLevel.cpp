/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ByteLevel.h"

#include "Split.h"

namespace tinygpt::tokenizer {

// Ref https://github.com/openai/gpt-2/blob/master/src/encoder.py
std::vector<char32_t> ByteLevel::bytesToUtf8() {
  std::vector<char32_t> b2u(256, 0);

  std::vector<std::pair<uint8_t, uint8_t>> ranges = {{'!', '~'}, {'\xA1', '\xAC'}, {'\xAE', '\xFF'}};
  for (auto& [start, end] : ranges) {
    for (int16_t i = start; i <= end; i++) {
      b2u[i] = static_cast<char32_t>(i);
    }
  }

  uint8_t n = 0;
  for (int16_t i = 0; i < 256; i++) {
    if (b2u[i] == 0) {
      b2u[i] = static_cast<char32_t>(256 + n);
      n++;
    }
  }
  return b2u;
}

static std::array<std::array<char, 2>, 256> makeByteUtf8Table(const std::vector<char32_t>& bytesChar,
                                                              std::array<uint8_t, 256>& outLen) {
  std::array<std::array<char, 2>, 256> table{};
  for (int16_t i = 0; i < 256; i++) {
    char32_t cp = bytesChar[i];
    if (cp <= 0x7F) {
      table[i][0] = static_cast<char>(cp);
      outLen[i] = 1;
    } else if (cp <= 0x7FF) {
      table[i][0] = static_cast<char>(0xC0 | (cp >> 6));
      table[i][1] = static_cast<char>(0x80 | (cp & 0x3F));
      outLen[i] = 2;
    } else {
      // invalid
      table[i][0] = '?';
      outLen[i] = 1;
    }
  }
  return table;
}

const std::vector<char32_t> ByteLevel::bytesChar_ = bytesToUtf8();
const std::array<uint8_t, 256> ByteLevel::byteUtf8Len_ = [] {
  std::array<uint8_t, 256> len{};
  makeByteUtf8Table(ByteLevel::bytesChar_, len);
  return len;
}();
const std::array<std::array<char, 2>, 256> ByteLevel::byteUtf8Table_ = [] {
  std::array<uint8_t, 256> len{};
  return makeByteUtf8Table(ByteLevel::bytesChar_, len);
}();

std::string ByteLevel::utf8ToBytes(std::string_view str) {
  static ankerl::unordered_dense::map<std::string_view, uint8_t> utf8ToByte;
  static bool inited = false;
  if (!inited) {
    for (int16_t i = 0; i < 256; ++i) {
      const auto len = byteUtf8Len_[i];
      std::string_view key(byteUtf8Table_[i].data(), len);
      utf8ToByte[key] = static_cast<uint8_t>(i);
    }
    inited = true;
  }

  std::string result;
  size_t i = 0;
  while (i < str.size()) {
    bool found = false;
    for (auto len = 1; len <= 2; len++) {
      if (i + len <= str.size()) {
        auto key = str.substr(i, len);
        auto it = utf8ToByte.find(key);
        if (it != utf8ToByte.end()) {
          result.push_back(static_cast<char>(it->second));
          i += len;
          found = true;
          break;
        }
      }
    }
    if (!found) {
      LOGE("Invalid byte level utf8 string %s at position %d", str.data(), i);
    }
  }
  return result;
}

int32_t ByteLevel::findIncompletePos(std::string_view str) {
  const auto len = static_cast<int32_t>(str.size());
  if (len == 0) {
    return -1;
  }

  auto maxCheck = std::min(4, len);
  auto first = len - 1;
  while (first >= 0 && (len - first) <= 4) {
    auto c = static_cast<unsigned char>(str[first]);
    if ((c & 0xC0) != 0x80) {
      int32_t expected;
      if ((c & 0x80) == 0) {
        expected = 1;
      } else if ((c & 0xE0) == 0xC0) {
        expected = 2;
      } else if ((c & 0xF0) == 0xE0) {
        expected = 3;
      } else if ((c & 0xF8) == 0xF0) {
        expected = 4;
      } else {
        return first;
      }
      auto actual = len - first;
      if (actual < expected) {
        return first;
      }
      return -1;
    }
    first--;
  }
  if (maxCheck == len) {
    return 0;
  }
  return -1;
}

std::vector<std::string_view> ByteLevel::splitUTF8(std::string_view str) {
  std::vector<std::string_view> results;
  results.reserve(str.length());

  size_t idx = 0;
  while (idx < str.size()) {
    auto c = static_cast<unsigned char>(str[idx]);
    size_t charLen = 0;
    if ((c & 0x80) == 0) {
      // 1-byte: 0xxxxxxx
      charLen = 1;
    } else if ((c & 0xE0) == 0xC0) {
      // 2-byte: 110xxxxx 10xxxxxx
      if (idx + 1 < str.size() && (static_cast<unsigned char>(str[idx + 1]) & 0xC0) == 0x80) {
        charLen = 2;
      } else {
        // malformed
        LOGE("splitUTF8 error: invalid char: %c", c);
        charLen = 1;
      }
    } else {
      LOGE("splitUTF8 error: invalid char: %c", c);
      // malformed
      charLen = 1;
    }
    results.emplace_back(str.substr(idx, charLen));
    idx += charLen;
  }
  return results;
}

ByteLevel::ByteLevel(bool addPrefixSpace, bool useRegex) : addPrefixSpace_(addPrefixSpace), useRegex_(useRegex) {
  if (useRegex_) {
    // Ref https://github.com/openai/gpt-2/blob/master/src/encoder.py
    static const std::string PATTERN_GPT2 =
        R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
    matcher_ = std::make_unique<Regex>(PATTERN_GPT2);
    ASSERT(matcher_->valid());
  }
}

StringPieces ByteLevel::preTokenize(const StringPieces& text) {
  ASSERT(!text.pieces.empty());
  ASSERT(!text.backStr.empty());

  if (!useRegex_) {
    auto& pieces = text.pieces;
    auto& firstRange = pieces[0];
    if (!addPrefixSpace_ || text.backStr[firstRange.first] == ' ') {
      return byteLevelEncode(&pieces[0], pieces.size(), text.backStr, {});
    } else {
      std::string firstPiece;
      firstPiece.reserve(firstRange.second - firstRange.first + 1);
      firstPiece.push_back(' ');
      firstPiece.append(text.backStr.data() + firstRange.first, firstRange.second - firstRange.first);
      return byteLevelEncode(pieces.size() > 1 ? &pieces[1] : nullptr, pieces.size() - 1, text.backStr, firstPiece);
    }
  } else {
    std::string_view inputView = text.backStr;
    std::string inputWithSpace;
    if (addPrefixSpace_ && inputView[0] != ' ') {
      inputWithSpace.reserve(inputView.size() + 1);
      inputWithSpace.push_back(' ');
      inputWithSpace.append(inputView);
      inputView = inputWithSpace;
    }
    auto pieces = Split::split(inputView, *matcher_, SplitDelimiterBehavior::ISOLATED);
    return byteLevelEncode(&pieces[0], pieces.size(), inputView, {});
  }
}

StringPieces ByteLevel::byteLevelEncode(const Range* pieces, size_t pieceCnt, std::string_view backStr,
                                        std::string_view firstPiece) {
  StringPieces ret;
  ret.pieces.reserve(pieceCnt + (firstPiece.empty() ? 0 : 1));
  ret.backStr.reserve(backStr.size() * 2);

  auto appendPiece = [&](std::string_view sv) {
    const auto pos = ret.backStr.size();
    for (auto c : sv) {
      const auto ch = static_cast<uint8_t>(c);
      ret.backStr.append(byteUtf8Table_[ch].data(), byteUtf8Len_[ch]);
    }
    ret.pieces.emplace_back(pos, ret.backStr.size());
  };

  if (!firstPiece.empty()) {
    appendPiece(firstPiece);
  }

  for (size_t i = 0; i < pieceCnt; i++) {
    auto& r = pieces[i];
    appendPiece(backStr.substr(r.first, r.second - r.first));
  }
  return ret;
}

std::vector<int32_t> ByteLevel::postProcess(const std::vector<int32_t>& ids, bool addSpecialTokens) {
  // 'trimOffsets' not support
  return ids;
}

std::string ByteLevel::decode(const std::vector<std::string>& pieces) {
  std::string ret;
  size_t len = 0;
  for (auto& s : pieces) {
    len += s.size();
  }
  ret.reserve(len);
  for (auto& s : pieces) {
    ret.append(s);
  }
  return ret;
}

}  // namespace tinygpt::tokenizer